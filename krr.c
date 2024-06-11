#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define THRESH 0.00001

typedef struct {
  int f1;        // number of features + 1
  int n;         // number of samples in the current process
  int total;     // total number of samples
  int *each;     // number of samples in each process
  int *offsets;  // number of samples up to each process
} size_info;

typedef struct {
  double lambda;
  double s;
  int n_iter;
  double train_rmse;
  double test_rmse;
} result;

// process csv strings
void parse_csv_to_array(char *input, double **array, int *size)
{
  int count, index;
  char *temp, *input_copy, *token;

  // count the number of commas to determine the number of elements
  temp = input;
  count = 1;
  while (*temp) {
    if (*temp == ',') {
      count++;
    }
    temp++;
  }
  *size = count;

  // parse the input string into the array
  *array = (double *)malloc(count * sizeof(double));
  input_copy = strdup(input);
  token = strtok(input_copy, ",");
  index = 0;
  while (token != NULL) {
    (*array)[index++] = atof(token);
    token = strtok(NULL, ",");
  }
  free(input_copy);
}

// process the user input
void process_input(int argc, char *argv[], int rank, char *train_path, char *test_path,
    size_info *train_size, size_info *test_size, double **lambda, int *lambda_size, double **s, int *s_size)
{
  // report proper structure if arguments are missing
  if (argc < 6) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <Train file prefix> <Test file prefix> <Number of features> <lambda values> <s values>.\n", argv[0]);
      fprintf(stderr, "\t\t<lambda values> and <s values> should be comma-separated lists of doubles.\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // allocate the input parameters
  sprintf(train_path, "./data/%s_%d.dat", argv[1], rank);
  sprintf(test_path, "./data/%s_%d.dat", argv[2], rank);
  train_size->f1 = test_size->f1 = atoi(argv[3]) + 1;
  parse_csv_to_array(argv[4], lambda, lambda_size);
  parse_csv_to_array(argv[5], s, s_size);
}

// read the file data
void read_data(char *file_path, size_info *size, double **data)
{
  FILE *file;
  int i, j;
  char ch;

  // open the file for reaching
  file = fopen(file_path, "r");
  if (!file) {
    fprintf(stderr, "Could not open file %s.\n", file_path);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // count the number of samples
  size->n = 0;
  while (!feof(file)) {
    ch = fgetc(file);
    if (ch == '\n') {
      (size->n)++;
    }
  }
  rewind(file);

  // read the data into the array
  *data = (double *)malloc(size->n * size->f1 * sizeof(double));
  for (i = 0; i < size->n; i++) {
    for (j = 0; j < size->f1; j++) {
      fscanf(file, "%lf", &((*data)[i * size->f1 + j]));
      if (j < size->f1 - 1) {
        // skip the comma
        fgetc(file);
      }
    }
  }

  // close the file
  fclose(file);
}

// share sample size information
void collect_size_info(int nprocs, size_info *size)
{
  int i;
  
  // collect each size
  size->each = (int *)malloc(nprocs * sizeof(int));
  MPI_Allgather(&size->n, 1, MPI_INT, size->each, 1, MPI_INT, MPI_COMM_WORLD);

  // compute the remaining information
  size->offsets = (int *)malloc(nprocs * sizeof(int));
  size->total = 0;
  for (i = 0; i < nprocs; i++) {
    size->offsets[i] = size->total;
    size->total += size->each[i];
  }
}

// standardize the data
void standardize_data(double *data, size_info *size)
{
  int n_features, i, j;
  double *local_sums,  *local_sumsq, *global_sums, *global_sumsq, *means, *stddevs;

  // allocate memory and initialize to zeros
  n_features = size->f1 - 1;
  local_sums = (double *)calloc(n_features, sizeof(double));
  local_sumsq = (double *)calloc(n_features, sizeof(double));
  global_sums = (double *)calloc(n_features, sizeof(double));
  global_sumsq = (double *)calloc(n_features, sizeof(double));
  means = (double *)calloc(n_features, sizeof(double));
  stddevs = (double *)calloc(n_features, sizeof(double));

  // compute local sums and sum of squares
  for (i = 0; i < size->n; i++) {
    for (j = 0; j < n_features; j++) {
      local_sums[j] += data[i * size->f1 + j];
      local_sumsq[j] += data[i * size->f1 + j] * data[i * size->f1 + j];
    }
  }

  // compute global sums and sum of squares
  MPI_Allreduce(local_sums, global_sums, n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(local_sumsq, global_sumsq, n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // compute means and standard deviations
  for (j = 0; j < n_features; j++) {
    means[j] = global_sums[j] / size->total;
    stddevs[j] = sqrt(global_sumsq[j] / size->total - means[j] * means[j]);
  }

  // standardize the data
  for (i = 0; i < size->n; i++) {
    for (j = 0; j < n_features; j++) {
      data[i * size->f1 + j] = (data[i * size->f1 + j] - means[j]) / stddevs[j];
    }
  }

  // free memory
  free(local_sums);
  free(local_sumsq);
  free(global_sums);
  free(global_sumsq);
  free(means);
  free(stddevs);
}

// compute the rbf kernel of two feature vectors
void compute_kernel(double *x1, double *x2, size_info *size, double s, double *result)
{
  int i;
  double diff, distance_squared;

  // distance squared
  distance_squared = 0.;
  for (i = 0; i < size->f1 - 1; i++) {
    diff = x1[i] - x2[i];
    distance_squared += diff * diff;
  }

  // radial basis function
  *result = exp(-distance_squared / (2 * s * s));
}

// compute the a block of the kernel matrix
void compute_kernel_block(double *local_data, size_info *size,
    double *foreign_data, int foreign_n, int foreign_offset, double s, double **matrix)
{
  int i, j, index;
  double *x1, *x2;

  // compute the block at the right offset
  for (i = 0; i < size->n; i++) {
    for (j = 0; j < foreign_n; j++) {
      index = i * size->total + foreign_offset + j;
      x1 = &local_data[i * size->f1];
      x2 = &foreign_data[j * size->f1];
      compute_kernel(x1, x2, size, s, &(*matrix)[index]);
    }
  }
}

// compute the complete kernel matrix
void compute_kernel_matrix(double *local_data, size_info *size, int nprocs,
    int rank, double s, double **matrix)
{
  int send_to, recv_from, recv_count, i;
  double *foreign_data;
  MPI_Request req[2];

  // allocate memory for the local part of the kernel matrix
  *matrix = (double *)malloc(size->n * size->total * sizeof(double));

  // compute the diagonal blocks locally
  compute_kernel_block(local_data, size, local_data, size->n, size->offsets[rank], s, matrix);

  // compute the other blocks of the matrix one by one
  for (i = 1; i < nprocs; i++) {
    send_to = (rank - i + nprocs) % nprocs;
    recv_from = (rank + i) % nprocs;
    recv_count = size->each[recv_from] * size->f1;
    foreign_data = (double *)malloc(recv_count * sizeof(double));
    MPI_Isend(local_data, size->n * size->f1, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(foreign_data, recv_count, MPI_DOUBLE, recv_from, MPI_ANY_TAG, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    compute_kernel_block(local_data, size, foreign_data, size->each[recv_from], size->offsets[recv_from], s, matrix);
    free(foreign_data);
  }
}

// extract labels from data
void extract_labels(double *local_data, size_info *size, double **labels)
{
  int i;

  // extract labels
  *labels = (double *)malloc(size->n * sizeof(double));
  for (i = 0; i < size->n; i++) {
    (*labels)[i] = local_data[i * size->f1 + (size->f1 - 1)];
  }
}

// add ridge parameter to the diagonal of the kernel matrix
void add_ridge_parameter(double lambda, size_info *size, int rank, double **matrix)
{
  int i;

  // add ridge parameter to the diagonal
  for (i = 0; i < size->n; i++) {
    (*matrix)[i * size->total + size->offsets[rank] + i] += lambda;
  }
}

// distributed matrix-vector multiplication
void mv_prod(double *matrix, double *vector, size_info *size, double **result)
{
  int i, j;
  double *complete_vector;

  // collect complete vector
  complete_vector = (double *)malloc(size->total * sizeof(double));
  MPI_Allgatherv(vector, size->n, MPI_DOUBLE, complete_vector, size->each,
      size->offsets, MPI_DOUBLE, MPI_COMM_WORLD);
  
  // compute the local part of the matrix-vector product
  *result = (double *)malloc(size->n * sizeof(double));
  for (i = 0; i < size->n; i++) {
    (*result)[i] = 0.;
    for (j = 0; j < size->total; j++) {
      (*result)[i] += matrix[i * size->total + j] * complete_vector[j];
    }
  }

  // free memory
  free(complete_vector);
}

// distributed inner product computation
void inner_product(double *x1, double *x2, size_info *size, double *result)
{
  int i;
  double local_part;

  // compute the local part
  local_part = 0.;
  for (i = 0; i < size->n; i++) {
    local_part += x1[i] * x2[i];
  }

  // sum the parts together
  MPI_Allreduce(&local_part, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// conjugate gradient algorithm
void cg(double *A, double *y, size_info *size, int rank, double **alpha, result *current_result)
{
  int i;
  double *A_alpha, *r, *p, r_r, E, *A_p, p_A_p, q, new_r_r, beta;

  // construct the initial guess
  *alpha = (double *)malloc(size->n * sizeof(double));
  for (i = 0; i < size->n; i++) {
    (*alpha)[i] = 0.;
  }

  // compute the initial residual
  mv_prod(A, *alpha, size, &A_alpha);
  r = (double *)malloc(size->n * sizeof(double));
  for (i = 0; i < size->n; i++) {
    r[i] = y[i] - A_alpha[i];
  }

  // compute the initial descent direction
  p = (double *)malloc(size->n * sizeof(double));
  for (i = 0; i < size->n; i++) {
    p[i] = r[i];
  }

  // compute the initial norm and norm squared of the residual
  inner_product(r, r, size, &r_r);
  E = sqrt(r_r);
  current_result->n_iter = 0;
  if (rank == 0) {
    printf("Iter 0: Residual = %lf\n", E);
  }

  // execute the iterative scheme
  while (E > THRESH) {
    mv_prod(A, p, size, &A_p);
    inner_product(p, A_p, size, &p_A_p);
    q = r_r / p_A_p;
    for (i = 0; i < size->n; i++) {
      (*alpha)[i] += q * p[i];
      r[i] -= q * A_p[i];
    }
    inner_product(r, r, size, &new_r_r);
    E = sqrt(new_r_r);
    beta = new_r_r / r_r;
    for (i = 0; i < size->n; i++) {
      p[i] = r[i] + beta * p[i];
    }
    r_r = new_r_r;
    (current_result->n_iter)++;
    if (rank == 0) {
      printf("Iter %d: Residual = %lf\n", current_result->n_iter, E);
    }
  }

  // free memory
  free(A_alpha);
  free(r);
  free(p);
  free(A_p);
}

// find root from global index
void find_root(int nprocs, int index, size_info *test_size, int *root)
{
  int i;

  // find the rank of the root
  for (i = 0; i < nprocs; i++) {
    if (test_size->offsets[i] <= index && index < test_size->offsets[i] + test_size->each[i]) {
      *root = i;
      break;
    }
  }
}

// compute the complete rmse
void compute_rmse(double *train_data, size_info *train_size, double *test_data, size_info *test_size,
    int nprocs, int rank, double s, double *alpha, result *current_result, char train_or_test)
{
  int i, j, root;
  double *x, local_part_f, kernel_value, f, diff, square_diff_sum;

  // compute the local sum of square differences
  square_diff_sum = 0.;
  for (i = 0; i < test_size->total; i++) {
    // broadcast test vector to all processes
    find_root(nprocs, i, test_size, &root);
    if (rank == root) {
      x = &test_data[(i - test_size->offsets[rank]) * test_size->f1];
      MPI_Bcast(x, test_size->f1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    } else {
      x = malloc(test_size->f1 * sizeof(double));
      MPI_Bcast(x, test_size->f1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }

    // compute local part
    local_part_f = 0.;
    for (j = 0; j < train_size->n; j++) {
      compute_kernel(x, &train_data[j * train_size->f1], train_size, s, &kernel_value);
      local_part_f += alpha[j] * kernel_value;
    }

    // compute prediction
    MPI_Reduce(&local_part_f, &f, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    // compute squared difference
    if (rank == root) {
      diff = x[test_size->f1 - 1] - f;
      square_diff_sum += diff * diff;
    }
  }

  // compute the rmse
  if (train_or_test == 'a') {
    MPI_Reduce(&square_diff_sum, &current_result->train_rmse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      current_result->train_rmse = sqrt(current_result->train_rmse / test_size->total);
    }
  } else if (train_or_test == 'b') {
    MPI_Reduce(&square_diff_sum, &current_result->test_rmse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      current_result->test_rmse = sqrt(current_result->test_rmse / test_size->total);
    }
  }
}

// main function
int main(int argc, char *argv[])
{
  int nprocs, rank, lambda_size, s_size, i, j;
  char train_path[100], test_path[100];
  size_info train_size, test_size;
  double *lambda, *s, *train_data, *test_data, *labels, *matrix, *alpha;
  result *results, *current_result;

  // process input and collect size info
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_input(argc, argv, rank, train_path, test_path, &train_size, &test_size, &lambda, &lambda_size, &s, &s_size);
  read_data(train_path, &train_size, &train_data);
  read_data(test_path, &test_size, &test_data);
  collect_size_info(nprocs, &train_size);
  collect_size_info(nprocs, &test_size);
  standardize_data(train_data, &train_size);
  standardize_data(test_data, &test_size);
  extract_labels(train_data, &train_size, &labels);

  // obtain results
  results = (result *)malloc(lambda_size * s_size * sizeof(result));
  for (i = 0; i < lambda_size; i++) {
    for (j = 0; j < s_size; j++) {
      // current result
      current_result = &results[i * lambda_size + j];
      current_result->lambda = lambda[i];
      current_result->s = s[j];

      // train the model
      compute_kernel_matrix(train_data, &train_size, nprocs, rank, s[j], &matrix);
      add_ridge_parameter(lambda[i], &train_size, rank, &matrix);
      cg(matrix, labels, &train_size, rank, &alpha, current_result);
      
      // test the model
      compute_rmse(train_data, &train_size, train_data, &train_size, nprocs, rank, s[j], alpha, current_result, 'a');
      compute_rmse(train_data, &train_size, test_data, &test_size, nprocs, rank, s[j], alpha, current_result, 'b');

      // print result during computations
      if (rank == 0) {
        printf("lambda = %lf\ts = %lf\tn_iter = %d\ttrain_rmse = %lf\ttest_rmse = %lf\n",
            lambda[i], s[j], current_result->n_iter, current_result->train_rmse, current_result->test_rmse);
      }
    }
  }

  // present results together
  if (rank == 0) {
    printf("== FINAL RESULTS ==\n");
    for (i = 0; i < lambda_size; i++) {
      for (j = 0; j < s_size; j++) {
        printf("lambda = %lf\ts = %lf\tn_iter = %d\ttrain_rmse = %lf\ttest_rmse = %lf\n",
            lambda[i], s[j], results[i * lambda_size + j].n_iter, results[i * lambda_size + j].train_rmse, results[i * lambda_size + j].test_rmse);
      }
    }
  }
  
  // clean up
  free(train_data);
  free(test_data);
  free(labels);
  free(matrix);
  free(alpha);
  free(lambda);
  free(s);
  free(results);
  MPI_Finalize();

  // exit
  return EXIT_SUCCESS;
}
