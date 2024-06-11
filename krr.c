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

// compute the rbf kernel of two feature vectors
double compute_kernel(double *x1, double *x2, size_info *size, double s)
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
  return exp(-distance_squared / (2 * s * s));
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
      (*matrix)[index] = compute_kernel(x1, x2, size, s);
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

  // compute the diagonal blocks locally (matrix memory is pre-allocated)
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
void mv_product(double *matrix, double *vector, size_info *size, double **result)
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
double inner_product(double *x1, double *x2, size_info *size)
{
  int i;
  double local_part, result;

  // compute the local part
  local_part = 0.;
  for (i = 0; i < size->n; i++) {
    local_part += x1[i] * x2[i];
  }

  // sum the parts together
  MPI_Allreduce(&local_part, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return result;
}

// conjugate gradient algorithm
void cg(double *A, double *y, size_info *size, int rank, double **alpha, result *current_result)
{
  int i;
  double *A_alpha, *r, *p, r_r, E, *A_p, q, new_r_r, beta;

  // construct the initial guess (alpha memory is pre-allocated)
  for (i = 0; i < size->n; i++) {
    (*alpha)[i] = 0.;
  }

  // compute the initial residual
  mv_product(A, *alpha, size, &A_alpha);
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
  r_r = inner_product(r, r, size);
  E = sqrt(r_r);
  current_result->n_iter = 0;
  if (rank == 0) {
    printf("Iter 0: Residual = %lf\n", E);
  }

  // execute the iterative scheme
  while (E > THRESH) {
    mv_product(A, p, size, &A_p);
    q = r_r / inner_product(p, A_p, size);
    for (i = 0; i < size->n; i++) {
      (*alpha)[i] += q * p[i];
      r[i] -= q * A_p[i];
    }
    new_r_r = inner_product(r, r, size);
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

// compute root mean squared error
void compute_rmse(double *train_data, size_info *train_size, double *test_data, size_info *test_size,
    int nprocs, int rank, double s, double *alpha, result *current_result, char train_or_test)
{
  int i, j, k, m;
  double *foreign_data, *f_parts, *f_totals, *labels, diff, square_diff_sum;

  for (i = 0; i < nprocs; i++) {
    // broadcast the m test samples located in process i
    m = test_size->each[i];
    foreign_data = (double *)malloc(m * test_size->f1 * sizeof(double));
    if (rank == i) {
      MPI_Bcast(test_data, m * test_size->f1, MPI_DOUBLE, i, MPI_COMM_WORLD);
      memcpy(foreign_data, test_data, m * test_size->f1 * sizeof(double));
    } else {
      MPI_Bcast(foreign_data, m * test_size->f1, MPI_DOUBLE, i, MPI_COMM_WORLD);
    }

    // compute local parts of the prediction values
    f_parts = (double *)calloc(m, sizeof(double));
    for (j = 0; j < m; j++) {
      for (k = 0; k < train_size->n; k++) {
        f_parts[j] += alpha[k] * compute_kernel(&foreign_data[j * test_size->f1], &train_data[k * train_size->f1], train_size, s);
      }
    }
    free(foreign_data);

    // compute the total prediction values in the root process
    f_totals = (double *)malloc(m * sizeof(double));
    MPI_Reduce(f_parts, f_totals, m, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
    free(f_parts);

    // compute square diff sum in the root process
    if (rank == i) {
      extract_labels(test_data, test_size, &labels);
      square_diff_sum = 0.;
      for (j = 0; j < m; j++) {
        diff = labels[j] - f_totals[j];
        square_diff_sum += diff * diff;
      }
      free(labels);
    }
    free(f_totals);
  }

  // compute rmse
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

  // allocate memory
  results = (result *)malloc(lambda_size * s_size * sizeof(result));
  matrix = (double *)malloc(train_size.n * train_size.total * sizeof(double));
  alpha = (double *)malloc(train_size.n * sizeof(double));

  // obtain results
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

  // print final results
  if (rank == 0) {
    printf("\n== FINAL RESULTS ==\n");
    for (i = 0; i < lambda_size; i++) {
      for (j = 0; j < s_size; j++) {
        printf("lambda = %lf\ts = %lf\tn_iter = %d\ttrain_rmse = %lf\ttest_rmse = %lf\n",
            lambda[i], s[j], results[i * lambda_size + j].n_iter, results[i * lambda_size + j].train_rmse, results[i * lambda_size + j].test_rmse);
      }
    }
  }
  
  // free memory
  free(train_data);
  free(test_data);
  free(labels);
  free(lambda);
  free(s);
  free(matrix);
  free(alpha);
  free(results);

  // finalize and exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}
