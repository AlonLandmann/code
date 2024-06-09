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

// process the user input
void process_input(int argc, char *argv[], int rank, size_info *size,
    char *file_path, double *lambda, double *s)
{
  // report proper structure if arguments are missing
  if (argc < 5) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <Number of Features> <File Name Prefix> <lambda> <s>\n", argv[0]);
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // allocate the input parameters
  size->f1 = atoi(argv[1]) + 1;
  sprintf(file_path, "./data/%s_%d.dat", argv[2], rank);
  *lambda = atof(argv[3]);
  *s = atof(argv[4]);
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
        fgetc(file); // skip the comma
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

// compute the rbf kernel of two feature vectors
void compute_kernel(double *x1, double *x2, size_info *size, double s, double *result)
{
  int i;

  // distance squared
  double diff, distance_squared = 0.;
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

  // free the assembled memory
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
void cg(double *A, double *y, size_info *size, int rank, double **alpha)
{
  int i, iter;
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
  iter = 0;
  if (rank == 0) {
    printf("Iter %d: Residual = %lf\n", iter, E);
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
    iter++;
    if (rank == 0) {
      printf("Iter %d: Residual = %lf\n", iter, E);
    }
  }
}

// main function
int main(int argc, char *argv[])
{
  int nprocs, rank;
  size_info size;
  char file_path[100];
  double lambda, s, *local_data, *matrix, *labels, *alpha;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  process_input(argc, argv, rank, &size, file_path, &lambda, &s);
  read_data(file_path, &size, &local_data);
  collect_size_info(nprocs, &size);
  compute_kernel_matrix(local_data, &size, nprocs, rank, s, &matrix);
  extract_labels(local_data, &size, &labels);
  add_ridge_parameter(lambda, &size, rank, &matrix);
  cg(matrix, labels, &size, rank, &alpha);

  // ...
  // double result;
  // inner_product(labels, labels, &size, &result);
  // printf("Process: %d - Result: %lf\n", rank, result);

  // double *result;
  // mv_prod(matrix, labels, &size, &result);
  // printf("result\n");
  // for (int i = 0; i < size.n; i++) {
  //   printf("%lf\n", result[i]);
  // }

  // if (rank == 5) {
  //   int i, j;
  //   printf("Process: %d\n", rank);
  //   for (i = 0; i < size.n; i++) {
  //     for (j = 0; j < size.total; j++) {
  //       printf("%lf  ", matrix[i * size.total + j]);
  //     }
  //     printf("\n");
  //   }
  // }
 
  // printf("process: %d, n: %d, total: %d, each[rank]: %d, offsets[rank]: %d\n",
  //     rank, size.n, size.total, size.each[rank], size.offsets[rank]);

  // if (rank == 0) {
  //   int i, j;
  //   for (i = 0; i < 20; i++) {
  //     for (j = 0; j < size.f1; j++) {
  //       printf("%lf\t", local_data[i * size.f1 + j]);
  //     }
  //     printf("\n");
  //   }
  // }
  // ...


  // XXX CLEAN UP
  MPI_Finalize();

  return EXIT_SUCCESS;
}
