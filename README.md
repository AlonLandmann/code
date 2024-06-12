## General
The code in the `krr.c` file can be used to perform a kernel ridge regression on the data located in `data/cal_housing.dat`.
The bash script `split_data.sh` was used to split the data randomly into 70% for training and 30% for testing, and to further split the data evenly into 6 pieces each for parallel processing.
mpich is used for parallel computing.

## Usage
To recompile run
```
mpicc -o krr -Wall -O3 krr.c -lm
```

To use the code run
```
mpirun -np 6 ./krr train test 8 <lambda values> <s values>
```

The lambda and s values are the hyper-parameters used in the regression. Provide a range of real values separated by commas, as illustrated below to perform the corresponding grid search. The command
```
mpirun -np 6 ./krr train test 8 "0.01,0.05,0.1,0.5,1" "1,3,5"
```
would train and test the 15 different combinations indicated by the two ranges.
