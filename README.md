# K-means clustering algorithm parallelization with OpenMP and MPI (hybrid)
- kmeans.cpp - naive non-parallel algorithm
- kmeans_parallel.cpp - hybrid parallelization with OpenMP and MPI

## Data
Extended Iris flower data set (3000 samples). Simple concatenation of 20 original data sets to better demonstrate the parallelization effect.

## How to compile
```mpic++ -fopenmp -o kmeans_parallel kmeans_parallel.cpp```

## How to run
```mpiexec -n 2 ./kmeans_parallel```,

where -n 2 - number of MPI processes.
Number of OpenMP threads is hardcoded in kmeans_parallel.cpp
