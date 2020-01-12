# K-means clustering algorithm parallelisation with OpenMP and MPI (hybrid)
- kmeans.cpp - naive non-parallel realisation
- kmeans_parallel.cpp - hybrid parallelisation with OpenMP and MPI

## How to compile
```mpic++ -fopenmp -o kmeans_parallel kmeans_parallel.cpp```

## How to run
```mpiexec -n 2 ./kmeans_parallel```
where -n 2 - number of MPI processes.
Number of OpenMP threads is hardcoded in kmeans_parallel.cpp
