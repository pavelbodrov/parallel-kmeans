#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <omp.h>
#include <random>
#include <mpi.h>
#include <map>
#include <ctime>

using namespace std;

double** make_matrix(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

// Read csv file into 2d array and return its shape
map<string, int> read_csv(double** &data, const string &filename) {
    string line, column, token;
    ifstream f(filename);
    int row_cnt = 0;
    int col_cnt = 0;
    map<string, int> shape;

    if (f.is_open()) {
        while (getline(f, line)) {
            if (row_cnt == 0) {
                stringstream ss(line);
                while (getline(ss, token, ',')) {
                    col_cnt += 1;
                }
            }
            row_cnt += 1;
        }

        f.clear();
        f.seekg(0, ios::beg);

        data = make_matrix(row_cnt, col_cnt);

        for (int row = 0; row < row_cnt; row++) {
            getline(f, line);
            stringstream ss(line);
            for (int col = 0; col < col_cnt; col++) {
                getline(ss, token, ',');
                data[row][col] = stod(token);
            }
        }
        f.close();
    }

    shape.insert(pair<string, int>("rows", row_cnt));
    shape.insert(pair<string, int>("cols", col_cnt));
    return shape;
}

// Calculate euclidean distance between 2 arrays
double euclidean_distance(double *p, double *q, int size) {

    double squared_sum = 0;

    for (int i=0; i < size; i++) {
        squared_sum += pow(q[i] - p[i], 2);
    }
    return sqrt(squared_sum);
}

// Function for random initialization of centroids. Just pick N random samples as N centroids
void init_centroids(double** &X, double** &centroids, int n_centroids, int n_samples, int n_features) {
    int random_idx = 0;
    random_device rd;
    default_random_engine random_engine(rd());
    uniform_int_distribution<int> uniform_dist(0, n_samples - 1);

    for (int i=0; i < n_centroids; i++) {
        random_idx = uniform_dist(random_engine);
        for (int j=0;j<n_features;j++) {
            centroids[i][j] = X[random_idx][j];
        }
    }
}

int main(int argc, char*argv[]) {

    double** data = nullptr;
    double** X_all = nullptr;

    map<string, int> shape;
    int nearest_centroid, n_samples, n_samples_all = 0, n_features = 0, n_clusters = 3, n_iter=0;
    double min_dist, curr_dist, dist_sum = 0, dist_sum_new = 0;
    double tol = 0.1;

    // MPI rank and size
    int rank, size;
    // Real labels
    int *y = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read csv file in root process
    if (rank == 0) {
        shape = read_csv(data, "iris_encoded_big.csv");
        n_samples_all = shape.at("rows");
        n_features = shape.at("cols") - 1;

        y = new int[n_samples_all];

        X_all = make_matrix(n_samples_all, n_features);

        for (int i=0; i < n_samples_all; i++) {
            for (int j=0; j < n_features; j++) {
                // Features
                X_all[i][j] = data[i][j];
            }
            // Cluster labels
            y[i] = (int) data[i][n_features];
        }
    }
    else {
        // MPI_Scatter needs to access *X_all in all processes.
        double* dummy_ptr= nullptr;
        X_all = &dummy_ptr;
    }

    double start_time = 0.0;
    // Start timer in root process
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Send basic information (size of matrix, number of clusters) from root to other processes
    MPI_Bcast(&n_samples_all,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n_features,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n_clusters,1,MPI_INT,0,MPI_COMM_WORLD);

    // Distribute the load equally in the root process (rank=0). This information is redundant to non root processes.
    // sendcounts and displs are significant only at root, for other processes let it be null.
    int *sendcounts = nullptr;
    int *displs = nullptr;

    if (rank == 0) {

        int rem = n_samples_all % size;
        int offset = 0;

        sendcounts = new int[size];
        displs = new int[size];

        for (int i = 0; i < size; i++) {
            sendcounts[i] = n_samples_all / size;
            if (rem > 0) {
                sendcounts[i]++;
                rem--;
            }
            sendcounts[i] = sendcounts[i] * n_features;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Each process needs to know number of rows to work with.
    // sendcount is a buffer for storing value equal to sendcounts[rank] from root process, which is n_samples*n_features
    int sendcount = 0;

    // Root process should send to other processes their portion of work (number of rows).
    MPI_Scatter(sendcounts, 1, MPI_INT, &sendcount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Number of samples (rows in the matrix/table) for each process.
    n_samples = sendcount/n_features;

    // Local matrix for each process
    double **X;
    X = make_matrix(n_samples, n_features);

    // Distribute the load equally
    MPI_Scatterv(*X_all, sendcounts, displs, MPI_DOUBLE, *X, n_samples*n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double **centroids = make_matrix(n_clusters, n_features);
    double **new_centroids = make_matrix(n_clusters, n_features);
    int cluster_sizes[n_clusters];

    // Random initialization of centroids in root process
    if (rank == 0) {
        init_centroids(X_all, centroids, n_clusters, n_samples_all, n_features);
        for (int i=0; i<n_clusters; i++) {
            for (int j=0; j<n_features; j++) {
                cluster_sizes[i] = 0;
                new_centroids[i][j] = 0;
            }
        }
    }
    // Zero initialization in other processes
    else {
        for (int i=0; i<n_clusters; i++) {
            for (int j=0; j<n_features; j++) {
                cluster_sizes[i] = 0;
                centroids[i][j] = 0;
                new_centroids[i][j] = 0;
            }
        }
    }

    // Send randomly initialized centroids from root to others
    MPI_Bcast(*centroids, n_clusters*n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int cluster = 0;
    int *labels = new int[n_samples];

    do {
        n_iter++;
        dist_sum = dist_sum_new;
        dist_sum_new = 0;

        // Calculate distances
        #pragma omp parallel num_threads(2) private(nearest_centroid, min_dist, curr_dist) reduction(+:dist_sum_new)
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < n_samples; i++) {
                nearest_centroid = 0;
                min_dist = euclidean_distance(X[i], centroids[nearest_centroid], n_features);
                for (int j = 1; j < n_clusters; j++) {
                    curr_dist = euclidean_distance(X[i], centroids[j], n_features);
                    if (curr_dist < min_dist) {
                        nearest_centroid = j;
                        min_dist = curr_dist;
                    }
                }
                labels[i] = nearest_centroid;
                dist_sum_new += min_dist;
            }
        }

        // Update centroids

        // As the total number of samples in each cluster is not known yet,
        // here we are just calculating the sum, not the mean.
        for (int i = 0; i < n_samples; i++) {
            cluster = labels[i];
            cluster_sizes[cluster]++;
            for (int j=0; j < n_features; j++) {
                new_centroids[cluster][j] += X[i][j];
            }
        }

        // All processes need to synchronize centroids information for centroids update
        MPI_Allreduce(MPI_IN_PLACE, *new_centroids, n_clusters*n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, cluster_sizes, n_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int i=0; i < n_clusters; i++) {
            for (int j=0; j < n_features; j++) {
                if (cluster_sizes[i] > 0) {
                    // Convert sum to mean
                    centroids[i][j] = new_centroids[i][j] / cluster_sizes[i];
                }
                new_centroids[i][j] = 0.0; // Fill with zeros for the next iteration
            }
            cluster_sizes[i] = 0; // Fill with zeros for the next iteration
        }

        // To test convergence, we need the global sum of distances
        MPI_Allreduce(MPI_IN_PLACE, &dist_sum_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Print epoch and inertia info from the root process
        if (rank == 0) {
            printf("Epoch %d, Inertia=%f, Inertia diff=%f\n", n_iter, dist_sum_new, abs(dist_sum - dist_sum_new));
        }
    }
//    while ((abs(dist_sum - dist_sum_new) > tol) && n_iter < 10);
    while (n_iter < 100);

    // Gather results (labels) in root process
    int *labels_all = nullptr;
    int *recvcounts_gather = nullptr;
    int *displs_gather = nullptr;

    if (rank == 0) {
        labels_all = new int[n_samples_all];
        recvcounts_gather = new int[size];
        displs_gather = new int[size];

        for (int i = 0; i < size; i++) {
            recvcounts_gather[i] = sendcounts[i] / n_features;
            displs_gather[i] = displs[i] / n_features;
        }
    }

    MPI_Gatherv(labels, n_samples, MPI_INT, labels_all, recvcounts_gather, displs_gather, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Stop timer
        double elapsed_time = MPI_Wtime() - start_time;

        cout << "Results (predicted label/ground truth): \n";
        for (int i=0; i < n_samples_all;i++) {
            cout << labels_all[i] << " " << y[i] << "\n";
        }
        cout << elapsed_time*1000.0 << "\n";
    }

    MPI_Finalize();
    return 0;
}

// mpic++ -fopenmp -o kmeans_parallel kmeans_parallel.cpp
// mpiexec -n 2 ./kmeans_parallel