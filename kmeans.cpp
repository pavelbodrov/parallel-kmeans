#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <random>
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
    double** X = nullptr;

    map<string, int> shape;
    int nearest_centroid, n_samples = 0, n_features = 0, n_clusters = 3, n_iter=0;
    double min_dist, curr_dist, dist_sum = 0, dist_sum_new = 0;
    double tol = 0.1;


    int *y = nullptr;

    shape = read_csv(data, "iris_encoded_big.csv");
    n_samples = shape.at("rows");
    n_features = shape.at("cols") - 1;

    y = new int[n_samples];

    X = make_matrix(n_samples, n_features);

    for (int i=0; i < n_samples; i++) {
        for (int j=0; j < n_features; j++) {
            X[i][j] = data[i][j];
        }
        // Cluster labels
        y[i] = (int) data[i][n_features];
    }


    // Start timer
    double start_time = clock();

    double** centroids = make_matrix(n_clusters, n_features);
    double** new_centroids = make_matrix(n_clusters, n_features);
    int cluster_sizes[n_clusters];

    init_centroids(X, centroids, n_clusters, n_samples, n_features);
    for (int i=0; i<n_clusters; i++) {
        for (int j=0; j<n_features; j++) {
            cluster_sizes[i] = 0;
            new_centroids[i][j] = 0;
        }
    }

    int cluster = 0;
    int *labels = new int[n_samples];

    do {
        n_iter++;
        dist_sum = dist_sum_new;
        dist_sum_new = 0;

        // Calculate distances
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


        // Update centroids
        for (int i = 0; i < n_samples; i++) {
            cluster = labels[i];
            cluster_sizes[cluster]++;
            for (int j=0; j<n_features; j++) {
                new_centroids[cluster][j] += X[i][j];
            }
        }

        for (int i=0; i < n_clusters; i++) {
            for (int j=0; j < n_features; j++) {
                if (cluster_sizes[i] > 0) {
                    centroids[i][j] = new_centroids[i][j] / cluster_sizes[i];
                }
                new_centroids[i][j] = 0.0; // Fill with zeros for the next iteration
            }
            cluster_sizes[i] = 0; // Fill with zeros for the next iteration
        }

        printf("Epoch %d, Inertia=%f, Inertia diff=%f\n", n_iter, dist_sum_new, abs(dist_sum - dist_sum_new));
    }
//    while ((abs(dist_sum - dist_sum_new) > tol) && n_iter < 10);
    while (n_iter < 100);

    // Stop timer
    double elapsed_time = clock() - start_time;

    cout << "Results (predicted label/ground truth): \n";
    for (int i=0; i < n_samples;i++) {
        cout << labels[i] << " " << y[i] << "\n";
    }
    cout << elapsed_time/CLOCKS_PER_SEC*1000 << "\n";

    return 0;
}