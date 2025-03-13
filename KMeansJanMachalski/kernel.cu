#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda/atomic>
#include    <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstring>
#include <vector>

#define MAX_ITER 100
#define BLOCK_SIZE 512
#define CHECK_CUDA_ERROR(call)                                                \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct Data {
    int N, d, k;
    std::vector<float> points;    // Jednowymiarowa tablica punktów
    std::vector<float> centroids; // Jednowymiarowa tablica centroidów
    std::vector<int> assignments; // Przynależność punktów do klastrów
};
// Compute means for all clusters
struct compute_mean {
    __host__ __device__
        float operator()(float sum, int count) {
        return count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
};

void read_txt(const char* filename, Data& data);
void read_bin(const char* filename, Data& data);
void write_output(const char* filename, const Data& data);
void kmeans_cpu(Data& data);
void kmeans_gpu(Data& data);
template <typename T, unsigned int blockSize>
cudaError_t reduce_to_single_value(T* d_in, T* d_out, int N);
template <unsigned int dim>
cudaError_t process_dimension(
    Data& data,
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_changes_array,
    int* d_block_counts,
    float* d_block_sums,
    int N, int k, int grid_size,
    float* h_sums, int* h_counts,
    int d 
);


void kmeans_gpu_thrust(Data& data);

int main(int argc, char* argv[]) {

    if (argc != 5) {
        fprintf(stderr, "Usage: KMeans data_format computation_method input_file output_file\n");
        return EXIT_FAILURE;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    std::string data_format = argv[1];
    std::string computation_method = argv[2];
    const char* input_file = argv[3];
    const char* output_file = argv[4];

    Data data;

    // Wczytywanie danych
    printf("Loading data from %s...\n", input_file);
    clock_t load_start = clock();

    if (data_format == "txt") {
        read_txt(input_file, data);
    }
    else if (data_format == "bin") {
        read_bin(input_file, data);
    }
    else {
        fprintf(stderr, "Error: Unknown data format '%s'. Use 'txt' or 'bin'.\n", data_format.c_str());
        return EXIT_FAILURE;
    }

    clock_t load_end = clock();
    printf("Data loaded in %.2fs\n", (double)(load_end - load_start) / CLOCKS_PER_SEC);
    printf("Number of points: %d, Dimensions: %d, Clusters: %d\n", data.N, data.d, data.k);

    // Uruchamianie algorytmu
    if (computation_method == "cpu")
        kmeans_cpu(data);
    else if (computation_method == "gpu1")
        kmeans_gpu_thrust(data);
    else if (computation_method == "gpu2")
        kmeans_gpu(data);

    // Zapis wyników
    printf("Writing results to %s...\n", output_file);
    clock_t save_start = clock();
    write_output(output_file, data);
    clock_t save_end = clock();
    printf("Results written in %.2fs\n", (double)(save_end - save_start) / CLOCKS_PER_SEC);

    // Koniec całkowitego pomiaru czasu
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    printf("Total execution time: %.2fs\n", total_elapsed.count());

    return EXIT_SUCCESS;
}

void read_txt(const char* filename, Data& data) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d %d", &data.N, &data.d, &data.k);

    data.points.resize(data.N * data.d);

    for (int i = 0; i < data.N; ++i) {
        for (int j = 0; j < data.d; ++j) {
            float val;
            fscanf(file, "%f", &val);
            data.points[j * data.N + i] = val; // Przechowywanie w formacie [dim * N + i]
        }
    }

    data.centroids.resize(data.k * data.d);
    for (int dim = 0; dim < data.d; ++dim) {
        for (int c = 0; c < data.k; ++c) {
            data.centroids[dim * data.k + c] = data.points[dim * data.N + c];
        }
    }
    data.assignments.resize(data.N, -1);

    fclose(file);
}

void read_bin(const char* filename, Data& data) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fread(&data.N, sizeof(int), 1, file);
    fread(&data.d, sizeof(int), 1, file);
    fread(&data.k, sizeof(int), 1, file);

    data.points.resize(data.N * data.d);

    std::vector<float> buffer(data.d);

    for (int i = 0; i < data.N; ++i) {
        fread(buffer.data(), sizeof(float), data.d, file);
        for (int j = 0; j < data.d; ++j) {
            data.points[j * data.N + i] = buffer[j];
        }
    }

    data.centroids.resize(data.k * data.d);
    for (int dim = 0; dim < data.d; ++dim) {
        for (int c = 0; c < data.k; ++c) {
            data.centroids[dim * data.k + c] = data.points[dim * data.N + c];
        }
    }
    data.assignments.resize(data.N, -1);

    fclose(file);
}

void write_output(const char* filename, const Data& data) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int c = 0; c < data.k; ++c) {
        for (int j = 0; j < data.d; ++j) {
            fprintf(file, "%.6f ", data.centroids[j * data.k + c]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i < data.N; ++i) {
        fprintf(file, "%d\n", data.assignments[i]);
    }
    fclose(file);
}

void kmeans_cpu(Data& data) {
    printf("Starting K-Means on CPU...\n");
    clock_t start_time = clock();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        printf("Iteration %d: ", iter + 1);
        clock_t iter_start = clock();

        // Krok 1: Przypisanie punktów do najbliższego centroidu
        int changes = 0;
        for (int i = 0; i < data.N; ++i) {
            int best_cluster = -1;
            float best_distance = INFINITY;

            for (int c = 0; c < data.k; ++c) {
                float dist = 0.0;
                for (int dim = 0; dim < data.d; ++dim) {
                    float diff = data.points[dim * data.N + i] - data.centroids[dim * data.k + c];
                    dist += diff * diff;
                }
                if (dist < best_distance) {
                    best_distance = dist;
                    best_cluster = c;
                }
            }

            if (data.assignments[i] != best_cluster) {
                data.assignments[i] = best_cluster;
                ++changes;
            }
        }
        printf("%d points changed\n", changes);

        // Jeśli brak zmian, zakończ algorytm
        if (changes == 0) break;

        // Krok 2: Aktualizacja centroidów
        std::vector<float> new_centroids(data.k * data.d, 0.0f);
        std::vector<int> cluster_sizes(data.k, 0);

        for (int i = 0; i < data.N; ++i) {
            int cluster = data.assignments[i];
            for (int dim = 0; dim < data.d; ++dim) {
                new_centroids[dim * data.k + cluster] += data.points[dim * data.N + i];
            }
            ++cluster_sizes[cluster];
        }

        for (int c = 0; c < data.k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (int dim = 0; dim < data.d; ++dim) {
                    new_centroids[dim * data.k + c] /= cluster_sizes[c];
                }
            }
        }

        data.centroids = new_centroids;

        clock_t iter_end = clock();
        printf("Iteration time: %.2fs\n", (double)(iter_end - iter_start) / CLOCKS_PER_SEC);
    }

    clock_t end_time = clock();
    printf("K-Means completed in %.2fs\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
}
template <typename T, unsigned int blockSize>
__global__ void reduce_array(T* d_in, T* d_out, int N) {
    extern __shared__ char shared_mem[];  // Dynamiczna pamięć współdzielona
    T* sdata = reinterpret_cast<T*>(shared_mem);  // Rzutowanie na odpowiedni typ

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Pierwsza faza redukcji: wczytanie dwóch elementów z globalnej pamięci
    T val = 0;
    if (i < N) {
        val = d_in[i];
        if (i + blockDim.x < N) {  // Sprawdzenie, czy drugi element jest w granicach
            val += d_in[i + blockDim.x];
        }
    }
    sdata[tid] = val;  // Zapisanie do pamięci współdzielonej
    __syncthreads();

    // Rozwinięcie pętli redukcji w pamięci współdzielonej
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    // Unrollowanie ostatniego warpu
    if (tid < 32) {
        volatile T* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Zapisanie wyniku redukcji bloku
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

template <typename T, unsigned int blockSize>
cudaError_t reduce_to_single_value(T* d_in, T* d_out, int N) {
    int blocks = (N + blockSize * 2 - 1) / (blockSize * 2);  // Liczba bloków
    size_t shared_memory_size = blockSize * sizeof(T);  // Rozmiar pamięci współdzielonej

    cudaError_t err = cudaSuccess;
    while (N > 1) {
        reduce_array<T, blockSize> << <blocks, blockSize, shared_memory_size >> > (d_in, d_out, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "reduce_array kernel error: %s\n", cudaGetErrorString(err));
            return err;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return err;
        }
        N = blocks;  // Nowa liczba elementów to liczba bloków
        d_in = d_out;  // Aktualizacja wskaźników
        blocks = (N + blockSize * 2 - 1) / (blockSize * 2);  // Obliczenie nowej liczby bloków
    }
    return err;
}



__global__ void assign_points_to_centroids(
    const float* points,
    const float* centroids,
    int* assignments,
    int* changes_array,
    int d, int N, int k
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= N) return;

    int prev_assignment = assignments[point_idx];

    float min_dist = FLT_MAX;
    int best_cluster = -1;

    // Iteracja po centroidach
    for (int c = 0; c < k; ++c) {
        float dist = 0.0f;

        // Oblicz odległość euklidesową do centroidu c
        for (int dim = 0; dim < d; ++dim) {
            float diff = points[dim * N + point_idx] - centroids[dim * k + c];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    // Sprawdź, czy przypisanie się zmieniło
    if (prev_assignment != best_cluster) {
        changes_array[point_idx] = 1; // Zapisz informację o zmianie
        assignments[point_idx] = best_cluster;
    }
    else {
        changes_array[point_idx] = 0; // Brak zmiany
    }
}

template <unsigned int blockSize>
__global__ void reduce_centroid_counts(
    const int* d_assignments,
    int c,  // Indeks centroidu
    int* d_block_counts,  // Wyjściowe liczności dla bloków
    int N
) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;  // Dwukrotnie większy krok bloku

    // Załaduj dwa elementy i wykonaj pierwsze sumowanie
    int val = 0;
    if (i < N && d_assignments[i] == c) {
        val = 1;
    }
    if (i + blockDim.x < N && d_assignments[i + blockDim.x] == c) {
        val += 1;
    }
    sdata[tid] = val;
    __syncthreads();

    // Pełne rozwinięcie warunków w zależności od blockSize
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    // Unrollowanie ostatniego warpu
    if (tid < 32) {
        volatile int* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Zapisanie wyniku redukcji bloku
    if (tid == 0) {
        d_block_counts[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce_centroid_sums(
    const float* d_points_dim,
    const int* d_assignments,
    int c,  // Indeks centroidu
    float* d_block_sums,  // Wyjściowe sumy dla bloków
    int N
) {
    extern __shared__ float sdata_f[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;  // Dwukrotnie większy krok bloku

    // Załaduj dwa elementy i wykonaj pierwsze sumowanie
    float val = 0.0f;
    if (i < N && d_assignments[i] == c) {
        val = d_points_dim[i];
    }
    if (i + blockDim.x < N && d_assignments[i + blockDim.x] == c) {
        val += d_points_dim[i + blockDim.x];
    }
    sdata_f[tid] = val;
    __syncthreads();

    // Pełne rozwinięcie warunków w zależności od blockSize
    if (blockSize >= 1024) {
        if (tid < 512) { sdata_f[tid] += sdata_f[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata_f[tid] += sdata_f[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata_f[tid] += sdata_f[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata_f[tid] += sdata_f[tid + 64]; } __syncthreads();
    }

    // Unrollowanie ostatniego warpu
    if (tid < 32) {
        volatile float* vsmem = sdata_f;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Zapisanie wyniku redukcji bloku
    if (tid == 0) {
        d_block_sums[blockIdx.x] = sdata_f[0];
    }
}

void kmeans_gpu_thrust(Data& data)
{
    int d = data.d;
    int N = data.N;
    int k = data.k;

    printf("Starting K-Means on GPU...\n");
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaSuccess; // Zmienna do przechowywania kodu błędu
    float* d_points = nullptr;
    float* d_centroids = nullptr;
    int* d_assignments = nullptr;
    int* d_changes_array = nullptr;

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iteration = 0;
    int h_changes = data.N;

    thrust::device_vector<int> d_counts(k, 0);    // Liczba punktów w każdym centroidzie
    thrust::device_vector<int> d_assignments_vec(N);
    thrust::device_vector<float> d_points_vec(N);
    thrust::device_vector<float> d_sums(k, 0.0f); // Suma współrzędnych punktów dla każdego centroidu
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator> new_end;

    // Alokacja pamięci na urządzeniu
    err = cudaMalloc(&d_points, N * d * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_points, data.points.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc(&d_centroids, k * d * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_centroids, data.centroids.data(), k * d * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc(&d_assignments, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_assignments, data.assignments.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc(&d_changes_array, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    while (iteration < MAX_ITER && h_changes != 0)
    {
        printf("Iteration %d\n", iteration + 1);

        auto iter_start = std::chrono::high_resolution_clock::now();

        // Wywołanie kernela przypisującego punkty
        assign_points_to_centroids << <grid_size, BLOCK_SIZE >> > (
            d_points,
            d_centroids,
            d_assignments,
            d_changes_array,
            d, N, k
            );
        err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "Kernel assign_points_to_centroids error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        // Użycie Thrust do zsumowania zmian
        thrust::device_ptr<int> changes_ptr(d_changes_array);
        h_changes = thrust::reduce(changes_ptr, changes_ptr + N, 0, thrust::plus<int>());

        printf("Points changed: %d\n", h_changes);

        if (h_changes == 0)
        {
            auto iter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> iter_elapsed = iter_end - iter_start;
            printf("Iteration time: %.6f seconds\n", iter_elapsed.count());
            break;
        }

        // Aktualizacja centroidów za pomocą Thrust
        for (int dim = 0; dim < d; ++dim) {
            // Wskaźniki do punktów w bieżącym wymiarze
            float* d_points_dim = d_points + dim * N;
            float* d_centroids_dim = d_centroids + dim * k;

            // Kopiowanie przypisań do thrust::device_vector
            thrust::copy(thrust::device_ptr<int>(d_assignments), thrust::device_ptr<int>(d_assignments + N), d_assignments_vec.begin());

            // Kopiowanie punktów do thrust::device_vector
            thrust::copy(thrust::device_ptr<float>(d_points_dim), thrust::device_ptr<float>(d_points_dim + N), d_points_vec.begin());

            // Sortowanie punktów względem przypisań
            thrust::stable_sort_by_key(
                d_assignments_vec.begin(), d_assignments_vec.end(),
                d_points_vec.begin()
            );

            // Obliczenie sumy współrzędnych i liczby punktów dla każdego centroidu
            thrust::device_vector<int> unique_clusters = d_assignments_vec;

            new_end = thrust::reduce_by_key(
                d_assignments_vec.begin(), d_assignments_vec.end(),
                d_points_vec.begin(),
                unique_clusters.begin(),  // Klucze centroidów
                d_sums.begin()
            );
            if (dim == 0) //wystarczy zliczyć tylko raz
            {
                thrust::reduce_by_key(
                    d_assignments_vec.begin(), d_assignments_vec.end(),
                    thrust::constant_iterator<int>(1),
                    unique_clusters.begin(),
                    d_counts.begin()
                );
            }

            // Obliczenie nowych centroidów
            thrust::transform(
                d_sums.begin(), d_sums.end(),
                d_counts.begin(),
                d_sums.begin(),
                compute_mean()
            );

            // Kopiowanie nowych centroidów do pamięci urządzenia
            thrust::copy(d_sums.begin(), d_sums.begin() + k, thrust::device_ptr<float>(d_centroids_dim));
        }

        // Pomiar czasu iteracji
        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_elapsed = iter_end - iter_start;
        printf("Iteration time: %.6f seconds\n", iter_elapsed.count());

        iteration++;
    }

    // Kopiowanie wyników na hosta
    err = cudaMemcpy(data.centroids.data(), d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(data.assignments.data(), d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Zwolnienie pamięci
cleanup:
    // Zwolnienie pamięci
    if (d_points) cudaFree(d_points);
    if (d_centroids) cudaFree(d_centroids);
    if (d_assignments) cudaFree(d_assignments);
    if (d_changes_array) cudaFree(d_changes_array);

    // Pomiar całkowitego czasu
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    printf("Total computing time: %.6f seconds\n", total_elapsed.count());
}

void kmeans_gpu(Data& data)
{
    int d = data.d;
    int N = data.N;
    int k = data.k;

    printf("Starting K-Means on GPU...\n");
    auto total_start = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaSuccess;
    float* d_points = nullptr;
    float* d_centroids = nullptr;
    int* d_assignments = nullptr;
    int* d_changes_array = nullptr;
    int* d_output_changes = nullptr;
    float* d_block_sums = nullptr;
    int* d_block_counts = nullptr;

    // ustawienia kerneli
    int grid_size = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    int iteration = 0;
    int h_changes = data.N;

	// pamięć na hosta
    float* h_sums = new float[k];
    int* h_counts = new int[k];


	// pamięć na urządzeniu
    err = cudaMalloc(&d_points, N * d * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_points error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(d_points, data.points.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_points error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_centroids, k * d * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_centroids error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(d_centroids, data.centroids.data(), k * d * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_centroids error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_assignments, N * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_assignments error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(d_assignments, data.assignments.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_assignments error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_changes_array, N * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_changes_array error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_output_changes, sizeof(int) * ((N + BLOCK_SIZE-1) / BLOCK_SIZE));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_output_changes error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_block_sums, grid_size * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_block_sums error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&d_block_counts, grid_size * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_block_counts error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    while (iteration < MAX_ITER && h_changes != 0)
    {
        printf("Iteration %d\n", iteration + 1);

        auto iter_start = std::chrono::high_resolution_clock::now();

        // przypisanie punktów
        assign_points_to_centroids << < (N + BLOCK_SIZE * 2 - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (
            d_points,
            d_centroids,
            d_assignments,
            d_changes_array,
            d, N, k
            );
        err = cudaGetLastError();
        if (err != cudaSuccess) { fprintf(stderr, "Kernel assign_points_to_centroids error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        err = reduce_to_single_value<int, BLOCK_SIZE>(d_changes_array, d_output_changes, N);
        if (err != cudaSuccess) { fprintf(stderr, "reduce_to_single_value error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        err = cudaMemcpy(&h_changes, d_output_changes, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy h_changes error: %s\n", cudaGetErrorString(err)); goto cleanup; }

        printf("Points changed: %d\n", h_changes);

        if (h_changes == 0)
        {
            auto iter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> iter_elapsed = iter_end - iter_start;
            printf("Iteration time: %.6f seconds\n", iter_elapsed.count());
            break;
        }

        // aktualizacja centroidów      

        switch (d) {
        case 20: err = process_dimension<20>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums,N, k, grid_size, h_sums, h_counts, d); break;
        case 19: err = process_dimension<19>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums,N, k, grid_size, h_sums, h_counts, d); break;
        case 18: err = process_dimension<18>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 17: err = process_dimension<17>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 16: err = process_dimension<16>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 15: err = process_dimension<15>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 14: err = process_dimension<14>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 13: err = process_dimension<13>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 12: err = process_dimension<12>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 11: err = process_dimension<11>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 10: err = process_dimension<10>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 9: err = process_dimension<9>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 8: err = process_dimension<8>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 7: err = process_dimension<7>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 6: err = process_dimension<6>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 5: err = process_dimension<5>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 4: err = process_dimension<4>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 3: err = process_dimension<3>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 2: err = process_dimension<2>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
        case 1: err = process_dimension<1>(data, d_points, d_centroids, d_assignments, d_changes_array, d_block_counts, d_block_sums, N, k, grid_size, h_sums, h_counts, d); break;
       
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in process_dimension: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }

        // Measure iteration time
        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_elapsed = iter_end - iter_start;
        printf("Iteration time: %.6f seconds\n", iter_elapsed.count());

        iteration++;
    }

    // Copy results to host
    err = cudaMemcpy(data.centroids.data(), d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy data.centroids error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(data.assignments.data(), d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy data.assignments error: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Free memory
cleanup:
    // Zwolnienie pamięci GPU
    if (d_points) cudaFree(d_points);
    if (d_centroids) cudaFree(d_centroids);
    if (d_assignments) cudaFree(d_assignments);
    if (d_changes_array) cudaFree(d_changes_array);
    if (d_block_sums) cudaFree(d_block_sums);
    if (d_block_counts) cudaFree(d_block_counts);
    if (d_output_changes) cudaFree(d_output_changes);

    // Zwolnienie pamięci host
    delete[] h_sums;
    delete[] h_counts;

    // Measure total time
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    printf("Total computing time: %.6f seconds\n", total_elapsed.count());
}

template <>
cudaError_t  process_dimension<0>(Data&, float*, float*, int*, int*, int*, float*, int, int, int, float*, int*, int) {
    // Nic nie robimy dla dim=0 - bazowy przypadek
	return cudaSuccess;
}

template <unsigned int dim>
cudaError_t  process_dimension(
    Data& data,
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_changes_array,
    int* d_block_counts,
    float* d_block_sums,
    int N, int k, int grid_size,
    float* h_sums, int* h_counts,
    int d // dodajemy parametr d
) {
    int current_dim = dim - 1;
	cudaError_t err = cudaSuccess;

    if (current_dim >= d) {
        return err;
    }

    float* d_points_dim = d_points + current_dim * N;
    float* d_centroids_dim = d_centroids + current_dim * k;

    //redukcja punktów dla każdego klastra osobno
    for (int c = 0; c < k; ++c) {
        cudaError_t err;
        err = cudaMemset(d_block_sums, 0, grid_size * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemset d_block_sums error: %s\n", cudaGetErrorString(err));
            return err; 
        }

        if (current_dim == 0) {
            err = cudaMemset(d_block_counts, 0, grid_size * sizeof(int));
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMemset d_block_counts error: %s\n", cudaGetErrorString(err));
                return err;
            }
        }
        //suma punktów dla każdego bloku
        reduce_centroid_sums<BLOCK_SIZE> << <grid_size, BLOCK_SIZE, (BLOCK_SIZE * sizeof(float)) >> > (
            d_points_dim,
            d_assignments,
            c,
            d_block_sums,
            N
            );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel reduce_centroid_sums error: %s\n", cudaGetErrorString(err));
            return err;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return err;
        }

        // łączna suma punktów dla wszystkich bloków
        float total_sum = 0.0f;
        err = reduce_to_single_value<float, BLOCK_SIZE>(d_block_sums, d_block_sums, grid_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "reduce_to_single_value error: %s\n", cudaGetErrorString(err));
            return err;
        }

        err = cudaMemcpy(&total_sum, d_block_sums, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy total_sum error: %s\n", cudaGetErrorString(err));
            return err;
        }

        // wystarczy zliczyć przynależność tylko dla jednego wymiaru
        if (current_dim == d-1) {
            reduce_centroid_counts<BLOCK_SIZE> << <grid_size, BLOCK_SIZE, (BLOCK_SIZE * sizeof(int)) >> > (
                d_assignments,
                c,
                d_block_counts,
                N
                );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Kernel reduce_centroid_counts error: %s\n", cudaGetErrorString(err));
                return err;
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
                return err;
            }

            int total_count = 0;
            err = reduce_to_single_value<int, BLOCK_SIZE>(d_block_counts, d_block_counts, grid_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "reduce_to_single_value error: %s\n", cudaGetErrorString(err));
                return err;
            }

            err = cudaMemcpy(&total_count, d_block_counts, sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy total_count error: %s\n", cudaGetErrorString(err));
                return err;
            }

            h_counts[c] = total_count;
        }

        h_sums[c] = total_sum;
    }

    // aktualizacja centroidów
    for (int c = 0; c < k; ++c) {
        int count = h_counts[c];
        float centroid_value = (count > 0) ? h_sums[c] / count : 0.0f;
        cudaError_t err = cudaMemcpy(d_centroids_dim + c, &centroid_value, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy centroid_value error: %s\n", cudaGetErrorString(err));
            return err;
        }
    }

    // Rekurencja - przechodzimy do niższego wymiaru

        return process_dimension<dim - 1>(
            data, d_points, d_centroids, d_assignments, d_changes_array,
            d_block_counts, d_block_sums,
            N, k, grid_size, h_sums, h_counts,
            d 
        );

}

