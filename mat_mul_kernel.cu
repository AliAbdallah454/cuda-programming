#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "helper.h"

using namespace std;

#define M 512
#define N 512
#define K 512
#define BLOCK_DIM 32

// void fill_matrix(float *matrix, int r, int c) {
//     for (int i = 0; i < r; ++i)
//         for (int j = 0; j < c; ++j)
//             matrix[i * c + j] = rand() % 100; // Random number [0, 99]
// }

// void print_matrix(const char *name, float *matrix, int r, int c) {
//     printf("%s:\n", name);
//     for (int i = 0; i < r; ++i) {
//         for (int j = 0; j < c; ++j)
//             printf("%4f ", matrix[i * c + j]);
//         printf("\n");
//     }
//     printf("\n");
// }

// void mat_mul_cpu(float *A, float *B, float *C, int m, int k, int n) {
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             float sum = 0;
//             for (int l = 0; l < k; ++l) {
//                 sum += A[i * k + l] * B[j + l * n];
//             }
//             C[i * n + j] = sum;
//         }
//     }
// }

__global__ void mat_mul_gpu(float *A, float *B, float *C, int m, int k, int n) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        float sum = 0;
        for (int l = 0; l < k; l++) {
            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
    }

}

int main() {

    float A[M * K];
    float B[K * N];
    float C_cpu[M * N];
    float C_gpu[M * N];

    int bytes_A = sizeof(A);
    int bytes_B = sizeof(B);
    int bytes_C = sizeof(C_cpu);

    cout << "Matrix A size: " << M << "x" << K << " → " << bytes_A << " bytes\n";
    cout << "Matrix B size: " << K << "x" << N << " → " << bytes_B << " bytes\n";
    cout << "Matrix C size: " << M << "x" << N << " → " << bytes_C << " bytes\n\n";
    srand(5);

    fill_matrix(A, M, K);
    fill_matrix(B, K, N);

    auto start_cpu = chrono::high_resolution_clock::now();
    mat_mul_cpu((float*)A, (float*)B, (float*)C_cpu, M, K, N);
    auto end_cpu = chrono::high_resolution_clock::now();    
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu);

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 number_of_blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    float *d_A;
    float *d_B;
    float *d_C;

    // Warm up START
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    mat_mul_gpu <<<number_of_blocks, threads_per_block>>> (d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    // Warm up END

    auto start_gpu = chrono::high_resolution_clock::now();

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

    mat_mul_gpu <<<number_of_blocks, threads_per_block>>> (d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);

    auto end_gpu = chrono::high_resolution_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(end_gpu - start_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    bool results_match = true;
    for (int i = 0; i < M && results_match; i++) {
        for (int j = 0; j < N && results_match; j++) {
            if (C_cpu[i * N + j] != C_gpu[i * N + j]) {
                results_match = false;
            }
        }
    }

    cout << "=== Performance Results ===\n";
    cout << "CPU Time: " << cpu_duration.count() << " microseconds\n";
    cout << "GPU Time: " << gpu_duration.count() << " microseconds\n";
    
    if (gpu_duration.count() > 0) {
        double speedup = (double)cpu_duration.count() / gpu_duration.count();
        double speedup_percentage = ((double)cpu_duration.count() - gpu_duration.count()) / cpu_duration.count() * 100.0;
        
        cout << "Speedup: " << speedup << "x\n";
    }
    
    cout << "Results match: " << (results_match ? "YES" : "NO") << "\n";
    
    return 0;
}