#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

#define M 4
#define N 4
#define K 4
#define BLOCK_DIM 4

void fill_matrix(float *matrix, int r, int c) {
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            matrix[i * c + j] = rand() % 100; // Random number [0, 99]
}

void print_matrix(const char *name, float *matrix, int r, int c) {
    printf("%s:\n", name);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j)
            printf("%4f ", matrix[i * c + j]);
        printf("\n");
    }
    printf("\n");
}

void mat_mul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[j + l * n];
            }
            C[i * n + j] = sum;
        }
    }
}

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

extern "C" void launch_mat_mul(float *A, float *B, float *C, int m, int k, int n) {
    
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);

    const int threads = 32;
    dim3 dim_block(threads, threads);
    dim3 dim_grid((n + threads - 1) / threads, (m + threads - 1) / threads);

    mat_mul_gpu<<<dim_grid, dim_block>>>(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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

    mat_mul_cpu((float*)A, (float*)B, (float*)C_cpu, M, K, N);
    launch_mat_mul(A, B, C_gpu, M, K, N);

    bool results_match = true;
    for (int i = 0; i < M && results_match; i++) {
        for (int j = 0; j < N && results_match; j++) {
            if (C_cpu[i * N + j] != C_gpu[i * N + j]) {
                results_match = false;
            }
        }
    }

    print_matrix("cpu mat", C_cpu, M, N);
    print_matrix("gpu mat", C_gpu, M, N);

    cout << "Results match: " << (results_match ? "YES" : "NO") << "\n";
    
    return 0;
}