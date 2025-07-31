#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <iostream>
#include "helper.h"

using namespace std;

#define M 32 * 20
#define K 32 * 20
#define N 32 * 20

int main() {

    // Host matrices (row-major for ease)
    float h_A[M * K];     
    float h_B[K * N];
    float h_C_cpu[M * N];
    float h_C_cublas[M * N];

    int bytes_A = M * K * sizeof(float);
    int bytes_B = K * N * sizeof(float);
    int bytes_C = M * N * sizeof(float);

    cout << "Matrix A size: " << M << "x" << K << " → " << bytes_A << " bytes\n";
    cout << "Matrix B size: " << K << "x" << N << " → " << bytes_B << " bytes\n";
    cout << "Matrix C size: " << M << "x" << N << " → " << bytes_C << " bytes\n\n";
    
    srand(5);
    fill_matrix(h_A, M, K);
    fill_matrix(h_B, K, N);

    // CPU matrix multiplication
    auto start_cpu = chrono::high_resolution_clock::now();
    mat_mul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    auto end_cpu = chrono::high_resolution_clock::now();    
    auto cpu_duration = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu);

    float *d_A, *d_B, *d_C;

    // cuBLAS assumes column-major by default
    // We compute: C = alpha * B^T * A^T + beta * C
    float alpha = 1.0f;
    float beta  = 0.0f;

    /// Warm up START

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cublasSgemm arguments: column-major so A and B are transposed
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,       // C is (N x M)
                &alpha,
                d_B, N,        // B is (N x K)
                d_A, K,        // A is (K x M)
                &beta,
                d_C, N);       // C is (N x M)
    cudaDeviceSynchronize();
    
    /// Warm up END

    auto start_gpu = chrono::high_resolution_clock::now();


    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    cublasHandle_t handle_2;
    cublasCreate(&handle_2);

    // cublasSgemm arguments: column-major so A and B are transposed
    cublasSgemm(handle_2,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,       // C is (N x M)
                &alpha,
                d_B, N,        // B is (N x K)
                d_A, K,        // A is (K x M)
                &beta,
                d_C, N);       // C is (N x M)
    cudaDeviceSynchronize();

    auto end_gpu = chrono::high_resolution_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::microseconds>(end_gpu - start_gpu);

    // Copy result back to host
    cudaMemcpy(h_C_cublas, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Clean up
    cublasDestroy(handle);
    cublasDestroy(handle_2);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Verify results match
    bool results_match = true;
    for (int i = 0; i < M && results_match; i++) {
        for (int j = 0; j < N && results_match; j++) {
            if (abs(h_C_cpu[i * N + j] - h_C_cublas[i * N + j]) > 1e-5) {
                results_match = false;
            }
        }
    }

    cout << "=== Performance Results ===\n";
    cout << "CPU Time: " << cpu_duration.count() << " microseconds\n";
    cout << "cuBLAS Time: " << gpu_duration.count() << " microseconds\n";
    
    if (gpu_duration.count() > 0) {
        double speedup = (double)cpu_duration.count() / gpu_duration.count();
        cout << "Speedup: " << speedup << "x\n";
    }
    
    cout << "Results match: " << (results_match ? "YES" : "NO") << "\n";

    return 0;
}
