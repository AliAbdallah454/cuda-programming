#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

#define SIZE 8

void fill_matrix(int matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            matrix[i][j] = rand() % 100; // Random number [0, 99]
}

void print_matrix(const char* name, int matrix[SIZE][SIZE]) {
    printf("%s:\n", name);
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%4d ", matrix[i][j]);
        printf("\n");
    }
    printf("\n");
}

__global__ void mat_add(int A[SIZE*SIZE], int B[SIZE * SIZE], int C[SIZE * SIZE]) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < SIZE && j < SIZE) {
        C[i + j * SIZE] = A[i + j * SIZE] + B[i + j * SIZE];
    }

}

int main() {
    int A[SIZE][SIZE];
    int B[SIZE][SIZE];
    int C[SIZE][SIZE];

    int bytes = sizeof(A);
    cout << "sizeof A is: " << bytes << "\n";

    srand(5);

    fill_matrix(A);
    fill_matrix(B);

    print_matrix("Matrix A", A);
    print_matrix("Matrix B", B);

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(4, 4);
    dim3 number_of_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);

    mat_add <<<number_of_blocks, threads_per_block>>> (d_A, d_B, d_C);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    print_matrix("Matrix C", C);
    
    return 0;
}