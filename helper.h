#include <stdio.h>
#include <stdlib.h>

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