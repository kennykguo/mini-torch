#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Neuron.h"
#include <vector>

__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

extern "C" void cudaMatrixMultiply(const std::vector<std::vector<Neuron>>& A, 
                                   const std::vector<std::vector<Neuron>>& B, 
                                   std::vector<std::vector<Neuron>>& C) {
    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    // Flatten matrices
    std::vector<double> flatA(m * n), flatB(n * k), flatC(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            flatA[i * n + j] = A[i][j].value;
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            flatB[i * k + j] = B[i][j].value;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_B, n * k * sizeof(double));
    cudaMalloc(&d_C, m * k * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, flatA.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), n * k * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // Copy result back to host
    cudaMemcpy(flatC.data(), d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Reshape result
    C.resize(m, std::vector<Neuron>(k));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            C[i][j].value = flatC[i * k + j];
}