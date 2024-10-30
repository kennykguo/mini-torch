#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Neuron.h"
#include <vector>

using namespace std;

#define TILE_SIZE 32

__global__ void matrixMultiplyKernelOptimized(double* A, double* B, double* C, int m, int n, int k) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < m && tile * TILE_SIZE + tx < n)
            As[ty][tx] = A[row * n + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;
        
        if (col < k && tile * TILE_SIZE + ty < n)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * k + col];
        else
            Bs[ty][tx] = 0.0;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += As[ty][i] * Bs[i][tx];
        
        __syncthreads();
    }
    
    if (row < m && col < k)
        C[row * k + col] = sum;
}

extern "C" void cudaMatrixMultiply(const vector<vector<Neuron>>& A, 
                                   const vector<vector<Neuron>>& B, 
                                   vector<vector<Neuron>>& C) {
    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    // Flatten matrices
    vector<double> flatA(m * n), flatB(n * k), flatC(m * k);
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

    // Use asynchronous memory operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy data to device asynchronously
    cudaMemcpyAsync(d_A, flatA.data(), m * n * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, flatB.data(), n * k * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((k + TILE_SIZE - 1) / TILE_SIZE, 
                   (m + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyKernelOptimized<<<numBlocks, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, m, n, k);

    // Copy result back to host asynchronously
    cudaMemcpyAsync(flatC.data(), d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);

    // Free device memory and destroy stream
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);

    // Reshape result
    C.resize(m, vector<Neuron>(k));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            C[i][j].value = flatC[i * k + j];
}

/*


Tiling: We use shared memory to load tiles of matrices A and B. This reduces global memory accesses and improves performance.
Coalesced Memory Access: Threads in a warp access consecutive memory addresses when loading data into shared memory, which improves memory bandwidth utilization.
Shared Memory Usage: We use __shared__ memory to store tiles of matrices A and B, which is much faster to access than global memory.
Loop Unrolling: We use #pragma unroll to unroll the inner loop, which can improve instruction-level parallelism.
Asynchronous Memory Operations: We use CUDA streams and asynchronous memory operations to overlap computation with data transfer.

Additional optimizations and considerations:

Error Handling: Add proper CUDA error checking for all CUDA API calls.
Pinned Memory: Consider using pinned memory for the host arrays to improve transfer speeds between host and device.
Vectorized Memory Access: For newer GPU architectures, you could use vector types (double2, double4) to further improve memory bandwidth utilization.
Persistent Threads: For very large matrices, you might want to implement a persistent threads approach to reduce kernel launch overhead.
Mixed Precision: If your application allows, consider using lower precision types (like float instead of double) for faster computation and lower memory usage.
Tensor Cores: For compatible GPU architectures, you could leverage Tensor Cores for even faster matrix multiplication.





 */