#include <cuda_runtime.h>
#include "Neuron.h"
#include <vector>
using namespace std;

__global__ void ReLUKernel(double* input, double* output, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * cols) {
        if (input[index] > 0.0) {
            output[index] = input[index];
        } else {
            output[index] = 0.0;
        }
    }
}


extern "C" void cudaReLU(const vector<vector<Neuron>>& input, 
                          vector<vector<Neuron>>& output, 
                          int rows, int cols) {

    vector<double> flat_input(rows * cols), flat_output(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat_input[i * cols + j] = input[i][j].value;
        }
    }

    double *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(double));
    cudaMalloc(&d_output, rows * cols * sizeof(double));
    
    cudaMemcpy(d_input, flat_input.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, rows * cols * sizeof(double));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (rows * cols + blockSize - 1) / blockSize;
    ReLUKernel<<<numBlocks, blockSize>>>(d_input, d_output, rows, cols);

    // Check for errors during kernel execution
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }

    // Copy result back to host
    cudaMemcpy(flat_output.data(), d_output, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Reshape result
    output.resize(rows, vector<Neuron>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i][j].value = flat_output[i * cols + j];
        }
    }
}
