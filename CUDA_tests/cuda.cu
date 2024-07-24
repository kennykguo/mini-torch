#include <cuda_runtime.h>
#include <iostream>

__global__ void forward_kernel(float *input, float *weights, float *output, int input_size, int output_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = 1.0f / (1.0f + exp(-sum));  // Sigmoid activation
    }
}

void forward(const std::vector<float> &input, const std::vector<float> &weights, std::vector<float> &output, int input_size, int output_size) {
    float *d_input, *d_weights, *d_output;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    forward_kernel<<<numBlocks, blockSize>>>(d_input, d_weights, d_output, input_size, output_size);

    cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}
