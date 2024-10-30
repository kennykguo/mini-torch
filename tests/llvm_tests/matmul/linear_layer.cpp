// LinearLayer.cpp
vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& input) {
    // Prepare data for CUDA
    float *d_input, *d_weights, *d_output;
    // ... allocate and copy data to GPU ...
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Use LLVM-optimized CUDA kernel
    matrixMultiply(d_input, d_weights, d_output,
                   input.size(), inputSize, outputSize, stream);
    
    // ... copy results back and cleanup ...
    
    return output;
}