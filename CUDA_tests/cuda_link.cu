// Example usage of CUDA-accelerated matrix multiplication in LinearLayer class

// Inside LinearLayer.cpp
#include "LinearLayer.h"
#include "Network.h"
#include "LinearLayer.cu"  // Include CUDA implementations

// Forward propagation function using CUDA-accelerated matrix multiplication
vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& inputMatrix) {
    // Call CUDA-accelerated matrix multiplication
    vector<vector<Neuron>> output = network.matrixMultiplyCUDA(inputMatrix, inputMatrix.size(), inputMatrix[0].size(),
                                                              weightsMatrix, fan_out, fan_in);

    // Example: Add bias (to be optimized with CUDA if needed)
    for (int i = 0; i < fan_out; ++i) {
        for (int j = 0; j < inputMatrix[0].size(); ++j) {
            output[i][j].value += biasMatrix[i].value;
        }
    }

    this->output_rows = inputMatrix.size();
    this->output_cols = fan_out;
    return output;
}

// CUDA Headers and Declarations: Use .cu files for CUDA kernels and include corresponding headers (LinearLayer.h, Network.h) in both CPU and CUDA files.
// Compilation: Compile CUDA files (nvcc -c LinearLayer.cu) separately and link with your main C++ program during final linking (g++ Network.cpp LinearLayer.o -o YourExecutable -L/path/to/cuda/libraries -lcudart).
// cuh file vs cu file?