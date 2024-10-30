// CUDA matrix multiplication kernel
__global__ void matrixMultiplyCUDA(Neuron* matrixOne, int rowsOne, int colsOne,
                                   Neuron* matrixTwo, int rowsTwo, int colsTwo,
                                   Neuron* result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsOne && col < colsTwo) {
        double currentSum = 0.0;
        for (int k = 0; k < colsOne; ++k) {
            currentSum += matrixOne[row * colsOne + k].value * matrixTwo[k * colsTwo + col].value;
        }
        result[row * colsTwo + col].value = currentSum;
    }
}

// Matrix multiplication function using CUDA
vector<vector<Neuron>> Network::matrixMultiplyCUDA(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {

    // Allocate memory on GPU
    Neuron *d_matrixOne, *d_matrixTwo, *d_result;
    size_t sizeOne = rowsOne * colsOne * sizeof(Neuron);
    size_t sizeTwo = rowsTwo * colsTwo * sizeof(Neuron);
    cudaMalloc(&d_matrixOne, sizeOne);
    cudaMalloc(&d_matrixTwo, sizeTwo);
    cudaMalloc(&d_result, rowsOne * colsTwo * sizeof(Neuron));

    // Copy input matrices to GPU
    cudaMemcpy(d_matrixOne, matrixOne.data(), sizeOne, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixTwo, matrixTwo.data(), sizeTwo, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16);  // Thread block dimensions
    dim3 gridDim((colsTwo + blockDim.x - 1) / blockDim.x, (rowsOne + blockDim.y - 1) / blockDim.y);  // Grid dimensions

    // Launch CUDA kernel
    matrixMultiplyCUDA<<<gridDim, blockDim>>>(d_matrixOne, rowsOne, colsOne, d_matrixTwo, rowsTwo, colsTwo, d_result);

    // Copy result matrix back from GPU to CPU
    vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));
    cudaMemcpy(result.data(), d_result, rowsOne * colsTwo * sizeof(Neuron), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_matrixOne);
    cudaFree(d_matrixTwo);
    cudaFree(d_result);

    return result;
}
