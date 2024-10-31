#include "Network.h"
#include <iostream>

using namespace std;

// Takes in a unique_ptr to a Layer object
void Network::addLayer(unique_ptr<Layer> layer) {
    // dynamic_cast is used for runtime type checking
    // It attempts to cast the Layer pointer to a SoftmaxCrossEntropy pointer. If sucessful, then nullptr is returned
    if (dynamic_cast<SoftmaxCrossEntropy*>(layer.get())) {
        this->loss_layer = dynamic_cast<SoftmaxCrossEntropy*>(layer.get());
    }
    // Move the unique_ptr into the layers vector
    // The function move() transfers ownership of the unique_ptr to the layers vector
    layers.push_back(move(layer));
}


vector<vector<Neuron>> Network::forward(const vector<vector<Neuron>>& input, const vector<vector<Neuron>>& labels) {
    // Before every forward propagation, clear the 3D matrix. This removes all elements from the vector
    this->layer_outputs.clear();
    // Add the inputs as the first entry
    this->layer_outputs.push_back(input);
    // If the loss_layer exists, set the labels before the forward pass
    if (this->loss_layer) {
        this->loss_layer->setLabels(labels);
    }

    // Forward pass through each layer
    vector<vector<Neuron>> current = input;
    for (const unique_ptr<Layer>& layer : this->layers) {
        current = layer->forward(current);
        this->layer_outputs.push_back(current);
    }

    // Compute loss if the lossLayer is set
    if (this->loss_layer) {
        loss = this->loss_layer->getLoss();
    }
    return current;
}


void Network::backward() {
    vector<vector<Neuron>> grad = this->layer_outputs.back();
    for (int i = layers.size() - 1; i >= 0; --i) {
        grad = layers[i]->backward(grad);
    }
}

// (mat1, A, B, mat2, B, C) -> (A, B) @ (B, C) -> (A, C)
vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrix_one, int rows_one, int cols_one,
    const vector<vector<Neuron>>& matrix_two, int rows_two, int cols_two) {
    vector<vector<Neuron>> result;
    cudaMatrixMultiply(matrix_one, matrix_two, result);
    return result;
}


// // NO CUDA matrix multiplication
// vector<vector<Neuron>> Network::matrixMultiply(
//     const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
//     const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {
    
//     // cout << "Entering matrixMultiply function" << endl;
//     // cout << "matrixOne dimensions: " << matrixOne.size() << "x" << (matrixOne.empty() ? 0 : matrixOne[0].size()) << endl;
//     // cout << "matrixTwo dimensions: " << matrixTwo.size() << "x" << (matrixTwo.empty() ? 0 : matrixTwo[0].size()) << endl;
//     // cout << "Passed dimensions: " << rowsOne << "x" << colsOne << " * " << rowsTwo << "x" << colsTwo << endl;

//     // Ensure the dimensions are correct
//     // assert(colsOne == rowsTwo && "Matrix dimensions must match for multiplication.");
//     // assert(rowsOne == matrixOne.size() && "rowsOne doesn't match matrixOne size");
//     // assert(colsTwo == matrixTwo[0].size() && "colsTwo doesn't match matrixTwo column size");

//     // cout << "Initializing result matrix" << endl;
//     // // Initialize result matrix with the correct dimensions
//     vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));

//     // cout << "Starting matrix multiplication" << endl;
//     // Perform the matrix multiplication
//     for (int i = 0; i < rowsOne; ++i) {
//         for (int j = 0; j < colsTwo; ++j) {
//             double currentSum = 0.0;
//             for (int k = 0; k < colsOne; ++k) {
//                 // cout << "Accessing matrixOne[" << i << "][" << k << "] and matrixTwo[" << k << "][" << j << "]" << endl;
//                 currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
//             }
//             result[i][j].value = currentSum;
//         }
//     }

//     // cout << "Matrix multiplication completed" << endl;
//     return result;
// }





void Network::setLearningRate(double lr) {
    for (const auto& layer : layers) {
        layer->setLearningRate(lr);
    }
    // cout << "Learning rate successfully set to: "<< lr<< endl;
}