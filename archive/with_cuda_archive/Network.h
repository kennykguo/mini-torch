#pragma once
#include <vector>
#include "LinearLayer.h"
#include "Neuron.h"

using namespace std;

class Network {

public:

    int numLayers; // Number of layers in the network

    // TODO: Update this so the user manually has to set this
    vector<LinearLayer> networkLayers; // Vector of LinearLayer objects

// ------------------------------------------------------------------------------------------------------------------
    // ReLU activation function (takes in a reference vector of vectors (a matrix))
    void ReLU(vector<vector<Neuron>>& matrix);

    // Derivative of ReLU activation function (takes in a reference vector of vectors (a matrix))
    void der_ReLU(vector<vector<Neuron>>& matrix);

    // Matrix multiplication function
    // Takes in two matrices of neurons and their corresponding rows and columns, and returns a matrice
    // Original matrices are not modified
    vector<vector<Neuron>> matrixMultiply(const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne, const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo);
};

extern "C" void cudaMatrixMultiply(const std::vector<std::vector<Neuron>>& A, 
                                   const std::vector<std::vector<Neuron>>& B, 
                                   std::vector<std::vector<Neuron>>& C);