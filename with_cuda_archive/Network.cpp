#include "Network.h"
#include <iostream>
#include <cassert>
#include <iostream>

using namespace std;


// The Network classes define the architecture, and you must choose the layers before you compile the program


// ReLU activation function
void Network::ReLU(vector<vector<Neuron>>& matrix) {
    // Loops over every row and every neuron in the row
    // Auto allows the compiler to automatically determine the type of a variable
    // Loops through every vector of vectors in matrix (every row)
    // The : convention is the range based loop -> 
    for (vector<Neuron>& row : matrix) {
        // Iterate through each neuron in the row
        for (Neuron& neuron : row) {
            // Apply ReLU activation function: set value to 0 if it's less than or equal to 0
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}



// Derivative of ReLU activation function
void Network::der_ReLU(vector<vector<Neuron>>& matrix) {
    for (vector<Neuron>& row : matrix) {
        // Iterate through each neuron in the row
        for (Neuron& neuron : row) {
            // Apply the derivative of ReLU: set value to 0 if it's less than or equal to 0
            if (neuron.value <= 0) {
                neuron.value = 0;
            }
        }
    }
}



vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {
    
    vector<vector<Neuron>> result;
    cudaMatrixMultiply(matrixOne, matrixTwo, result);
    return result;
}



