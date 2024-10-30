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
    
    cout << "Entering matrixMultiply function" << endl;
    cout << "matrixOne dimensions: " << matrixOne.size() << "x" << (matrixOne.empty() ? 0 : matrixOne[0].size()) << endl;
    cout << "matrixTwo dimensions: " << matrixTwo.size() << "x" << (matrixTwo.empty() ? 0 : matrixTwo[0].size()) << endl;
    cout << "Passed dimensions: " << rowsOne << "x" << colsOne << " * " << rowsTwo << "x" << colsTwo << endl;

    // Ensure the dimensions are correct
    assert(colsOne == rowsTwo && "Matrix dimensions must match for multiplication.");
    assert(rowsOne == matrixOne.size() && "rowsOne doesn't match matrixOne size");
    assert(colsTwo == matrixTwo[0].size() && "colsTwo doesn't match matrixTwo column size");

    cout << "Initializing result matrix" << endl;
    // // Initialize result matrix with the correct dimensions
    vector<vector<Neuron>> result(rowsOne, vector<Neuron>(colsTwo));

    cout << "Starting matrix multiplication" << endl;
    // Perform the matrix multiplication
    for (int i = 0; i < rowsOne; ++i) {
        for (int j = 0; j < colsTwo; ++j) {
            double currentSum = 0.0;
            for (int k = 0; k < colsOne; ++k) {
                // cout << "Accessing matrixOne[" << i << "][" << k << "] and matrixTwo[" << k << "][" << j << "]" << endl;
                currentSum += matrixOne[i][k].value * matrixTwo[k][j].value;
            }
            result[i][j].value = currentSum;
        }
    }

    cout << "Matrix multiplication completed" << endl;
    return result;
}



