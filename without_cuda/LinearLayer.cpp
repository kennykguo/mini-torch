#include "LinearLayer.h"
#include "Network.h"
#include <iostream>
#include <cassert>

using namespace std;

// // Constructor
// // Use of : initiates member variables before the constructor body executes
// LinearLayer::LinearLayer(Network& network, int fan_in, int fan_out):network(network), fan_in(fan_in), fan_out(fan_out) {
    
//     cout << "Created a LinearLayer" << endl;

//     // Initialize weightsMatrix and biasMatrix
//     // vector.resize() changes the size of the vector
//     weightsMatrix.resize(fan_out, vector<Neuron>(fan_in));
//     biasMatrix.resize(fan_out);

//     // Initialize outputActivations with dummy values; it will be resized during forward pass
//     outputActivations.resize(1, vector<Neuron>(fan_out)); // Dummy initialization

//     cout << "Weights Matrix Shape:\n";
//     cout << "Rows: " << fan_in << '\n';
//     cout << "Columns: " << fan_out << '\n';
// }


// Ensure weightsMatrix and biasMatrix are properly initialized in LinearLayer constructor
LinearLayer::LinearLayer(Network& network, int fan_in, int fan_out)
    : network(network), fan_in(fan_in), fan_out(fan_out) {
    
    // Initialize weightsMatrix and biasMatrix
    weightsMatrix.resize(fan_in, vector<Neuron>(fan_out));
    // biasMatrix.resize(fan_out);

    // Fill weightsMatrix and biasMatrix with random values or initial values
    for (auto& row : weightsMatrix) {
        for (auto& neuron : row) {
            neuron.value = Neuron::randomValue();
        }
    }
    for (auto& neuron : biasMatrix) {
        neuron.value = Neuron::randomValue();
    }

    cout << "Created a LinearLayer" << endl;
    cout << "Weights Matrix Shape:\n";
    cout << "Rows: " << fan_in << '\n';
    cout << "Columns: " << fan_out << '\n';
}






// Forward propagation function
vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& inputMatrix) {
    
    // Size gets the size of the corresponding vector
    // Vectors can be indexed like an array
    int inputRows = inputMatrix.size();
    int inputCols = inputMatrix[0].size();

    // Passed this assertion
    assert(inputCols == fan_in && "Input matrix columns must match the number of input features (fan_in).");
    
    cout << "Forward propagating:\n";
    cout << "Input Rows: " << inputRows << "\n";
    cout << "Input Columns: " << inputCols << "\n";
    cout << "Weights Rows: " << fan_in << "\n";
    cout << "Weights Columns: " << fan_out << "\n";

    // Perform matrix multiplication
    vector<vector<Neuron>> output = network.matrixMultiply(inputMatrix, inputRows, inputCols, weightsMatrix, fan_in, fan_out);
    
    // // Add the bias
    // for (int i = 0; i < fan_out; ++i) {
    //     for (int j = 0; j < inputCols; ++j) {
    //         output[i][j].value += biasMatrix[i].value;
    //     }
    // }
    // // Debugging: print some values from the output to ensure the bias was added
    // cout << "Sample values from the output matrix after adding bias:\n";
    // if (!output.empty() && !output[0].empty()) {
    //     cout << "output[0][0]: " << output[0][0].value << "\n";
    //     cout << "output[0][1]: " << output[0][1].value << "\n";
    // }
    
    // Update the outputActivations
    this -> outputActivations = output;
    // this->output_rows = inputsRows;
    // this->output_cols = fan_out;
    return output;
}

