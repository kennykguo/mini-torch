#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

class Network; // Forward declaration of Network class

class LinearLayer {
    
public:

    Network& network; // Reference to the associated Network object

    int fan_in;  // Number of input features
    int fan_out; // Number of output features

    // int output_rows; // Number of rows in the output
    // int output_cols; // Number of columns in the output

    vector<vector<Neuron>> weightsMatrix; // Weight matrix
    vector<Neuron> biasMatrix; // Bias vector
    vector<vector<Neuron>> outputActivations; // Output activations after forward pass

    // Constructor with initialization list
    LinearLayer(Network& network, int fan_in, int fan_out);

    // Forward propagation function
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& inputMatrix);
};
