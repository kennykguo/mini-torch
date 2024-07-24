#pragma once

// Forward declaration of Network class to avoid circular dependency
class Network;

#include "Neuron.h" // Include Neuron.h for Neuron class


class Layer {
public:
    int fan_in;
    int fan_out;
    int output_rows;
    int output_cols;
    Neuron** weightsMatrix;
    Neuron* biasMatrix;
    Neuron** outputActivations;
    Network& network; // Reference to a Network object

    // Constructor with initialization list
    Layer(Network& network, int fan_in, int fan_out);
    Neuron** forwardPropagate(Network& network, Neuron** weightsMatrix, int weightsRows, int weightsCols, Neuron** inputMatrix, int inputsRows, int inputCols, Neuron* biasMatrix);
};

