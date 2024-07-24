#pragma once
#include "LinearLayer.h"
#include "Neuron.h"

class Layer; // Forward declaration

class Network
{
public:
    int numLayers;
    Layer* networkLayers;
    char* buffer;
    // Constructor
    Network(int* modelLayers, int numLayers);
    void ReLU(Neuron**, int rows, int cols);
    void der_ReLU(Neuron** matrix, int rows, int cols);
    Neuron** matrixMultiply(Neuron** matrixOne, int rowsOne, int colsOne, Neuron** matrixTwo, int rowsTwo, int colsTwo);
};



