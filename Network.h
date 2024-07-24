#pragma once
#include <vector>
#include <memory>
#include "Layer.h"
#include "SoftmaxCrossEntropy.h"

using namespace std;

class Network {
public:
    // unique_ptr is a smart pointer that owns and manages another object through a pointer
    void addLayer(unique_ptr<Layer> layer);

    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input, const vector<vector<Neuron>>& labels);

    void backward();

    double getLoss() const { return loss; }
    
    void setLearningRate(double lr);

    static vector<vector<Neuron>> matrixMultiply(
        const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
        const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo);

private:
    // Vector of unique_ptr to Layer objects, allowing polymorphism
    vector<unique_ptr<Layer>> layers;
    // There are three dimensions because we store all of the outputs here
    vector<vector<vector<Neuron>>> layerOutputs;
    // Pointer to the SoftmaxCrossEntropy layer
    SoftmaxCrossEntropy* lossLayer;
    // Store the current loss calculated
    double loss;
};

// External C function declaration for CUDA matrix multiplication
extern "C" void cudaMatrixMultiply(const vector<vector<Neuron>>& A, 
                                   const vector<vector<Neuron>>& B, 
                                   vector<vector<Neuron>>& C);