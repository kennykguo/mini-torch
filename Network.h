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
        const vector<vector<Neuron>>& matrix_one, int rows_one, int cols_one,
        const vector<vector<Neuron>>& matrix_two, int rows_two, int cols_two);

private:
    // Vector of unique_ptr to Layer objects
    vector<unique_ptr<Layer>> layers;
    // Three dimensions because we store all of the outputs here
    vector<vector<vector<Neuron>>> layer_outputs;
    // Pointer to the SoftmaxCrossEntropy layer
    SoftmaxCrossEntropy* loss_layer;
    // Store the current loss calculated
    double loss;
};

// External C function declaration for CUDA matrix multiplication
extern "C" void cudaMatrixMultiply(const vector<vector<Neuron>>& A, 
                                   const vector<vector<Neuron>>& B, 
                                   vector<vector<Neuron>>& C);