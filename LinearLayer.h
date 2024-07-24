#pragma once
#include "Layer.h"
using namespace std;

// LinearLayer class inherits from Layer, implementing polymorphism
class LinearLayer : public Layer {
public:

    LinearLayer(int inputSize, int outputSize);
    
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;
    void setLearningRate(double lr) override { learningRate = lr; }

private:
    // All Linear layers store the input, and output size
    int inputSize;
    int outputSize;
    // Weights matrix
    vector<vector<Neuron>> weights;
    // Input matrix
    vector<vector<Neuron>> lastInput;
};