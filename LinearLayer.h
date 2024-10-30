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
    int inputSize;
    int outputSize;
    // Weights matrix
    vector<vector<Neuron>> weights;
    // Input matrix into layer
    vector<vector<Neuron>> lastInput;
};