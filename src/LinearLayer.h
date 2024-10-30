#pragma once
#include "Layer.h"
using namespace std;

// LinearLayer class inherits from Layer
class LinearLayer : public Layer {
public:
    LinearLayer(int input_size, int output_size);
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& grad_output) override;
    void setLearningRate(double lr) override { learning_rate = lr; }

private:
    int input_size;
    int output_size;
    vector<vector<Neuron>> weights;
    vector<vector<Neuron>> input;
};