#pragma once
#include "Layer.h"

using namespace std;

// ReLU class inherits from Layer
class ReLU : public Layer {
public:
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& grad_output) override;

private:
    vector<vector<Neuron>> input;
};