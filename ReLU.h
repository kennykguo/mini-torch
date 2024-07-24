#pragma once
#include "Layer.h"

using namespace std;

// ReLU class inherits from Layer, implementing polymorphism
class ReLU : public Layer {
public:
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;

private:
    vector<vector<Neuron>> lastInput;
};