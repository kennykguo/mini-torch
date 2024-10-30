// Header guard
#pragma once

#include <vector>
#include "Neuron.h"

using namespace std;

// Abstract base class for all layers
class Layer {
public:
    // Functions that must be implemented by derived classes
    virtual vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) = 0;
    virtual vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) = 0;
    virtual void setLearningRate(double lr) {}
    // Virtual destructor - if you delete the object through the base pointer, the destructor is called correctly
    virtual ~Layer() = default;

// Only the Layer class and its derived classes can access these attributes
protected:
    double learning_rate = 0.01;
};