#pragma once
#include <vector>
#include "Neuron.h"

using namespace std;

// Abstract base class for all layers, enabling polymorphism
class Layer {
public:
    // Pure virtual functions that must be implemented by derived classes
    virtual vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) = 0;
    virtual vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) = 0;
    // Virtual destructor for proper cleanup of derived classes
    virtual void setLearningRate(double lr) { }
    virtual ~Layer() = default;

protected:
    double learningRate = 0.01;
};