#pragma once
#include "Layer.h"

using namespace std;


// SoftmaxCrossEntropy class inherits from Layer, implementing polymorphism
class SoftmaxCrossEntropy : public Layer {
public:

    // Constructor takes in num of classes
    SoftmaxCrossEntropy(int num_classes);
    
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& grad_output) override;
    
    // Sets the labels as an attribute of the class
    void setLabels(const vector<vector<Neuron>>& labels);

    // Returns the loss stored by the layer
    double getLoss() const { return loss; }

private:
    int num_classes;
    vector<vector<Neuron>> input;
    vector<vector<Neuron>> output;
    vector<vector<Neuron>> labels;
    double loss;
};