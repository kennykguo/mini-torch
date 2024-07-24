#pragma once
#include "Layer.h"

using namespace std;


// SoftmaxCrossEntropy class inherits from Layer, implementing polymorphism
class SoftmaxCrossEntropy : public Layer {
public:

    // Constructor takes in num of classes
    SoftmaxCrossEntropy(int numClasses);
    
    // Virtual functions from Layer class, overridden here
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;
    vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;
    
    // Sets the labels as an attribute of the class
    void setLabels(const vector<vector<Neuron>>& labels);

    // Returns the loss stored by the layer
    double getLoss() const { return loss; }

private:
    int numClasses;
    vector<vector<Neuron>> lastInput;
    vector<vector<Neuron>> lastOutput;
    vector<vector<Neuron>> labels;
    double loss;
};