#include "Network.h"
#include <iostream>

using namespace std;

void Network::addLayer(unique_ptr<Layer> layer) {
    // dynamic_cast is used for runtime type checking
    // It attempts to cast the Layer pointer to a SoftmaxCrossEntropy pointer
    if (dynamic_cast<SoftmaxCrossEntropy*>(layer.get())) {
        lossLayer = dynamic_cast<SoftmaxCrossEntropy*>(layer.get());
    }
    // Move the unique_ptr into the layers vector
    layers.push_back(move(layer));
}


vector<vector<Neuron>> Network::forward(const vector<vector<Neuron>>& input, const vector<vector<Neuron>>& labels) {
    // Before every forward propagation, clear the 3D matrix
    layerOutputs.clear();

    // Add the inputs to the layerOutputs matrix
    layerOutputs.push_back(input);

    // If the lossLayer exists, set the labels before the forward pass
    if (lossLayer) {
        // cout << "Setting labels size: " << labels.size() << " || " << (labels.empty() ? 0 : labels[0].size()) << endl;
        lossLayer->setLabels(labels);
    }

    // Forward pass through each layer
    vector<vector<Neuron>> current = input;
    for (const auto& layer : layers) {
        // cout << "Layer forward pass" << endl;
        // cout << "Current size: " << current.size() << " || " << current[0].size() << endl;
        current = layer->forward(current);
        layerOutputs.push_back(current);
    }

    // Compute loss if the lossLayer is set
    if (lossLayer) {
        loss = lossLayer->getLoss();
    }

    return current;
}

void Network::backward() {
    vector<vector<Neuron>> grad = layers.back()->backward(layerOutputs.back());

    for (int i = layers.size() - 2; i >= 0; --i) {
        grad = layers[i]->backward(grad);
    }
}

vector<vector<Neuron>> Network::matrixMultiply(
    const vector<vector<Neuron>>& matrixOne, int rowsOne, int colsOne,
    const vector<vector<Neuron>>& matrixTwo, int rowsTwo, int colsTwo) {
    
    vector<vector<Neuron>> result;

    cudaMatrixMultiply(matrixOne, matrixTwo, result);

    return result;
}


void Network::setLearningRate(double lr) {
    for (const auto& layer : layers) {
        layer->setLearningRate(lr);
    }
    cout << "Learning rate successfully set to: "<< lr<< endl;
}