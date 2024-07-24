#include "LinearLayer.h"
#include "Network.h"
#include <random>
#include <cmath>
#include <iostream>

using namespace std;

LinearLayer::LinearLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {

    // Used to seed the random number generator
    random_device rd;
    // mt19937 is a Mersenne Twister random number engine
    mt19937 gen(rd());

    // normal_distribution generates random numbers according to a normal (Gaussian) distribution
    normal_distribution<> d(0, 1);

    weights.resize(inputSize, vector<Neuron>(outputSize));

    for (auto& row : weights) {
        for (auto& weight : row) {
            // Reinitialize weights using Kaiming He initialization
            weight.value = d(gen) * sqrt(2.0 / inputSize);
        }
    }
}


vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& input) {
    lastInput = input;
    // cout<< "linear layer"<<endl;
    // Use Network's matrix multiplication method
    // (A, B) @ (B, C)
    
    return Network::matrixMultiply(input, input.size(), inputSize, weights, inputSize, outputSize);
}


vector<vector<Neuron>> LinearLayer::backward(const vector<vector<Neuron>>& gradOutput) {
    // Transpose weights for gradient calculation
    vector<vector<Neuron>> weightsTransposed(outputSize, vector<Neuron>(inputSize));
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightsTransposed[j][i] = weights[i][j];
        }
    }

    // Calculate gradient with respect to input
    vector<vector<Neuron>> gradInput = Network::matrixMultiply(gradOutput, gradOutput.size(), outputSize, weightsTransposed, outputSize, inputSize);

    // Calculate gradient with respect to weights
    vector<vector<Neuron>> lastInputTransposed(inputSize, vector<Neuron>(lastInput.size()));
    for (size_t i = 0; i < lastInput.size(); ++i) {
        for (int j = 0; j < inputSize; ++j) {
            lastInputTransposed[j][i] = lastInput[i][j];
        }
    }

    vector<vector<Neuron>> weightGradients = Network::matrixMultiply(lastInputTransposed, inputSize, lastInput.size(), gradOutput, gradOutput.size(), outputSize);

    // Update weights using SGD
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[i][j].value -= learningRate  * weightGradients[i][j].value;
        }
    }

    return gradInput;
}