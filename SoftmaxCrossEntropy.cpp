#include "SoftmaxCrossEntropy.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>

using namespace std;

SoftmaxCrossEntropy::SoftmaxCrossEntropy(int numClasses) : numClasses(numClasses), loss(0) {}

vector<vector<Neuron>> SoftmaxCrossEntropy::forward(const vector<vector<Neuron>>& input) {
    // cout << "Entering SoftmaxCrossEntropy::forward" << endl;

    // Debug the input size
    // cout << "Input size: " << input.size() << ", Input[0] size: " << (input.empty() ? 0 : input[0].size()) << endl;

    if (input.empty() || input[0].size() != numClasses) {
        throw runtime_error("Invalid input dimensions");
    }

    // Code got up to here

    // cout << "Labels size: " << labels.size() << ", Labels[0] size: " << (labels.empty() ? 0 : labels[0].size()) << endl;

    if (labels.empty() || labels.size() != input.size() || labels[0].size() != numClasses) {
        throw runtime_error("Labels not set or have incorrect dimensions");
    }

    lastInput = input;
    lastOutput.resize(input.size(), vector<Neuron>(numClasses));
    loss = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        // Find the maximum value in the input for numerical stability
        double maxVal = input[i][0].value;
        for (int j = 1; j < numClasses; ++j) {
            maxVal = max(maxVal, input[i][j].value);
        }

        // Compute softmax values
        double sum = 0;
        for (int j = 0; j < numClasses; ++j) {
            lastOutput[i][j].value = exp(input[i][j].value - maxVal);
            sum += lastOutput[i][j].value;
        }

        // Normalize softmax values and compute cross-entropy loss
        for (int j = 0; j < numClasses; ++j) {
            lastOutput[i][j].value /= sum;
            if (labels[i][j].value == 1) {
                loss -= log(max(lastOutput[i][j].value, 1e-7)); // Prevent log(0)
            }
        }
    }

    // Compute average loss
    loss /= input.size();
    // cout << "Exiting SoftmaxCrossEntropy::forward with loss: " << loss << endl;
    return lastOutput;
}

vector<vector<Neuron>> SoftmaxCrossEntropy::backward(const vector<vector<Neuron>>& gradOutput) {
    if (gradOutput.empty() || gradOutput.size() != lastOutput.size() || gradOutput[0].size() != numClasses) {
        throw runtime_error("Invalid gradOutput dimensions");
    }

    vector<vector<Neuron>> gradInput = lastOutput;
    for (size_t i = 0; i < gradInput.size(); ++i) {
        for (int j = 0; j < numClasses; ++j) {
            // Compute gradient of softmax cross-entropy
            gradInput[i][j].value = (gradInput[i][j].value - labels[i][j].value) / gradInput.size();
        }
    }
    return gradInput;
}

void SoftmaxCrossEntropy::setLabels(const vector<vector<Neuron>>& labels) {
    if (labels.empty() || labels[0].empty()) {
        throw runtime_error("Labels not set or have incorrect dimensions");
    }

    this->labels = labels;

    // Print the size of the labels to verify they are set correctly
    // cout << "Setting labels size: " << labels.size() << " || " << labels[0].size() << endl;
}

