#include "SoftmaxCrossEntropy.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>

using namespace std;

SoftmaxCrossEntropy::SoftmaxCrossEntropy(int num_classes) : num_classes(num_classes), loss(0) {}

vector<vector<Neuron>> SoftmaxCrossEntropy::forward(const vector<vector<Neuron>>& input) {
    if (input.empty() || input[0].size() != this->num_classes) {
        throw runtime_error("Invalid input dimensions.");
    }
    if (labels.empty() || labels.size() != input.size() || labels[0].size() != this->num_classes) {
        throw runtime_error("Labels not set or have incorrect dimensions.");
    }

    this->input = input;
    this->output.resize(input.size(), vector<Neuron>(this->num_classes));
    loss = 0;
    
    // TODO: convert to CUDA kernel
    for (int i = 0; i < input.size(); ++i) {
        // Find the maximum value in the input for numerical stability
        double maxVal = input[i][0].value;
        for (int j = 1; j < this->num_classes; ++j) {
            maxVal = max(maxVal, input[i][j].value);
        }
        // Compute softmax values
        double sum = 0;
        for (int j = 0; j < this->num_classes; ++j) {
            this->output[i][j].value = exp(input[i][j].value - maxVal);
            sum += this->output[i][j].value;
        }
        // Normalize softmax values and compute cross-entropy loss
        for (int j = 0; j < this->num_classes; ++j) {
            this->output[i][j].value /= sum;
            if (labels[i][j].value == 1) {
                loss -= log(max(this->output[i][j].value, 1e-7)); // Prevent log(0)
            }
        }
    }
    // Compute average loss
    loss /= input.size();
    return this->output;
}

vector<vector<Neuron>> SoftmaxCrossEntropy::backward(const vector<vector<Neuron>>& gradOutput) {
    if (gradOutput.empty() || gradOutput.size() != this->output.size() || gradOutput[0].size() != this->num_classes) {
        throw runtime_error("Invalid gradOutput dimensions");
    }

    vector<vector<Neuron>> gradInput = this->output;
    // TODO: Convert to CUDA kernel
    for (size_t i = 0; i < gradInput.size(); ++i) {
        for (int j = 0; j < this->num_classes; ++j) {
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
}

