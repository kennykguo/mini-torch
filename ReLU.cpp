#include "ReLU.h"
#include <algorithm>

using namespace std;

vector<vector<Neuron>> ReLU::forward(const vector<vector<Neuron>>& input) {
    lastInput = input;
    vector<vector<Neuron>> output = input;
    for (auto& row : output) {
        for (auto& neuron : row) {
            // ReLU activation: max(0, x)
            neuron.value = max(0.0, neuron.value);
        }
    }
    return output;
}

vector<vector<Neuron>> ReLU::backward(const vector<vector<Neuron>>& gradOutput) {
    vector<vector<Neuron>> gradInput = gradOutput;
    for (size_t i = 0; i < gradInput.size(); ++i) {
        for (size_t j = 0; j < gradInput[i].size(); ++j) {
            // ReLU gradient: 1 if input > 0, 0 otherwise
            gradInput[i][j].value *= (lastInput[i][j].value > 0) ? 1 : 0;
        }
    }
    return gradInput;
}