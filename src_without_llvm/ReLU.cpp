#include "ReLU.h"
#include <algorithm>

using namespace std;

// TODO: Implement as a CUDA kernel
vector<vector<Neuron>> ReLU::forward(const vector<vector<Neuron>>& input) {
    this->input = input;
    vector<vector<Neuron>> output;
    cudaReLU(input, output, input.size(), input[0].size());
    return output;
}

// TODO: implement as a CUDA kernel
vector<vector<Neuron>> ReLU::backward(const vector<vector<Neuron>>& grad_output) {
    vector<vector<Neuron>> grad_input = grad_output;
    for (size_t i = 0; i < grad_input.size(); ++i) {
        for (size_t j = 0; j < grad_input[i].size(); ++j) {
            // ReLU gradient: 1 if input > 0, 0 otherwise
            grad_input[i][j].value *= (input[i][j].value > 0) ? 1 : 0;
        }
    }
    return grad_input;
}










// // TODO: Implement as a CUDA kernel
// vector<vector<Neuron>> ReLU::forward(const vector<vector<Neuron>>& input) {
//     this->input = input;
//     vector<vector<Neuron>> output = this->input;
//     for (auto& row : output) {
//         for (auto& neuron : row) {
//             // ReLU activation: max(0, x)
//             neuron.value = max(0.0, neuron.value);
//         }
//     }
//     return output;
// }

// // TODO: implement as a CUDA kernel
// vector<vector<Neuron>> ReLU::backward(const vector<vector<Neuron>>& grad_output) {
//     vector<vector<Neuron>> grad_input = grad_output;
//     for (size_t i = 0; i < grad_input.size(); ++i) {
//         for (size_t j = 0; j < grad_input[i].size(); ++j) {
//             // ReLU gradient: 1 if input > 0, 0 otherwise
//             grad_input[i][j].value *= (input[i][j].value > 0) ? 1 : 0;
//         }
//     }
//     return grad_input;
// }