#include "LinearLayer.h"
#include "Network.h"
#include <random>
#include <cmath>
#include <iostream>

using namespace std;

LinearLayer::LinearLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
    // Used to seed the random number generator
    random_device rd;

    // mt19937 is a Mersenne Twister random number engine
    mt19937 gen(rd());

    // Normal_distribution generates random numbers according to a normal (Gaussian) distribution
    normal_distribution<> d(0, 1);

    //How does this line work?
    weights.resize(input_size, vector<Neuron>(output_size));

    for (vector<Neuron>& row : weights) {
        for (Neuron & weight : row) {
            // Initialize weights using Kaiming He initialization sampled from normal distribution
            weight.value = d(gen) * sqrt(2.0 / input_size);
        }
    }
}


vector<vector<Neuron>> LinearLayer::forward(const vector<vector<Neuron>>& input) {
    this->input = input;
    // Use Network's matrix multiplication method
    // (A, B) @ (B, C) = (A, C) -> 2D vector implementation
    // input.size() returns the batch size
    return Network::matrixMultiply(input, input.size(), this->input_size, this->weights, this->input_size, this->output_size);
}


vector<vector<Neuron>> LinearLayer::backward(const vector<vector<Neuron>>& grad_output) {
    
    // Tranpose weights matrix
    // Flip the dimensions of the weights matrix dimensions -> (output_size, input_size)
    vector<vector<Neuron>> weights_transposed(output_size, vector<Neuron>(input_size));
    // TODO: Implement as a CUDA kernel
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights_transposed[j][i] = weights[i][j];
        }
    }

    // Calculate gradient with respect to input
    // gradOutput.size() is the batch size
    vector<vector<Neuron>> grad_input = Network::matrixMultiply(grad_output, grad_output.size(), this->output_size, weights_transposed, this->output_size, this->input_size);

    // Calculate weight gradients
    vector<vector<Neuron>> input_transposed(input_size, vector<Neuron>(input.size()));
    // TODO: Implement as a CUDA kernel
    for (size_t i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input_size; ++j) {
            input_transposed[j][i] = input[i][j];
        }
    }
    vector<vector<Neuron>> weights_gradients = Network::matrixMultiply(input_transposed, input_size, input.size(), grad_output, grad_output.size(), output_size);

    // Update weights
    // TODO: Implement as some optimizer.step() function
    // TODO: Implement as a CUDA kernel
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights[i][j].value -= learning_rate  * weights_gradients[i][j].value;
        }
    }

    return grad_input;
}