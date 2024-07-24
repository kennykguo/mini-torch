#include <cmath>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        weights.resize(input_size * output_size);
        for (auto &weight : weights) {
            weight = (rand() / float(RAND_MAX)) - 0.5f;
        }
    }

    std::vector<float> forward(const std::vector<float> &input) {
        std::vector<float> output(output_size, 0.0f);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                output[i] += input[j] * weights[j * output_size + i];
            }
            output[i] = 1.0f / (1.0f + exp(-output[i]));  // Sigmoid activation
        }
        return output;
    }

private:
    int input_size, output_size;
    std::vector<float> weights;
};
