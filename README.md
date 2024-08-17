# mini-torch

## Introduction

Mini PyTorch is a lightweight C++ library for building and training neural networks. It provides the essential components for constructing and managing neural networks, including various types of layers and functionalities for training and evaluation.

I tried to optimize the CUDA kernel function for matrix multiplication with memory coalescing and tiling (still a beginner (: )

## Classes

### `Neuron`

Represents a single neuron with properties such as `value` to store activation values.

### `Layer`

- **Description:** Abstract base class for all network layers.
- **Key Methods:**
  - `virtual vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) = 0;`
  - `virtual vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) = 0;`
  - `virtual void setLearningRate(double lr) { }`

### `LinearLayer`

- **Description:** Fully connected layer that performs a linear transformation.
- **Key Methods:**
  - Inherits from `Layer`.

### `ReLU`

- **Description:** Activation function layer that applies the ReLU (Rectified Linear Unit) function.
- **Key Methods:**
  - Inherits from `Layer`.

### `SoftmaxCrossEntropy`

- **Description:** Combines the softmax activation function with cross-entropy loss for classification tasks.
- **Key Methods:**
  - `vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override;`
  - `vector<vector<Neuron>> backward(const vector<vector<Neuron>>& gradOutput) override;`
  - `void setLabels(const vector<vector<Neuron>>& labels);`

### `Network`

- **Description:** Manages the sequence of layers and handles the forward and backward passes, including setting the learning rate.
- **Key Methods:**
  - `void addLayer(unique_ptr<Layer> layer);`
  - `vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input, const vector<vector<Neuron>>& labels);`
  - `void backward();`
  - `void setLearningRate(double lr);`
