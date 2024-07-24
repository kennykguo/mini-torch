int main() {
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;

    read_mnist_images("train-images-idx3-ubyte", train_images);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels);

    int input_size = IMG_SIZE;
    int output_size = 10;  // Number of classes

    NeuralNetwork nn(input_size, output_size);

    for (int i = 0; i < train_images.size(); ++i) {
        std::vector<float> output(output_size, 0.0f);
        forward(train_images[i], nn.weights, output, input_size, output_size);

        // Print the output for the first image
        if (i == 0) {
            for (auto &val : output) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
