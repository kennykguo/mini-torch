#include <iostream>
#include <fstream>
#include <vector>

const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;

void read_mnist_images(const std::string &filename, std::vector<std::vector<float>> &images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&rows, sizeof(rows));
        file.read((char*)&cols, sizeof(cols));
        magic_number = __builtin_bswap32(magic_number);
        number_of_images = __builtin_bswap32(number_of_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        for (int i = 0; i < number_of_images; ++i) {
            std::vector<float> image(IMG_SIZE);
            for (int j = 0; j < IMG_SIZE; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                image[j] = pixel / 255.0f;
            }
            images.push_back(image);
        }
    }
}

void read_mnist_labels(const std::string &filename, std::vector<int> &labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        magic_number = __builtin_bswap32(magic_number);
        number_of_labels = __builtin_bswap32(number_of_labels);
        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((int)label);
        }
    }
}

int main() {
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;

    read_mnist_images("train-images-idx3-ubyte", train_images);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels);

    std::cout << "Number of training images: " << train_images.size() << std::endl;
    std::cout << "Number of training labels: " << train_labels.size() << std::endl;

    return 0;
}
