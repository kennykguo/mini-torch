#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include "ReLU.h"
#include "SoftmaxCrossEntropy.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include <iomanip>
#include <memory>
using namespace std;


// CONSTANTS
const int INPUT_SIZE = 784; // Number of pixels in each image
const int BATCH_SIZE = 16; // Batch size for processing
const int NUM_CLASSES = 10; // Number of classes (digits 0-9)
const string FILE_NAME = "train.csv";
const int EPOCHS = 3;

// Helper function for reading train.csv -> returns tuples of 2D vectors (matrices)
tuple< vector<vector<Neuron>>, vector<vector<vector<Neuron>>> > readCSV(const string& filename) {
    ifstream file(filename);
    vector<vector<Neuron>> labels;
    vector<vector<vector<Neuron>>> batches;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return make_tuple(labels, batches);
    }

    string line;
    bool firstLine = true;

    vector<Neuron> currentLabelBatch;
    vector<vector<Neuron>> currentBatch;

    while (getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue; // Skip header line
        }

        stringstream ss(line);
        string cell;

        vector<Neuron> example;
        Neuron labelNeuron;

        // Read label
        if (!getline(ss, cell, ',')) {
            cerr << "Error reading label from line: " << line << endl;
            continue;
        }
        labelNeuron.value = stod(cell);
        currentLabelBatch.push_back(labelNeuron);

        // Read pixels
        for (int j = 0; j < INPUT_SIZE; ++j) {
            if (!getline(ss, cell, ',')) {
                cerr << "Error reading pixel " << j << " from line: " << line << endl;
                break;
            }
            Neuron pixelNeuron;
            pixelNeuron.value = stod(cell) / 255.0; // Normalize pixel value
            example.push_back(pixelNeuron);
        }

        currentBatch.push_back(example);

        // If batch is full, add it to batches and reset
        if (currentBatch.size() == BATCH_SIZE) {
            batches.push_back(currentBatch);
            labels.push_back(currentLabelBatch);
            currentBatch.clear();
            currentLabelBatch.clear();
        }
    }

    // Add the last batch if it's not empty
    if (!currentBatch.empty()) {
        batches.push_back(currentBatch);
        labels.push_back(currentLabelBatch);
    }

    file.close();
    return make_tuple(labels, batches);
}

// One hot encode vector - input is a label, output is a one hot vector
vector<vector<Neuron>> oneHotEncode(const vector<Neuron>& labels) {
    vector<vector<Neuron>> encodedLabels;
    for (const auto& label : labels) {
        vector<Neuron> oneHot(NUM_CLASSES, Neuron{0.0});
        int labelValue = static_cast<int>(label.value);
        if (labelValue >= 0 && labelValue < NUM_CLASSES) {
            oneHot[labelValue].value = 1.0;
        }
        encodedLabels.push_back(oneHot);
    }
    return encodedLabels;
}


int main() {
    cout << "Starting program with CUDA acceleration:\n";
    // Define our data
    string filename = FILE_NAME;
    auto [labels, batches] = readCSV(filename);
    cout << "Labels size: " << labels.size() << "\n";
    cout << "Batches size: " << batches.size() << "\n";
    // Print the shape of the first batch of labels
    // if (!labels.empty()) {
    //     cout << "First batch of labels size: " << labels[0].size() << "\n";
    // }
    // Example of one-hot encoding and its shape
    vector<vector<Neuron>> x = oneHotEncode(labels[0]);
    // cout << "One-hot encoded labels size: " << x.size() << " || " << x[0].size() << "\n";

    
    // Define model architecture
    Network network;
    network.addLayer(make_unique<LinearLayer>(256, 128));
    network.addLayer(make_unique<ReLU>());
    network.addLayer(make_unique<LinearLayer>(128, 128));
    network.addLayer(make_unique<ReLU>());
    network.addLayer(make_unique<LinearLayer>(128, 10));
    network.addLayer(make_unique<SoftmaxCrossEntropy>(10));
    network.setLearningRate(0.001);
    auto start_time = chrono::high_resolution_clock::now();


    // Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Looping over all examples in the batch
        for (int i = 0; i < batches.size(); ++i) {
            // Create a batch of one hot labels
            vector<vector<Neuron>> oneHotLabels = oneHotEncode(labels[i]);
            // Forward pass
            vector<std::vector<Neuron>> output = network.forward(batches[i], oneHotLabels);
            // Backward pass
            network.backward();
            // Print loss every 100 batches
            if (i % 100 == 0) {  
                cout << "Epoch " << epoch << " || Batch " << i << "|| Loss: " << network.getLoss() << endl;
            }
        }
    }
    
    // Calculate the total runtime and print it
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Training completed. Total runtime: " << duration.count() << " milliseconds" << endl;

    return 0;
}
