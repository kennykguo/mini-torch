#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include <iomanip>

using namespace std;

const int INPUT_SIZE = 784; // Number of pixels in each image
const int BATCH_SIZE = 16; // Batch size for processing

tuple<vector<vector<Neuron>>, vector<vector<vector<Neuron>>>> readCSV(const string& filename) {
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

void printExample(const vector<Neuron>& label, const vector<vector<Neuron>>& example) {
    // Print the label
    cout << "Label: " << label[0].value << endl;

    // Print the pixels in a 28x28 grid
    cout << "Pixels:" << endl;
    for (size_t i = 0; i < example.size(); ++i) {
        for (size_t j = 0; j < example[i].size(); ++j) {
            if (j > 0 && j % 28 == 0) {
                cout << endl;
            }
            cout << example[i][j].value << " ";
        }
        cout << endl;
    }
}




int main() {

    cout << "Starting program with CUDA acceleration:\n";

    // Define our data
    string filename = "train.csv";
    auto result = readCSV(filename);

    // Define model layers manually
    Network network;
    network.networkLayers.push_back(LinearLayer(network, INPUT_SIZE, 4096)); // (784, 128)
    network.networkLayers.push_back(LinearLayer(network, 4096, 4096)); // (128, 64)
    network.networkLayers.push_back(LinearLayer(network, 4096, 10)); // (64, 10)

    // Check if batch and labels are not empty
    auto labels = get<0>(result);
    auto batches = get<1>(result);

    cout << "Labels size: " << labels.size() << "\n";
    cout << "Batches size: " << batches.size() << "\n";


    auto start_time = std::chrono::high_resolution_clock::now();

    if (!labels.empty() && !batches.empty()) {
        auto inputBatch = batches[0];

        cout << "Input batch size: " << inputBatch.size() << "\n";

        // Forward through the first layer
        auto layer1Output = network.networkLayers[0].forward(inputBatch);
        cout << "Layer 1 output size: " << layer1Output.size() << "\n";

        // Forward through the second layer
        auto layer2Output = network.networkLayers[1].forward(layer1Output);
        cout << "Layer 2 output size: " << layer2Output.size() << "\n";

        // Forward through the third layer
        auto finalOutput = network.networkLayers[2].forward(layer2Output);
        cout << "Final output size: " << finalOutput.size() << "\n";
    }

    cout << "Done" << endl;

    // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print the runtime
    cout << "Total runtime: " << duration.count() << " milliseconds" << endl;

    return 0;
}
