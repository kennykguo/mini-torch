#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"
#include "ReLU.h"
#include "SoftmaxCrossEntropy.h"
#include "readCSV.h"
#include "constants.h"
#include "oneHotEncode.h"
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


int main() {
    cout << "Starting program with CUDA acceleration:\n";
    string filename = FILE_NAME;
    auto [labels, batches] = readCSV(filename);
    cout << "Labels size: " << labels.size() << "\n";
    cout << "Batches size: " << batches.size() << "\n";
    vector<vector<Neuron>> x = oneHotEncode(labels[0]);

    
    // Define model architecture
    Network network;
    network.addLayer(make_unique<LinearLayer>(784, 128));
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
