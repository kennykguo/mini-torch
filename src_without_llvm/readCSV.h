#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include <iomanip>
#include <memory>
#include "constants.h"
using namespace std;

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