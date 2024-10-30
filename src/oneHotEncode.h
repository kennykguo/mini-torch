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
#include "Neuron.h"
using namespace std;

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