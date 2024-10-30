#pragma once

class Neuron {
public:
    Neuron();
    Neuron(double val); // Add this constructor
    double value;
    double gradient;
};