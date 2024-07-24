#pragma once

class Neuron {
public:
    Neuron();
    double value;
    double gradient;

    static double randomValue();
};
