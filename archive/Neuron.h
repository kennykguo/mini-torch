#pragma once

#include <random>

using namespace std;

class Neuron
{
public:
    Neuron();
    double value;
    double gradient;
    static double randomValue(void) {
        return rand() / double(RAND_MAX); 
    }
};
