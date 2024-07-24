#include "Neuron.h"
#include <cstdlib> // Include for rand()

Neuron::Neuron() {
    value = randomValue();
    gradient = 0;
}

double Neuron::randomValue() {
    return rand() / double(RAND_MAX); // Generate a random value between 0 and 1
}
