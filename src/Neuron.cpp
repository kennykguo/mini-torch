#include "Neuron.h"
#include <cstdlib> // Include for rand()

Neuron::Neuron() : value(0.0), gradient(0.0) {}
Neuron::Neuron(double val) : value(val), gradient(0.0) {} // Implement the new constructor