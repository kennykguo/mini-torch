#include "Neuron.h"
#include <cstdlib> // Include for rand()

using namespace std;

Neuron::Neuron()
{
    // Initalize a random value when object instantiated
    value = randomValue();
    // Initalize the gradient to zero for now
    gradient = 0;
}
