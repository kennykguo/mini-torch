#include <iostream> // Include for cout
#include "Network.h"
// #include "Neuron.h"

int main() {
    cout << "Starting program\n";
    
    // Define the model layers
    int modelLayers[] = {3, 4, 2}; // Example: 3 input neurons, 4 neurons in hidden layer, 2 output neurons
    Network network(modelLayers, sizeof(modelLayers) / sizeof(modelLayers[0]));


    Neuron** input = new Neuron*[3];
    for (int i = 0; i < 3; ++i) {
        input[i] = new Neuron[3];
    }

    network.networkLayers[0].outputActivations = input;


    // Forward propagation
    for (int layerNum = 0; layerNum<network.numLayers - 1; layerNum++)
    {
        network.networkLayers[layerNum + 1].outputActivations = network.networkLayers[layerNum].forwardPropagate(network, network.networkLayers[layerNum].weightsMatrix, network.networkLayers[layerNum].fan_in, network.networkLayers[layerNum].fan_out, network.networkLayers[layerNum].outputActivations, network.networkLayers[layerNum].output_rows, network.networkLayers[layerNum].output_cols, network.networkLayers[layerNum].biasMatrix);
    }

    

    // // Cleanup (delete dynamic memory, etc.)
    // // Note: Replace with actual cleanup code

    return 0;
}
