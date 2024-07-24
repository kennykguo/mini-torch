
#include "LinearLayer.h"
#include "Network.h"
#include "Neuron.h"

#include <iostream>
using namespace std;

class Network;

// Default constructor
Layer::Layer(Network& network, int fan_in, int fan_out)
    : network(network), fan_in(fan_in), fan_out(fan_out) {
    cout << "Created a layer";
    this->fan_in = fan_in;
    this->fan_out = fan_out;

    // Create a 2D array for weightsMatrix
    weightsMatrix = new Neuron* [fan_out];
    for (int i = 0; i < fan_in; ++i) {
        weightsMatrix[i] = new Neuron[fan_in]; 
    }
    //Testing program
    cout << "Weights Matrix Shape:\n";
    cout << '\n';
    cout << "Rows: "<< fan_in;
    cout << '\n';
    cout << "Columns: " << fan_out;
    cout << '\n';

    // Create a 1D array for biasMatrix
    biasMatrix = new Neuron[fan_out]; // (1, fan_out)
}


Neuron** Layer::forwardPropagate(Network& network, Neuron** weightsMatrix, int weightsRows, int weightsCols, Neuron** inputMatrix, int inputsRows, int inputCols, Neuron* biasMatrix) {
    cout << "Forward propogating\n";
    cout << "weightRow: "<< weightsRows ;
    cout << "weightCol: "<< weightsCols ;
    cout << "inputRow: "<< inputsRows ;
    cout << "inputCol: "<< inputCols ;
    
    Neuron** output = network.matrixMultiply(weightsMatrix, weightsRows, weightsCols, inputMatrix, inputsRows, inputCols);

    for (int i = 0; i < weightsRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {  // Corrected loop range
            output[i][j].value += biasMatrix[j].value;
        }
    }
    this->output_rows = inputsRows;
    this->output_cols = weightsCols;
    return output;
}



