class ReLU : public Layer {
private:
    OptimizedReLU optimizer;
    vector<vector<Neuron>> lastInput;

public:
    vector<vector<Neuron>> forward(const vector<vector<Neuron>>& input) override {
        lastInput = input;
        vector<vector<Neuron>> output = input;
        
        // Use LLVM-optimized version instead of loops
        optimizer.process(output);
        
        return output;
    }
};