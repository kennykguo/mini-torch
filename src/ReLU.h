#pragma once
#include "Layer.h"
#include <llvm/IR/Module.h>         // For LLVM Module class
#include <llvm/IR/LLVMContext.h>    // For LLVM Context, which holds all the LLVM state
#include <llvm/IR/IRBuilder.h>       // For building LLVM IR programmatically
#include <llvm/Support/raw_ostream.h> // For raw output streams for debugging

// ReLU class inherits from Layer
class ReLU : public Layer {
public:
    // Constructor
    ReLU();

    // Overridden methods for forward and backward passes
    virtual std::vector<std::vector<Neuron>> forward(const std::vector<std::vector<Neuron>>& input) override;
    virtual std::vector<std::vector<Neuron>> backward(const std::vector<std::vector<Neuron>>& grad_output) override;

private:
    std::unique_ptr<llvm::LLVMContext> context; // Unique pointer to hold LLVM context
    std::unique_ptr<llvm::Module> module;       // Unique pointer for the LLVM module
    llvm::IRBuilder<> *builder;                  // LLVM IR Builder for constructing IR
    vector<vector<Neuron>> input;
    // Function to generate LLVM IR specifically for the ReLU operation
    void LLVMReLU();  // Rename to indicate it's for ReLU IR generation
};

// External CUDA function declaration
extern "C" void cudaReLU(const std::vector<std::vector<Neuron>>& input, 
                          std::vector<std::vector<Neuron>>& output, 
                          int rows, int cols);
