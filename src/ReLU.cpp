#include "ReLU.h"                        // Include the header for the ReLU class
#include <llvm/IR/Verifier.h>            // For verifying the correctness of LLVM functions
#include <cuda_runtime.h>                // For CUDA runtime API
#include <vector>                        // For using std::vector

// External CUDA function declaration
extern "C" void cudaReLU(const std::vector<std::vector<Neuron>>& input, 
                          std::vector<std::vector<Neuron>>& output, 
                          int rows, int cols);

// Constructor: Initializes the LLVM context, module, and builder
ReLU::ReLU() {
    // Create a new LLVM context and assign it to the context attribute
    context = std::make_unique<llvm::LLVMContext>();
    // Create a new LLVM module and assign it to the module attribute, passing the context
    module = std::make_unique<llvm::Module>("ReLU Module", *context);
    // Create a new IRBuilder for constructing LLVM IR and assign it to the builder attribute
    builder = new llvm::IRBuilder<>(*context);
    // Call the function for ReLU-specific IR generation
    LLVMReLU(); 
}

// Function to generate LLVM IR for the ReLU operation
void ReLU::LLVMReLU() {
    // Define LLVM function: void relu_execute(double* input, double* output, int rows, int cols)
    llvm::FunctionType *relu_func_type = llvm::FunctionType::get(
        builder->getVoidTy(),                           // Return type: void
        {llvm::Type::getDoublePtrTy(*context),        // Input type: double pointer
         llvm::Type::getDoublePtrTy(*context),        // Output type: double pointer
         llvm::Type::getInt32Ty(*context),            // Rows type: int32
         llvm::Type::getInt32Ty(*context)},           // Columns type: int32
        false                                          // Not a variadic function
    );

    // Create the function in the module
    llvm::Function *relu_func = llvm::Function::Create(
        relu_func_type, llvm::Function::ExternalLinkage, "relu_execute", module.get()
    );

    auto args = relu_func->args().begin();            // Get function arguments
    llvm::Value *input = args++;                        // Input pointer
    llvm::Value *output = args++;                       // Output pointer
    llvm::Value *rows = args++;                        // Number of rows
    llvm::Value *cols = args++;                        // Number of columns

    // Create the entry basic block
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(*context, "entry", relu_func);
    builder->SetInsertPoint(entry);                      // Set insert point to the entry block
    // Currently, just a placeholder for the entry point
    builder->CreateRetVoid();                            // Return from the function
    // Verify the function for correctness
    llvm::verifyFunction(*relu_func);
    // Print the LLVM IR for debugging
    module->print(llvm::errs(), nullptr);
}


// Forward method for ReLU activation
std::vector<std::vector<Neuron>> ReLU::forward(const std::vector<std::vector<Neuron>>& input) {
    this->input = input;                                // Store the input for backward pass
    std::vector<std::vector<Neuron>> output(input.size(), std::vector<Neuron>(input[0].size())); // Prepare output vector
    // Call the CUDA kernel for the actual ReLU computation
    cudaReLU(input, output, input.size(), input[0].size());
    return output;                                      // Return the computed output
}

// Backward method for ReLU activation
// TODO: turn into cuda Kernel
std::vector<std::vector<Neuron>> ReLU::backward(const std::vector<std::vector<Neuron>>& grad_output) {
    std::vector<std::vector<Neuron>> grad_input = grad_output; // Initialize gradient input with output gradients
    for (size_t i = 0; i < grad_input.size(); ++i) {
        for (size_t j = 0; j < grad_input[i].size(); ++j) {
            // ReLU gradient: 1 if input > 0, 0 otherwise
            grad_input[i][j].value *= (input[i][j].value > 0) ? 1 : 0; // Apply ReLU gradient
        }
    }
    return grad_input;                                 // Return the computed gradients
}
