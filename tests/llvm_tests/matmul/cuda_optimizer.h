// LLVMCudaOptimizer.h
#pragma once
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <cuda.h>
#include <cuda_runtime.h>

class LLVMCudaOptimizer {
public:
    LLVMCudaOptimizer();
    void optimizeMatrixMultiply(float* d_A, float* d_B, float* d_C, 
                               int M, int N, int K, cudaStream_t stream);

private:
    std::unique_ptr<llvm::LLVMContext> Context;
    std::unique_ptr<llvm::Module> Module;
    std::unique_ptr<llvm::IRBuilder<>> Builder;
    
    void initializeLLVMNVPTX();
    llvm::Function* createMatMulKernel();
    void optimizeKernel(llvm::Function* F);
};