// With LLVM - CPU side optimized
// LLVM can optimize data preparation and result processing
// by generating optimized machine code for the specific CPU architecture
auto optimizedPrep = llvm_compiler.generateOptimizedFunction([](Data* data) {
    // LLVM optimizes this code for your specific CPU
    // - Uses CPU vector instructions (SSE, AVX)
    // - Better memory access patterns
    // - Loop unrolling
    prepareBatchData();
});