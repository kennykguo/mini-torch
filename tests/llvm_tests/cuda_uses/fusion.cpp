// Without optimization - separate operations
Layer1->forward();  // CPU -> GPU transfer
Layer2->forward();  // GPU -> CPU -> GPU transfer
Layer3->forward();  // GPU -> CPU -> GPU transfer

// With LLVM optimization - fused operations
auto fused_layers = llvm_compiler.fuseOperations({Layer1, Layer2, Layer3});
fused_layers.forward();  // Single GPU operation, no intermediate transfers