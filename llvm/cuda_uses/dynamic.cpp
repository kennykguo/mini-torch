// Example: LLVM can generate specialized code based on input shapes
class DynamicNetwork {
    void forward(const Tensor& input) {
        if (!optimized_kernel || input.shape() != last_shape) {
            // LLVM generates new optimized code specifically for this input shape
            optimized_kernel = llvm_compiler.generateSpecializedKernel(input.shape());
            last_shape = input.shape();
        }
        optimized_kernel(input);
    }
};