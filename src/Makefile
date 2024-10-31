# Define compilers
NVCC = nvcc
CXX = g++

# Compiler flags
NVCCFLAGS = -std=c++17 -O3 -g                             # CUDA compiler flags
CXXFLAGS = -std=c++17 -O3 -g -fexceptions `llvm-config --cxxflags`  # C++ compiler flags with LLVM

# Source files
SOURCES = main.cpp Network.cpp LinearLayer.cpp Neuron.cpp ReLU.cpp SoftmaxCrossEntropy.cpp
CUDA_SOURCES = MM.cu ReLUKernel.cu

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Target executable
TARGET = nn

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ `llvm-config --ldflags --libs`  # Add LLVM link flags

# Rule for compiling C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean target to remove object files and the executable
clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET)
