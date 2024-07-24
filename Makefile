NVCC = nvcc
CXX = g++
# NVCCFLAGS = -std=c++17 -O3
# CXXFLAGS = -std=c++17 -O3

NVCCFLAGS = -std=c++17 -O3 -g
CXXFLAGS = -std=c++17 -O3 -g

SOURCES = main.cpp Network.cpp LinearLayer.cpp Neuron.cpp ReLU.cpp SoftmaxCrossEntropy.cpp

CUDA_SOURCES = cuda_operations.cu

OBJECTS = $(SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

TARGET = nn

all: $(TARGET)

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET)