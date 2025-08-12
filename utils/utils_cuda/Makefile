# Compiler
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -O2 -fdiagnostics-color=always -g -shared -fPIC \
           -I/home/hjkwon/dev/dlpack/include

# CUDA flags
CUDAFLAGS = -Xcompiler -fPIC --expt-relaxed-constexpr

# Python & pybind11
PYTHON = python3.10
PYTHON_INC = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT = $(shell $(PYTHON)-config --extension-suffix)

# CUDA
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

# Includes and linking
INCLUDES = -I$(OPEN3D_INCLUDE) -I$(EIGEN_INCLUDE) -I$(FMT_INCLUDE) \
           -I$(CUDA_INCLUDE) -I/home/hjkwon/dev/dlpack/include $(PYTHON_INC)

LDFLAGS = -L$(OPEN3D_LIB) -l$(OPEN3D_LIBNAME) \
          -L$(CUDA_LIB) -lcudart

# Files
CU_SRC = kernels.cu
CPP_SRC = bindings.cpp
OBJ = kernels.o
TARGET = mycuda$(PYTHON_EXT)

all: $(TARGET)

# Compile .cu to .o
$(OBJ): $(CU_SRC)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(INCLUDES)

# Compile and link .so
$(TARGET): $(CPP_SRC) $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJ)
