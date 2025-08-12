// kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

// (N,3) 레이아웃: x,y,z 순서
__global__ void height_filter_kernel(float* data, int64_t N, float height) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float z = data[idx * 3 + 2];
        if (z >= height) {    
            data[idx * 3 + 0] = 0.0f;
            data[idx * 3 + 1] = 0.0f;
            data[idx * 3 + 2] = 0.0f;
        }
    }
}

extern "C" void launch_height_filter(float* data, int64_t N, float height) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    height_filter_kernel<<<blocks, threads>>>(data, N, height);
    cudaDeviceSynchronize();
}
