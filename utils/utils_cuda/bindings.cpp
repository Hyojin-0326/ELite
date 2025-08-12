// bindings.cpp
#include <pybind11/pybind11.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// CUDA 런처 함수 선언
extern "C" void launch_height_filter(float* data, int64_t N, float height);

py::capsule height_filter(py::capsule dlpack_capsule, float height) {
    DLTensor* t = reinterpret_cast<DLTensor*>(dlpack_capsule.get_pointer());
    float* ptr = static_cast<float*>(t->data);
    int64_t N = t->shape[0];

    // CUDA 커널 실행
    launch_height_filter(ptr, N, height);

    return dlpack_capsule;
}

PYBIND11_MODULE(mycuda, m) {
    m.def("height_filter", &height_filter,
          "Apply height clamp to point cloud positions",
          py::arg("capsule"), py::arg("height"));
}
