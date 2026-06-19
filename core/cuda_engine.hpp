#ifndef NEXUS_HIC_CUDA_ENGINE_HPP
#define NEXUS_HIC_CUDA_ENGINE_HPP

#include <stdexcept>
#include <cstddef>
#include <string>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define NEXUS_HIC_HAS_CUDA 1
#else
#define NEXUS_HIC_HAS_CUDA 0
#endif

namespace nexus_hic {

class CUDABuffer {
    void* ptr_;
    size_t size_;
public:
    CUDABuffer() : ptr_(nullptr), size_(0) {}

    explicit CUDABuffer(size_t nbytes) : size_(nbytes) {
#if NEXUS_HIC_HAS_CUDA
        auto err = cudaMalloc(&ptr_, nbytes);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
#else
        throw std::runtime_error("CUDA runtime not available (rebuild with CUDA toolkit)");
#endif
    }

    ~CUDABuffer() {
#if NEXUS_HIC_HAS_CUDA
        if (ptr_) cudaFree(ptr_);
#endif
    }

    CUDABuffer(CUDABuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CUDABuffer& operator=(CUDABuffer&& other) noexcept {
        if (this != &other) {
#if NEXUS_HIC_HAS_CUDA
            if (ptr_) cudaFree(ptr_);
#endif
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    void* ptr() const { return ptr_; }
    size_t size() const { return size_; }
};

inline void cuda_copy_to_device(void* dst, const void* src, size_t nbytes) {
#if NEXUS_HIC_HAS_CUDA
    auto err = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
#else
    (void)dst; (void)src; (void)nbytes;
    throw std::runtime_error("CUDA runtime not available");
#endif
}

inline void cuda_copy_to_host(void* dst, const void* src, size_t nbytes) {
#if NEXUS_HIC_HAS_CUDA
    auto err = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
#else
    (void)dst; (void)src; (void)nbytes;
    throw std::runtime_error("CUDA runtime not available");
#endif
}

} // namespace nexus_hic

#endif
