#ifndef NEXUS_HIC_STORAGE_HPP
#define NEXUS_HIC_STORAGE_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <sys/mman.h>

namespace nexus_hic {

enum class DataType { FLOAT16, FLOAT32, INT8 };

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DataType dtype;
    void* data_ptr;
    size_t byte_size;

    ~Tensor() {
        if (owned_ && data_ptr) {
            munmap(data_ptr, byte_size);
        }
    }

    Tensor() : data_ptr(nullptr), byte_size(0), owned_(false) {}

    Tensor(Tensor&& other) noexcept
        : shape(std::move(other.shape)),
          strides(std::move(other.strides)),
          dtype(other.dtype),
          data_ptr(other.data_ptr),
          byte_size(other.byte_size),
          owned_(other.owned_)
    {
        other.data_ptr = nullptr;
        other.byte_size = 0;
        other.owned_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owned_ && data_ptr) munmap(data_ptr, byte_size);
            shape = std::move(other.shape);
            strides = std::move(other.strides);
            dtype = other.dtype;
            data_ptr = other.data_ptr;
            byte_size = other.byte_size;
            owned_ = other.owned_;
            other.data_ptr = nullptr;
            other.byte_size = 0;
            other.owned_ = false;
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void compute_strides() {
        strides.resize(shape.size());
        int64_t acc = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = acc;
            acc *= shape[i];
        }
    }

private:
    bool owned_;

    friend Tensor load_raw_tensor(const std::string&, const std::vector<int64_t>&, DataType);
};

Tensor load_raw_tensor(const std::string& filepath, const std::vector<int64_t>& shape, DataType dtype);

} // namespace nexus_hic

#endif
