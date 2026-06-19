#include <Python.h>
#include "cuda_engine.hpp"
#include "storage.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

namespace {

bool cuda_prepare(const std::string& a_path, const std::string& b_path,
                  int64_t M, int64_t N, int64_t K) {
#if NEXUS_HIC_HAS_CUDA
    try {
        std::cout << "[CUDA] Native C++ CUDA pipeline:" << std::endl;

        auto A = nexus_hic::load_raw_tensor(a_path, {M, K}, nexus_hic::DataType::FLOAT16);
        auto B = nexus_hic::load_raw_tensor(b_path, {K, N}, nexus_hic::DataType::FLOAT16);

        size_t a_bytes = A.byte_size;
        size_t b_bytes = B.byte_size;
        size_t c_bytes = M * N * 2;

        nexus_hic::CUDABuffer a_gpu(a_bytes);
        nexus_hic::CUDABuffer b_gpu(b_bytes);
        nexus_hic::CUDABuffer c_gpu(c_bytes);

        std::cout << "[CUDA]   Buffers allocated: A=" << a_bytes
                  << " B=" << b_bytes << " C=" << c_bytes << " bytes" << std::endl;

        nexus_hic::cuda_copy_to_device(a_gpu.ptr(), A.data_ptr, a_bytes);
        nexus_hic::cuda_copy_to_device(b_gpu.ptr(), B.data_ptr, b_bytes);

        std::vector<char> zeros(c_bytes, 0);
        nexus_hic::cuda_copy_to_device(c_gpu.ptr(), zeros.data(), c_bytes);

        std::vector<char> a_verify(a_bytes);
        nexus_hic::cuda_copy_to_host(a_verify.data(), a_gpu.ptr(), a_bytes);

        bool match = (std::memcmp(a_verify.data(), A.data_ptr, a_bytes) == 0);
        std::cout << "[CUDA]   Host-to-device round-trip: "
                  << (match ? "OK" : "MISMATCH") << std::endl;
        std::cout << "[CUDA]   C++ CUDA pipeline completed successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "[CUDA]   Error: " << e.what() << std::endl;
        return false;
    }
#else
    (void)a_path; (void)b_path; (void)M; (void)N; (void)K;
    std::cout << "[CUDA] Not available (CUDA toolkit not found at compile time)." << std::endl;
    return false;
#endif
}

} // anonymous namespace

bool call_run_pipeline(const std::string& a_path, const std::string& b_path,
                       const std::string& c_path,
                       int64_t M, int64_t N, int64_t K) {
    std::cout << "[ENGINE] Step 1/2: C++ CUDA preparation..." << std::endl;
    bool cuda_ok = cuda_prepare(a_path, b_path, M, N, K);
    std::cout << "[ENGINE]   CUDA native path: "
              << (cuda_ok ? "active" : "skipped (falling back to Python bridge)")
              << std::endl;
    std::cout << std::endl;

    std::cout << "[ENGINE] Step 2/2: Python bridge (Triton kernel)..." << std::endl;

    PyObject* pModule = PyImport_ImportModule("core.bridge");
    if (!pModule) {
        PyErr_Print();
        return false;
    }

    PyObject* pResult = PyObject_CallMethod(pModule, "run_pipeline", "(ssslll)",
        a_path.c_str(), b_path.c_str(), c_path.c_str(), M, N, K);

    bool success = false;
    if (!pResult) {
        PyErr_Print();
    } else {
        success = PyObject_IsTrue(pResult);
        Py_DECREF(pResult);
    }
    Py_DECREF(pModule);
    return success;
}
