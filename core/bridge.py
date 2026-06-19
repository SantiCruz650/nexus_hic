"""Python bridge — punto de entrada llamado desde C++ embebido."""


def run_pipeline(a_path, b_path, c_path, M, N, K):
    from kernels.runner import run_gemm

    return run_gemm(a_path, b_path, c_path, M, N, K)
