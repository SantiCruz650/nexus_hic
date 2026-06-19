import torch
import triton
from kernels.gemm import nexus_gemm_kernel


def run_gemm(a_path, b_path, c_path, M, N, K):
    print(f"[RUNNER] Cargando A ({M}x{K}) y B ({K}x{N}) desde disco...")

    A = torch.from_file(a_path, shared=False, size=M * K, dtype=torch.float16).reshape(
        M, K
    )
    B = torch.from_file(b_path, shared=False, size=K * N, dtype=torch.float16).reshape(
        K, N
    )

    if not torch.cuda.is_available():
        print("[RUNNER] ERROR: No hay GPU disponible.")
        return False

    torch.cuda.synchronize()
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    C_gpu = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    print(f"[RUNNER] Disparando kernel con autotune dinámico...")
    nexus_gemm_kernel[grid](
        A_gpu,
        B_gpu,
        C_gpu,
        M,
        N,
        K,
        A_gpu.stride(0),
        A_gpu.stride(1),
        B_gpu.stride(0),
        B_gpu.stride(1),
        C_gpu.stride(0),
        C_gpu.stride(1),
    )

    torch.cuda.synchronize()
    print(f"[RUNNER] Kernel completado. Guardando resultado en {c_path}...")

    C_cpu = C_gpu.cpu()
    with open(c_path, "wb") as f:
        f.write(C_cpu.numpy().tobytes())

    print(f"[RUNNER] Verificando contra torch.mm...")
    A_ref = A.cuda()
    B_ref = B.cuda()
    C_ref = torch.mm(A_ref.float(), B_ref.float()).half()
    max_diff = (C_gpu.float() - C_ref.float()).abs().max().item()
    print(f"[RUNNER] Diferencia máxima con torch.mm: {max_diff:.6f}")

    print(f"[RUNNER] GEMM real completado: {M}x{K} @ {K}x{N} = {M}x{N}")
    return True
