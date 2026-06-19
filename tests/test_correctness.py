#!/usr/bin/env python3
import torch
import triton
import time
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "correctness_report.json")
sys.path.insert(0, os.path.join(HERE, ".."))
from kernels.gemm import nexus_gemm_kernel


def relative_frobenius_error(C_triton, C_torch):
    num = (C_triton.float() - C_torch.float()).norm(p="fro")
    den = C_torch.float().norm(p="fro")
    return (num / den).item()


def max_ulp_error(C_triton, C_torch):
    C_t = C_triton.float()
    C_r = C_torch.float()
    return (C_t - C_r).abs().max().item()


def run_single_test(M, N, K, label=""):
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nexus_gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    C_ref = torch.mm(A.float(), B.float()).half()
    frob_err = relative_frobenius_error(C, C_ref)
    max_err = max_ulp_error(C, C_ref)

    ops = 2.0 * M * N * K
    tflops = ops / elapsed / 1e12

    return {
        "label": label or f"{M}x{K} @ {K}x{N}",
        "M": M,
        "N": N,
        "K": K,
        "frobenius_error": frob_err,
        "max_ulp_error": max_err,
        "elapsed_s": elapsed,
        "tflops": tflops,
        "passed": frob_err < 1e-3,
    }


def print_report(results, total_time):
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])

    print()
    print("=" * 88)
    print(f"  NEXUS HIC — CORRECTNESS REPORT")
    print("=" * 88)
    print(
        f"  {'Test':<30} {'Frobenius Err':<15} {'Max ULP':<12} {'TFLOPS':<10} {'Status':<8}"
    )
    print(f"  {'-' * 30} {'-' * 15} {'-' * 12} {'-' * 10} {'-' * 8}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['label']:<30} {r['frobenius_error']:<15.6e} {r['max_ulp_error']:<12.6f} {r['tflops']:<10.3f} {status:<8}"
        )
    print("=" * 88)
    print(
        f"  Total: {len(results)} tests | PASS: {passed} | FAIL: {failed} | Time: {total_time:.2f}s"
    )
    print("=" * 88)
    return failed


def main():
    if not torch.cuda.is_available():
        print("[TEST] ERROR: No hay GPU disponible. Pruebas de correctness omitidas.")
        report = {"status": "SKIPPED", "reason": "No GPU available", "tests": []}
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[TEST] Reporte guardado en {REPORT_PATH}")
        sys.exit(0)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    test_cases = [
        (1024, 1024, 1024, "1024x1024 (pow2)"),
        (2048, 2048, 2048, "2048x2048 (pow2)"),
        (4096, 4096, 4096, "4096x4096 (pow2)"),
        (1027, 769, 512, "1027x769  (irregular)"),
        (511, 1023, 257, "511x1023  (irregular)"),
        (769, 1027, 128, "769x1027  (irregular)"),
        (4096, 1024, 2048, "4096x1024 (rectangle)"),
        (256, 4096, 128, "256x4096  (rectangle)"),
        (128, 128, 128, "128x128   (small)"),
        (2, 2, 2, "2x2       (tiny)"),
    ]

    all_results = []
    t_start = time.perf_counter()

    for M, N, K, label in test_cases:
        try:
            r = run_single_test(M, N, K, label)
            all_results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"  {label:<30} frob={r['frobenius_error']:.2e}  maxerr={r['max_ulp_error']:.4f}  tflops={r['tflops']:.2f}  {status}"
            )
        except Exception as e:
            all_results.append(
                {
                    "label": label,
                    "M": M,
                    "N": N,
                    "K": K,
                    "frobenius_error": -1,
                    "max_ulp_error": -1,
                    "elapsed_s": -1,
                    "tflops": 0,
                    "passed": False,
                    "error": str(e),
                }
            )
            print(f"  {label:<30} ERROR: {e}")

    total_time = time.perf_counter() - t_start
    n_failed = print_report(all_results, total_time)

    report = {
        "status": "PASS" if n_failed == 0 else "FAIL",
        "total_tests": len(all_results),
        "passed": len(all_results) - n_failed,
        "failed": n_failed,
        "total_time_s": total_time,
        "threshold": 1e-3,
        "tests": [{k: v for k, v in r.items()} for r in all_results],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[TEST] Reporte JSON guardado en {REPORT_PATH}")

    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
