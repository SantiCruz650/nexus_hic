#!/usr/bin/env python3
import torch
import triton
import time
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "performance_report.json")
sys.path.insert(0, os.path.join(HERE, ".."))
from kernels.gemm import nexus_gemm_kernel


def benchmark_gemm(M, N, K, warmup_runs=2, bench_runs=10):
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.empty(M, N, dtype=torch.float16, device="cuda")

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    times_ns = []
    for i in range(warmup_runs + bench_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
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
        elapsed = time.perf_counter_ns() - t0
        times_ns.append(elapsed)

    warmup_time_ns = sum(times_ns[:warmup_runs]) / warmup_runs
    steady_times_ns = times_ns[warmup_runs:]
    avg_time_ns = sum(steady_times_ns) / len(steady_times_ns)
    min_time_ns = min(steady_times_ns)

    ops = 2.0 * M * N * K
    warmup_tflops = ops / (warmup_time_ns / 1e9) / 1e12
    avg_tflops = ops / (avg_time_ns / 1e9) / 1e12
    peak_tflops = ops / (min_time_ns / 1e9) / 1e12

    # torch.mm baseline
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    C_ref = torch.mm(A.float(), B.float())
    torch.cuda.synchronize()
    torch_time_ns = time.perf_counter_ns() - t0

    torch_tflops = ops / (torch_time_ns / 1e9) / 1e12
    speedup = torch_time_ns / avg_time_ns

    return {
        "M": M,
        "N": N,
        "K": K,
        "warmup_time_ms": warmup_time_ns / 1e6,
        "avg_time_ms": avg_time_ns / 1e6,
        "min_time_ms": min_time_ns / 1e6,
        "warmup_tflops": warmup_tflops,
        "avg_tflops": avg_tflops,
        "peak_tflops": peak_tflops,
        "torch_time_ms": torch_time_ns / 1e6,
        "torch_tflops": torch_tflops,
        "speedup_vs_torch": speedup,
    }


def print_report(results, total_time):
    print()
    print("=" * 110)
    print("  NEXUS HIC — PERFORMANCE BENCHMARK REPORT")
    print("=" * 110)
    header = (
        f"  {'Config':<22} {'Warmup(ms)':<12} {'Avg(ms)':<10} "
        f"{'Peak TFLOPS':<12} {'Avg TFLOPS':<12} {'torch(ms)':<10} {'Speedup':<8}"
    )
    print(header)
    print("  " + "-" * 86)
    for r in results:
        cfg = f"{r['M']}x{r['K']}@{r['K']}x{r['N']}"
        print(
            f"  {cfg:<22} {r['warmup_time_ms']:<12.3f} {r['avg_time_ms']:<10.3f} "
            f"{r['peak_tflops']:<12.2f} {r['avg_tflops']:<12.2f} {r['torch_time_ms']:<10.3f} {r['speedup_vs_torch']:<8.2f}x"
        )
    print("=" * 110)
    avg_speedup = sum(r["speedup_vs_torch"] for r in results) / len(results)
    peak_speedup = max(r["speedup_vs_torch"] for r in results)
    print(
        f"  Avg Speedup vs torch.mm: {avg_speedup:.2f}x  |  Peak Speedup: {peak_speedup:.2f}x  |  Total time: {total_time:.2f}s"
    )
    print("=" * 110)


def main():
    if not torch.cuda.is_available():
        print("[BENCH] ERROR: No hay GPU disponible. Benchmark omitido.")
        report = {
            "status": "SKIPPED",
            "reason": "No GPU available",
            "benchmarks": [],
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[BENCH] Reporte guardado en {REPORT_PATH}")
        sys.exit(0)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    test_cases = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 1024, 2048),
        (1024, 4096, 512),
        (2048, 512, 4096),
    ]

    all_results = []
    t_start = time.perf_counter()

    for M, N, K in test_cases:
        try:
            print(f"  Benchmarking {M}x{K} @ {K}x{N} ... ", end="", flush=True)
            r = benchmark_gemm(M, N, K)
            all_results.append(r)
            speedup_str = (
                f"{r['speedup_vs_torch']:.2f}x"
                if r["speedup_vs_torch"] >= 1.0
                else f"{1 / r['speedup_vs_torch']:.2f}x slower"
            )
            print(
                f"avg={r['avg_time_ms']:.2f}ms  peak={r['peak_tflops']:.1f}TFLOPS  speedup={speedup_str}"
            )
        except Exception as e:
            all_results.append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "error": str(e),
                    "speedup_vs_torch": 0,
                }
            )
            print(f"ERROR: {e}")

    total_time = time.perf_counter() - t_start

    valid = [r for r in all_results if "error" not in r]
    if valid:
        print_report(valid, total_time)

    report = {
        "status": "COMPLETED",
        "total_benchmarks": len(all_results),
        "total_time_s": total_time,
        "benchmarks": [{k: v for k, v in r.items()} for r in all_results],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[BENCH] Reporte JSON guardado en {REPORT_PATH}")

    n_errors = sum(1 for r in all_results if "error" in r)
    sys.exit(1 if n_errors > 0 else 0)


if __name__ == "__main__":
    main()
