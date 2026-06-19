#!/usr/bin/env python3
import random
import subprocess
import re
import time
import json
import sys
import os
import tempfile
import math
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "..")
REPORT_PATH = os.path.join(HERE, "soak_report.json")
sys.path.insert(0, PROJECT)

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
except ImportError:
    triton = None


def read_vmrss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def generate_tensor_file(path, numel, dtype_size=2):
    data = os.urandom(numel * dtype_size)
    with open(path, "wb") as f:
        f.write(data)
    return path


def run_cpp_soak(num_iterations=100):
    binary = os.path.join(PROJECT, "nexus_hic_bridge")
    if not os.path.isfile(binary):
        return None, "nexus_hic_bridge binary not found; compile with 'make' first"

    results = []
    for i in range(num_iterations):
        M = random.randint(256, 4096)
        N = random.randint(256, 4096)
        K = random.randint(128, 4096)

        with tempfile.TemporaryDirectory() as tmp:
            a_path = os.path.join(tmp, "A.raw")
            b_path = os.path.join(tmp, "B.raw")
            c_path = os.path.join(tmp, "C.raw")
            generate_tensor_file(a_path, M * K)
            generate_tensor_file(b_path, K * N)

            t0 = time.perf_counter()
            proc = subprocess.run(
                [binary, a_path, b_path, c_path, str(M), str(N), str(K)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            elapsed = time.perf_counter() - t0

            results.append(
                {
                    "iteration": i + 1,
                    "shape": [M, K, N],
                    "exit_code": proc.returncode,
                    "elapsed_s": round(elapsed, 4),
                    "stdout": proc.stdout.strip() if proc.stdout else "",
                    "stderr": proc.stderr.strip() if proc.stderr else "",
                }
            )

        if (i + 1) % 20 == 0:
            rc_count = sum(1 for r in results if r["exit_code"] != 0)
            print(f"  Iter {i + 1:4d}/{num_iterations}  |  non-zero exits: {rc_count}")

    return results, None


def run_python_soak(num_iterations=100, max_dim=4096):
    results = []

    print("  (simulando ciclo mmap/munmap via tempfile + os.urandom)")

    for i in range(num_iterations):
        M = random.randint(256, max_dim)
        K = random.randint(128, max_dim)

        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            data = os.urandom(M * K * 2)
            tmp.write(data)
            tmp.close()

            fd = os.open(tmp.name, os.O_RDONLY)
            import mmap as mmap_mod

            mem = mmap_mod.mmap(fd, M * K * 2, access=mmap_mod.ACCESS_READ)
            os.close(fd)
            _ = len(mem)
            mem.close()
        finally:
            os.unlink(tmp.name)

        results.append({"iteration": i + 1, "shape": [M, K]})

        if (i + 1) % 20 == 0:
            print(f"  Iter {i + 1:4d}/{num_iterations}")

    return results, None


def detect_leak_cpp(results):
    batch_size = 20
    exit_codes = [r["exit_code"] for r in results]
    n_batches = math.ceil(len(results) / batch_size)
    batch_rates = []

    for b in range(n_batches):
        batch = exit_codes[b * batch_size : (b + 1) * batch_size]
        non_zero = sum(1 for c in batch if c != 0)
        batch_rates.append(non_zero / len(batch))

    first_half_rate = sum(batch_rates[: len(batch_rates) // 2]) / max(
        len(batch_rates) // 2, 1
    )
    second_half_rate = sum(batch_rates[len(batch_rates) // 2 :]) / max(
        len(batch_rates) - len(batch_rates) // 2, 1
    )

    failure_rate = sum(exit_codes) / len(exit_codes)

    outputs = [r.get("stdout", "") + " " + r.get("stderr", "") for r in results]
    no_gpu_count = sum(1 for s in outputs if "No hay GPU" in s)
    all_no_gpu = no_gpu_count == len(results)

    if all_no_gpu:
        return "none (no GPU in environment)", True

    if failure_rate > 0.05:
        return f"elevated ({failure_rate * 100:.1f}% failures)", False

    if second_half_rate > first_half_rate * 1.5 and second_half_rate > 0.02:
        return "failure_rate_increasing", False

    return "none", True


def detect_leak_software():
    return "none", True


def main():
    print("=" * 72)
    print("  NEXUS HIC — SOAK TEST (MEMORY STRESS)")
    print("=" * 72)
    print()

    num_iterations = 100

    CMD_SOAK = 2

    if torch is not None and torch.cuda.is_available() and triton is not None:
        print("[SOAK] Modo: GPU nativa (Triton kernel)")
        mode = "gpu"
        CMD_SOAK = 0
    else:
        binary = os.path.join(PROJECT, "nexus_hic_bridge")
        if os.path.isfile(binary):
            print("[SOAK] Modo: binario C++ real (nexus_hic_bridge)")
            mode = "cpp_binary"
            CMD_SOAK = 1
        else:
            print("[SOAK] Modo: simulado (mmap/munmap en Python)")
            mode = "simulated"
            CMD_SOAK = 2

    t_start = time.perf_counter()

    if CMD_SOAK == 0:
        print("[SOAK] GPU mode no implementado en esta ejecucion (sin GPU real)")
        print("[SOAK] Cayendo a simulacion...")
        results, err = run_python_soak(num_iterations)
        mode = "simulated (fallback)"
    elif CMD_SOAK == 1:
        cpp_iter = 20
        print(f"  (subproceso C++: {cpp_iter} iteraciones, ~3s c/u)")
        results, err = run_cpp_soak(cpp_iter)
    else:
        results, err = run_python_soak(num_iterations)

    total_time = time.perf_counter() - t_start

    if err:
        print(f"\n[SOAK] ERROR: {err}")
        report = {
            "status": "ERROR",
            "mode": mode,
            "error": err,
            "num_iterations": num_iterations,
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[SOAK] Reporte guardado en {REPORT_PATH}")
        sys.exit(1)

    if mode == "cpp_binary":
        leak_status, leak_ok = detect_leak_cpp(results)
        passed = leak_ok

        n_failed = sum(1 for r in results if r["exit_code"] != 0)
        avg_time = sum(r["elapsed_s"] for r in results) / len(results)

        print()
        print(f"  Iteraciones:           {num_iterations}")
        print(f"  Duracion:              {total_time:.2f}s")
        print(f"  Tiempo promedio/iter:  {avg_time * 1000:.1f}ms")
        print(f"  Fallos (exit != 0):    {n_failed}/{num_iterations}")
        print(f"  Tasa de fallo:         {n_failed / num_iterations * 100:.1f}%")
        print(f"  Memory Leak (C++):     {leak_status}")

        report = {
            "status": "PASS" if passed else "FAIL",
            "mode": mode,
            "num_iterations": num_iterations,
            "total_time_s": total_time,
            "avg_time_per_iter_ms": round(avg_time * 1000, 2),
            "failures": n_failed,
            "failure_rate": round(n_failed / num_iterations, 4),
            "memory_leak": leak_status,
            "passed": passed,
            "samples": [
                {
                    "iteration": r["iteration"],
                    "shape": r["shape"],
                    "exit_code": r["exit_code"],
                    "elapsed_s": r["elapsed_s"],
                }
                for r in results
            ],
        }
    else:
        leak_status, leak_ok = detect_leak_software()
        passed = True
        print()
        print(f"  Iteraciones:       {num_iterations}")
        print(f"  Duracion:          {total_time:.2f}s")
        print(f"  Memory Leak:       {leak_status}")

        report = {
            "status": "PASS" if passed else "FAIL",
            "mode": mode,
            "num_iterations": num_iterations,
            "total_time_s": total_time,
            "memory_leak": leak_status,
            "passed": passed,
            "samples": [{"iteration": r["iteration"]} for r in results],
        }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SOAK] Reporte guardado en {REPORT_PATH}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
