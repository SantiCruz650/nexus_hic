# Nexus HIC

Enterprise-grade C++/Python/Triton hybrid engine for high-performance matrix multiplication (GEMM) on NVIDIA GPUs.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │       C++ Orchestrator          │
                    │    (main.cpp / engine.cpp)      │
                    │  Embedded Python (Py_Initialize) │
                    └──────────┬──────────────────────┘
                               │ PyImport_ImportModule
                    ┌──────────▼──────────────────────┐
                    │       Python Bridge              │
                    │      (core/bridge.py)            │
                    └──────────┬──────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼───────┐   ┌───────▼───────┐   ┌────────▼────────┐
  │   mmap I/O    │   │  Triton GEMM  │   │  torch.mm ref   │
  │ (torch.file)  │   │    Kernel     │   │  verification   │
  │  zero-copy    │   │  autotuned    │   │                 │
  └───────────────┘   └───────────────┘   └─────────────────┘
```

- **C++ layer**: mmap'd tensor loading (`storage.hpp`), embedded Python interpreter, RAII memory management
- **Python bridge**: Entry point called from C++, delegates to the kernel runner
- **Triton kernel**: `@triton.autotune` GEMM with 4 configs, Tensor Core `mma.sync`, fp16
- **Verification**: Kernel output validated against `torch.mm` via relative Frobenius error

## Prerequisites

- Linux x86_64 with NVIDIA GPU (Compute Capability 7.0+)
- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 3.0+
- g++ with C++17 support
- `python3-config` available on PATH

## Setup

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch triton numpy
```

## Build & Run

```bash
# Quick start (build + data generation + run)
./run_nexus.sh

# Or manually:
make                     # build the C++ binary
make data                # generate test tensors
./nexus_hic_bridge       # run the pipeline
```

The binary accepts optional arguments for custom tensor paths and dimensions:
```bash
./nexus_hic_bridge /path/to/A.raw /path/to/B.raw /path/to/C.raw 2048 2048 2048
```

## Testing

```bash
# Run the full CI suite
bash tests/run_tests.sh

# Or run individual test blocks
python3 tests/test_correctness.py   # Precision validation (10 cases)
python3 tests/test_performance.py   # Benchmark sweep (8 configs)
python3 tests/test_soak.py          # Memory stress test (C++ subprocess or Python sim)
```

All tests produce structured JSON reports in `tests/`. The CI script consolidates these into `tests/ci_report.json`.

### Test blocks

| Block | What it does | Threshold |
|-------|-------------|-----------|
| **Correctness** | 10 test cases (pow2, irregular, rectangular, tiny) vs `torch.mm` | Frobenius error < 1e-3 |
| **Performance** | 8-config sweep (256–4096), warmup + steady-state, speedup vs `torch.mm` | Reports TFLOPS & speedup |
| **Soak** | 3 modes: GPU (real kernel), C++ binary (20 iters), Python sim (100 iters) | Failure-rate leak detection |

## Project Structure

```
nexus_hic/
├── main.cpp                  # Portable entry point, path resolution
├── Makefile                  # Build system (python3-config)
├── run_nexus.sh              # Build & run script
├── pyproject.toml            # Python project metadata
├── LICENSE                   # MIT License
├── core/
│   ├── bridge.py             # Python bridge (called from C++)
│   ├── engine.cpp            # C++ orchestrator, Py_Initialize
│   ├── storage.hpp           # RAII Tensor (mmap/munmap, move semantics)
│   ├── storage.cpp           # load_raw_tensor implementation
│   └── cuda_engine.hpp       # Optional CUDA RAII wrappers
├── kernels/
│   ├── gemm.py               # @triton.autotune GEMM kernel
│   └── runner.py             # Kernel launcher + verification
├── data/
│   └── generate_data.py      # Random fp16 test data generator
└── tests/
    ├── test_correctness.py   # Precision validation suite
    ├── test_performance.py   # Benchmark sweep suite
    ├── test_soak.py          # Memory stress test suite
    └── run_tests.sh          # CI orchestrator
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXUS_VENV` | `./venv` | Path to Python virtual environment |
| `PYTHONPATH` | (auto) | Set by `main.cpp` for module resolution |
