import torch
import os


def generate(M=1024, N=1024, K=1024, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)

    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(K, N, dtype=torch.float16)

    with open(os.path.join(out_dir, "A.raw"), "wb") as f:
        f.write(A.numpy().tobytes())
    with open(os.path.join(out_dir, "B.raw"), "wb") as f:
        f.write(B.numpy().tobytes())

    print(f"Datos generados: A ({M}x{K}), B ({K}x{N}) en {out_dir}/")
    return True


if __name__ == "__main__":
    generate()
