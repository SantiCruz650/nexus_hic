import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    # Punteros a las matrices en VRAM
    a_ptr, b_ptr, c_ptr,
    # Dimensiones de las matrices
    M, N, K,
    # Strides (pasos de memoria) para indexación bidimensional
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Parámetros de Tiling (Metaparámetros de optimización)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel Triton optimizado con Tiling y ordenamiento por grupos (L2 Cache)
    """
    # Identificador del bloque de ejecución actual en la grilla 2D
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Agrupamiento de bloques (Group Scheduling) para maximizar el reuso de caché L2
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Crear offsets para los bloques de memoria internos (SRAM)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Punteros base para los bloques de A y B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Inicializar el acumulador del bloque C en registros internos de la GPU
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Ciclo de Tiling sobre la dimensión K (Multiplicación por bloques)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Máscaras de protección para evitar desbordes de memoria en tamaños irregulares
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)
        
        # Cargar bloques de datos a memoria compartida interna (SRAM)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Operación MMA (Matrix Multiply-Accumulate) por hardware Tensor Core
        accumulator = tl.dot(a, b, accumulator)
        
        # Avanzar los punteros al siguiente bloque sobre la dimensión K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Guardar el bloque C calculado de vuelta en la VRAM de forma segura
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_gemm(a, b):
    """
    Función puente que valida los datos de PyTorch, define la grilla de ejecución
    y orquesta el llamado al kernel Triton optimizado.
    """
    assert a.is_cuda and b.is_cuda, "Los tensores deben residir en GPU"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Dimensiones internas incompatibles para GEMM"
    
    # Alojar memoria en GPU para el resultado
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Configuración heurística óptima de bloques para la GPU T4
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Definición de la grilla de paralelización 1D mapeada internamente a 2D
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Lanzamiento del Kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return c
 
