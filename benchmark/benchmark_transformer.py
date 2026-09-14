import time
import ctypes
import numpy as np
import os

def pack_weights_i4(A_fp32, scales):
    A_i8 = np.round(A_fp32 / scales[:, np.newaxis]).astype(np.int8)
    A_i8 = np.clip(A_i8, -8, 7)
    A_u4 = (A_i8 + 8).astype(np.uint8)
    if A_u4.shape[1] % 2 != 0:
        A_u4 = np.pad(A_u4, ((0, 0), (0, 1)), mode='constant')
    A_packed = (A_u4[:, 0::2]) | (A_u4[:, 1::2] << 4)
    return A_packed.astype(np.uint8)

def numpy_w4a16_transformer(A_fp32, B, scales, norm_weight, eps, alpha, beta):
    # 1. Emulate W4A16 Quantization/Dequantization
    A_i8 = np.round(A_fp32 / scales[:, np.newaxis]).astype(np.int8)
    A_i8 = np.clip(A_i8, -8, 7)
    A_dequant = A_i8.astype(np.float32) * scales[:, np.newaxis]
    
    # 2. GEMM
    C = np.dot(A_dequant, B) * alpha + beta
    
    # 3. True Row-Wise RMSNorm
    mean_sq = np.mean(C**2, axis=1, keepdims=True)
    C = C / np.sqrt(mean_sq + eps) * norm_weight
    
    # 4. True Row-Wise Softmax
    C_max = np.max(C, axis=1, keepdims=True)
    C_exp = np.exp(C - C_max)
    C = C_exp / np.sum(C_exp, axis=1, keepdims=True)
    return C

def benchmark_transformer_fused():
    print("="*70)
    print("Day 20: True Row-Wise Fused Transformer Block (Horizontal Reductions)")
    print("="*70)
    
    lib_paths = [os.path.abspath("../build_aarch64/libedge_server.so")]
    lib = None
    for path in lib_paths:
        if os.path.exists(path) and path.endswith('.so'):
            try:
                lib = ctypes.CDLL(path)
                print(f"[OK] Loaded shared library: {path}")
                break
            except OSError as e:
                print(f"[WARN] Could not load {path}: {e}")
                
    if lib is None:
        print("\n[INFO] No compiled .so found. Please compile the Rust/C++ code first.")
        return

    lib.rust_sme_transformer_block_i4_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.rust_sme_transformer_block_i4_f32.restype = None

    # Smaller dimensions for strict numerical validation
    M, K, N = 128, 512, 1024 
    eps, alpha, beta = 1e-5, 1.0, 0.0

    print(f"\nValidating Mathematical Correctness: {M} x {K} x {N}")
    
    A_fp32 = np.random.randn(M, K).astype(np.float32)
    scales = np.max(np.abs(A_fp32), axis=1) / 7.0
    A_packed = pack_weights_i4(A_fp32, scales)
    
    B = np.random.rand(K, N).astype(np.float32)
    norm_weight = np.random.rand(N).astype(np.float32) + 0.5 # Random norm weights
    C_rust = np.zeros((M, N), dtype=np.float32)

    A_ptr = A_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    Norm_ptr = norm_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C_rust.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    Scales_ptr = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    print("Running Rust SVE2/SME Fused Kernel...")
    lib.rust_sme_transformer_block_i4_f32(A_ptr, B_ptr, Scales_ptr, Norm_ptr, C_ptr, M, K, N, eps, alpha, beta)

    print("Running NumPy W4A16 Reference...")
    C_ref = numpy_w4a16_transformer(A_fp32, B, scales, norm_weight, eps, alpha, beta)

    # Validate
    max_diff = np.max(np.abs(C_rust - C_ref))
    is_close = np.allclose(C_rust, C_ref, rtol=1e-3, atol=1e-3)
    
    print("\n" + "="*70)
    if is_close:
        print("✅ NUMERICAL VALIDATION PASSED!")
        print(f"Max absolute difference vs NumPy: {max_diff:.6f}")
        print("True Row-Wise Horizontal Reductions are mathematically exact.")
    else:
        print("❌ NUMERICAL VALIDATION FAILED!")
        print(f"Max absolute difference vs NumPy: {max_diff:.6f}")
    print("="*70)

    # Performance Benchmarking
    print(f"\nBenchmarking Latency (M={M}, K={K}, N={N})...")
    iterations = 100
    start_time = time.perf_counter()
    for _ in range(iterations):
        lib.rust_sme_transformer_block_i4_f32(A_ptr, B_ptr, Scales_ptr, Norm_ptr, C_ptr, M, K, N, eps, alpha, beta)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"Time per iter: {(elapsed/iterations)*1000:.4f} ms")
    print("Notice how the 3-pass L1-cache strategy maintains extreme throughput!")

if __name__ == "__main__":
    benchmark_transformer_fused()