import time
import ctypes
import os

# Load the compiled Rust SME library (adjust path as needed for your cross-compilation target)
lib_path = os.path.join(
    os.path.dirname(__file__), 
    "..", "src", "sme_kernel", "rust_sme", "target", "aarch64-unknown-linux-gnu", "release", "librust_sme.so"
)
lib = ctypes.CDLL(lib_path)

# Define function signature with the new seq_len parameter
lib.rust_sme_transformer_block_i4_f32.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), # a_packed_ptr
    ctypes.POINTER(ctypes.c_float), # b_ptr
    ctypes.POINTER(ctypes.c_float), # scales
    ctypes.POINTER(ctypes.c_float), # norm_weight
    ctypes.POINTER(ctypes.c_float), # c_ptr
    ctypes.c_uint32,                # m (padded batch size)
    ctypes.c_uint32,                # k
    ctypes.c_uint32,                # n
    ctypes.c_uint32,                # seq_len (NEW: valid sequence length)
    ctypes.c_float,                 # eps
    ctypes.c_float,                 # alpha
    ctypes.c_float                  # beta
]
lib.rust_sme_transformer_block_i4_f32.restype = None

def benchmark_variable_length_fusion():
    print("🚀 Day 25: Variable-Length Fusion Benchmark")
    print("=" * 60)
    
    # Simulate continuous batching with variable sequence lengths
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    m_padded = 2048  # Max allocated batch size
    k = 4096
    n = 4096
    
    # Allocate dummy data
    k_packed = (k + 1) // 2
    a_packed = (ctypes.c_uint8 * (m_padded * k_packed))()
    b = (ctypes.c_float * (k * n))()
    scales = (ctypes.c_float * m_padded)()
    norm_weight = (ctypes.c_float * n)()
    c = (ctypes.c_float * (m_padded * n))()
    
    print(f"{'Seq Len':<10} | {'Time (ms)':<12} | {'Effective TFLOPs':<18} | {'Compute Saved':<15}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        # Warmup
        for _ in range(3):
            lib.rust_sme_transformer_block_i4_f32(
                a_packed, b, scales, norm_weight, c,
                m_padded, k, n, seq_len, 1e-5, 1.0, 0.0
            )
        
        # Benchmark
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            lib.rust_sme_transformer_block_i4_f32(
                a_packed, b, scales, norm_weight, c,
                m_padded, k, n, seq_len, 1e-5, 1.0, 0.0
            )
        end = time.perf_counter()
        
        avg_time = (end - start) / iterations
        
        # TFLOPs calculation: 2 * seq_len * k * n (effective GEMM compute)
        effective_ops = 2 * seq_len * k * n
        tflops = (effective_ops / 1e12) / avg_time
        
        # Percentage of compute saved compared to fully padded batch
        compute_saved = ((m_padded - seq_len) / m_padded) * 100
        
        print(f"{seq_len:<10} | {avg_time*1000:<12.2f} | {tflops:<18.2f} | {compute_saved:>6.1f}%")

if __name__ == "__main__":
    benchmark_variable_length_fusion()