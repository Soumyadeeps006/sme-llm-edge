#pragma once
#include <cstdint>
#include <vector>

// ==========================================================
// Day 2 & 3: Existing ARM SVE2/SME Optimized Kernels
// ==========================================================
extern "C" void gemv_s8_f32(const int8_t* w_quant, const float* x, float* out, const float* scales, int out_dim, int in_dim);
extern "C" void gemv_s8_f32_batched(const int8_t* W, const float* X, float* Out, const float* scales, int M, int out_dim, int in_dim);
extern "C" void gemm_s8_f32_sme(const int8_t* A, const float* B, float* C, const float* scales, int M, int N, int K);

// ==========================================================
// Day 13, 17 & 18: Zero-Copy FFI Declarations for Rust SME GEMM
// ==========================================================
#ifdef __cplusplus
extern "C" {
#endif

// Day 13: FP32 Baseline
void rust_sme_gemm_f32(
    const float* A, const float* B, float* C,
    uint32_t M, uint32_t K, uint32_t N,
    float alpha, float beta
);

// Day 17: Fused INT8 Dequantization FFI Signature (W8A32)
void rust_sme_gemm_i8_f32(
    const int8_t* a_ptr, const float* b_ptr, float* c_ptr,
    const float* scales, uint32_t m, uint32_t k, uint32_t n, 
    float alpha, float beta
);

// Day 18: W4A16 Sub-byte Quantization FFI Signature
// A is packed INT4 (Row-Major: M x (K/2)), 2 weights per byte
void rust_sme_gemm_i4_f32(
    const uint8_t* a_packed_ptr,
    const float* b_ptr,
    float* c_ptr,
    const float* scales,
    uint32_t m, uint32_t k, uint32_t n, 
    float alpha, float beta
);

// ==========================================================
// Day 19: Fused Transformer Block FFI Signature (GEMM + RMSNorm + Softmax)
// ==========================================================
// Executes: C = Softmax(RMSNorm(GEMM_W4A16(A, B)))
// A is packed INT4 (Row-Major: M x (K/2))
// B is FP32 Activations (Row-Major: K x N)
// norm_weight is FP32 RMSNorm weights (Size: N)
// C is FP32 Output (Row-Major: M x N)
void rust_sme_transformer_block_i4_f32(
    const uint8_t* a_packed_ptr,
    const float* b_ptr,
    const float* scales,
    const float* norm_weight,
    float* c_ptr,
    uint32_t m, uint32_t k, uint32_t n, 
    float eps, float alpha, float beta
);

#ifdef __cplusplus
}
#endif

// High-level C++ wrapper for the zero-copy Rust SME kernel (Day 13/15/16)
void compute_sme_gemm_f32(
    const float* A, const float* B, float* C,
    uint32_t M, uint32_t K, uint32_t N,
    float alpha = 1.0f, float beta = 0.0f
);

// Backward-compatible vector-based wrapper
void compute_sme_gemm(
    const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
    uint32_t M, uint32_t K, uint32_t N,
    float alpha = 1.0f, float beta = 0.0f
);

// Day 17: High-level C++ wrapper for Fused INT8 Dequantization
void compute_sme_gemm_i8_f32(
    const int8_t* A, const float* B, float* C, const float* scales,
    uint32_t M, uint32_t K, uint32_t N, float alpha = 1.0f, float beta = 0.0f
);

// Day 18: High-level C++ wrapper for W4A16 Sub-byte Quantization
void compute_sme_gemm_i4_f32(
    const uint8_t* A_packed, const float* B, float* C, const float* scales,
    uint32_t M, uint32_t K, uint32_t N, float alpha = 1.0f, float beta = 0.0f
);

// Day 19: High-level C++ wrapper for Fused Transformer Block
void compute_transformer_block_i4_f32(
    const uint8_t* A_packed, const float* B, const float* scales, 
    const float* norm_weight, float* C,
    uint32_t M, uint32_t K, uint32_t N, 
    float eps = 1e-5f, float alpha = 1.0f, float beta = 0.0f
);