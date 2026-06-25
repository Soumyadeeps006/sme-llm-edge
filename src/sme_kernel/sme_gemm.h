#ifndef SME_GEMM_H
#define SME_GEMM_H

#include <vector>
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform quantized matrix-vector multiplication (GEMV)
 * out = W * x
 * 
 * W: Weight matrix of size M x K (int8_t)
 * x: Input vector of size K (float32_t)
 * out: Output vector of size M (float32_t)
 * scales: Scaling factors for dequantization of size M (float32_t)
 */
void gemv_s8_f32(const int8_t* W, const float* x, float* out, const float* scales, int M, int K);

/**
 * Perform quantized matrix-matrix multiplication (GEMM)
 * Out = A * B
 * 
 * A: Quantized activation matrix of size M x K (int8_t)
 * B: Quantized weight matrix of size K x N (int8_t)
 * Out: Output matrix of size M x N (float32_t)
 * scale_A: Scalar scale for A
 * scale_B: Array of scales for B of size N (float32_t)
 */
void gemm_s8_f32(const int8_t* A, const int8_t* B, float* Out, float scale_A, const float* scale_B, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // SME_GEMM_H
