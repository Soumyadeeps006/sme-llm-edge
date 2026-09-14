#include "sme_gemm.h"
#include "thread_pool.h"
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <stdexcept>
#include <future>

#ifdef __aarch64__
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_neon.h>

// ARM SVE2 Optimized GEMV (Dequantize and Multiply-Accumulate)
extern "C" void gemv_s8_f32(const int8_t* w_quant, const float* x, float* out, const float* scales, int out_dim, int in_dim) {
    std::memset(out, 0, out_dim * sizeof(float));
    svbool_t ptrue = svptrue_b8();
    svbool_t ptrue_f32 = svptrue_f32();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < out_dim; ++i) {
        float scale = scales[i];
        float sum = 0.0f;
        const int8_t* row_ptr = w_quant + (i * in_dim);
        int j = 0;
        
        for (; j <= in_dim - (int)svcntp_b8(ptrue, ptrue); j += svcntp_b8(ptrue, ptrue)) {
            svint8_t w_vec = svld1_s8(ptrue, row_ptr + j);
            svfloat32_t x_vec = svld1_f32(ptrue_f32, x + j);
            svfloat32_t w_dequant = svmul_f32_x(ptrue_f32, svcvt_f32_s8_x(ptrue_f32, w_vec), scale);
            sum = svadda_f32(ptrue_f32, sum, svmul_f32_x(ptrue_f32, w_dequant, x_vec));
        }
        for (; j < in_dim; ++j) {
            sum += ((float)row_ptr[j] * scale) * x[j];
        }
        out[i] = sum;
    }
}

extern "C" void gemv_s8_f32_batched(const int8_t* W, const float* X, float* Out, const float* scales, int M, int out_dim, int in_dim) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < out_dim; ++i) {
            float sum = 0.0f;
            float scale = scales[i];
            const int8_t* row_ptr = W + (i * in_dim);
            const float* x_ptr = X + (m * in_dim);
            
            int j = 0;
            svbool_t ptrue = svptrue_b8();
            svbool_t ptrue_f32 = svptrue_f32();
            
            for (; j <= in_dim - (int)svcntp_b8(ptrue, ptrue); j += svcntp_b8(ptrue, ptrue)) {
                svint8_t w_vec = svld1_s8(ptrue, row_ptr + j);
                svfloat32_t x_vec = svld1_f32(ptrue_f32, x_ptr + j);
                svfloat32_t w_dequant = svmul_f32_x(ptrue_f32, svcvt_f32_s8_x(ptrue_f32, w_vec), scale);
                sum = svadda_f32(ptrue_f32, sum, svmul_f32_x(ptrue_f32, w_dequant, x_vec));
            }
            for (; j < in_dim; ++j) {
                sum += ((float)row_ptr[j] * scale) * x_ptr[j];
            }
            Out[m * out_dim + i] = sum;
        }
    }
}

extern "C" void gemm_s8_f32_sme(const int8_t* A, const float* B, float* C, const float* scales, int M, int N, int K) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        float scale = scales[i];
        const int8_t* row_A = A + (i * K);
        
        for (int j = 0; j < N; j += svcntd()) { 
            svfloat64_t c_vec = svld1_f64(svptrue_b64(), C + i * N + j);
            
            for (int k = 0; k < K; ++k) {
                float b_val = B[k * N + j]; 
                svfloat64_t b_vec = svdup_f64(b_val);
                svfloat64_t a_vec = svdup_f64((float)row_A[k] * scale);
                c_vec = svmopa_f64_m(svptrue_b64(), c_vec, a_vec, b_vec);
            }
            svst1_f64(svptrue_b64(), C + i * N + j, c_vec);
        }
    }
}
#else
// Baseline scalar implementations for host (x86_64/WSL) testing
extern "C" void gemv_s8_f32(const int8_t* w_quant, const float* x, float* out, const float* scales, int out_dim, int in_dim) {
    std::memset(out, 0, out_dim * sizeof(float));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        float scale = scales[i];
        for (int j = 0; j < in_dim; ++j) {
            sum += ((float)w_quant[i * in_dim + j] * scale) * x[j];
        }
        out[i] = sum; 
    }
}

extern "C" void gemv_s8_f32_batched(const int8_t* W, const float* X, float* Out, const float* scales, int M, int out_dim, int in_dim) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < out_dim; ++i) {
            float sum = 0.0f;
            float scale = scales[i];
            const int8_t* row_ptr = W + (i * in_dim);
            const float* x_ptr = X + (m * in_dim);
            for (int j = 0; j < in_dim; ++j) {
                sum += ((float)row_ptr[j] * scale) * x_ptr[j];
            }
            Out[m * out_dim + i] = sum;
        }
    }
}

extern "C" void gemm_s8_f32_sme(const int8_t* A, const float* B, float* C, const float* scales, int M, int N, int K) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        float scale = scales[i];
        for (int k = 0; k < K; ++k) {
            float a_val = (float)A[i * K + k] * scale;
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}
#endif

// ==========================================================
// Day 13/15/16: High-level C++ wrapper with Multi-Core Dispatch
// ==========================================================

extern "C" void rust_sme_gemm_f32(
    const float* a_ptr, const float* b_ptr, float* c_ptr,
    uint32_t m, uint32_t k, uint32_t n, float alpha, float beta);

constexpr uint32_t MACRO_TILE_M = 64;
constexpr uint32_t MACRO_TILE_N = 256;

void compute_sme_gemm_f32(
    const float* A, const float* B, float* C,
    uint32_t M, uint32_t K, uint32_t N, float alpha, float beta) 
{
    static std::vector<int> big_cores = edge_ai::CoreAffinity::get_big_core_ids();
    static edge_ai::ThreadPool pool(big_cores);
    
    std::vector<std::future<void>> futures;
    futures.reserve((M / MACRO_TILE_M + 1) * (N / MACRO_TILE_N + 1));

    for (uint32_t m_idx = 0; m_idx < M; m_idx += MACRO_TILE_M) {
        for (uint32_t n_idx = 0; n_idx < N; n_idx += MACRO_TILE_N) {
            uint32_t current_m = std::min(MACRO_TILE_M, M - m_idx);
            uint32_t current_n = std::min(MACRO_TILE_N, N - n_idx);

            const float* a_offset = A + (m_idx * K);
            const float* b_offset = B + n_idx; 
            float* c_offset = C + (m_idx * N) + n_idx;

            uint32_t next_n_idx = n_idx + MACRO_TILE_N;
            if (next_n_idx < N) {
                __builtin_prefetch(B + next_n_idx, 0, 3);
            }
            
            uint32_t next_m_idx = m_idx + MACRO_TILE_M;
            if (next_m_idx < M && n_idx == 0) { 
                __builtin_prefetch(A + (next_m_idx * K), 0, 3);
            }

            futures.push_back(pool.enqueue([=]() {
                rust_sme_gemm_f32(a_offset, b_offset, c_offset, current_m, K, current_n, alpha, beta);
            }));
        }
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

void compute_sme_gemm(
    const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
    uint32_t M, uint32_t K, uint32_t N, float alpha, float beta
) {
    if (A.size() != static_cast<size_t>(M) * K || 
        B.size() != static_cast<size_t>(K) * N || 
        C.size() != static_cast<size_t>(M) * N) {
        throw std::invalid_argument("Matrix dimensions do not match FFI requirements");
    }
    compute_sme_gemm_f32(A.data(), B.data(), C.data(), M, K, N, alpha, beta);
}

// ==========================================================
// Day 17: High-level C++ wrapper for Fused INT8 Dequantization
// ==========================================================

extern "C" void rust_sme_gemm_i8_f32(
    const int8_t* a_ptr, const float* b_ptr, float* c_ptr,
    const float* scales, uint32_t m, uint32_t k, uint32_t n, 
    float alpha, float beta);

void compute_sme_gemm_i8_f32(
    const int8_t* A, const float* B, float* C, const float* scales,
    uint32_t M, uint32_t K, uint32_t N, float alpha, float beta) 
{
    static std::vector<int> big_cores = edge_ai::CoreAffinity::get_big_core_ids();
    static edge_ai::ThreadPool pool(big_cores);
    
    std::vector<std::future<void>> futures;
    futures.reserve((M / MACRO_TILE_M + 1) * (N / MACRO_TILE_N + 1));

    for (uint32_t m_idx = 0; m_idx < M; m_idx += MACRO_TILE_M) {
        for (uint32_t n_idx = 0; n_idx < N; n_idx += MACRO_TILE_N) {
            uint32_t current_m = std::min(MACRO_TILE_M, M - m_idx);
            uint32_t current_n = std::min(MACRO_TILE_N, N - n_idx);

            const int8_t* a_offset = A + (m_idx * K);
            const float* b_offset = B + n_idx; 
            float* c_offset = C + (m_idx * N) + n_idx;
            const float* scales_offset = scales + m_idx;

            uint32_t next_n_idx = n_idx + MACRO_TILE_N;
            if (next_n_idx < N) {
                __builtin_prefetch(B + next_n_idx, 0, 3);
            }

            futures.push_back(pool.enqueue([=]() {
                rust_sme_gemm_i8_f32(a_offset, b_offset, c_offset, scales_offset, current_m, K, current_n, alpha, beta);
            }));
        }
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

// ==========================================================
// Day 18: High-level C++ wrapper for W4A16 Sub-byte Quantization
// ==========================================================

extern "C" void rust_sme_gemm_i4_f32(
    const uint8_t* a_packed_ptr, const float* b_ptr, float* c_ptr,
    const float* scales, uint32_t m, uint32_t k, uint32_t n, 
    float alpha, float beta);

void compute_sme_gemm_i4_f32(
    const uint8_t* A_packed, const float* B, float* C, const float* scales,
    uint32_t M, uint32_t K, uint32_t N, float alpha, float beta) 
{
    static std::vector<int> big_cores = edge_ai::CoreAffinity::get_big_core_ids();
    static edge_ai::ThreadPool pool(big_cores);
    
    std::vector<std::future<void>> futures;
    futures.reserve((M / MACRO_TILE_M + 1) * (N / MACRO_TILE_N + 1));

    uint32_t K_packed = (K + 1) / 2;

    for (uint32_t m_idx = 0; m_idx < M; m_idx += MACRO_TILE_M) {
        for (uint32_t n_idx = 0; n_idx < N; n_idx += MACRO_TILE_N) {
            uint32_t current_m = std::min(MACRO_TILE_M, M - m_idx);
            uint32_t current_n = std::min(MACRO_TILE_N, N - n_idx);

            const uint8_t* a_offset = A_packed + (m_idx * K_packed);
            const float* b_offset = B + n_idx; 
            float* c_offset = C + (m_idx * N) + n_idx;
            const float* scales_offset = scales + m_idx;

            uint32_t next_n_idx = n_idx + MACRO_TILE_N;
            if (next_n_idx < N) {
                __builtin_prefetch(B + next_n_idx, 0, 3);
            }

            futures.push_back(pool.enqueue([=]() {
                rust_sme_gemm_i4_f32(a_offset, b_offset, c_offset, scales_offset, current_m, K, current_n, alpha, beta);
            }));
        }
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

// ==========================================================
// Day 19: High-level C++ wrapper for Fused Transformer Block
// ==========================================================

extern "C" void rust_sme_transformer_block_i4_f32(
    const uint8_t* a_packed_ptr, const float* b_ptr, const float* scales,
    const float* norm_weight, float* c_ptr,
    uint32_t m, uint32_t k, uint32_t n, 
    float eps, float alpha, float beta);

void compute_transformer_block_i4_f32(
    const uint8_t* A_packed, const float* B, const float* scales, 
    const float* norm_weight, float* C,
    uint32_t M, uint32_t K, uint32_t N, 
    float eps, float alpha, float beta) 
{
    static std::vector<int> big_cores = edge_ai::CoreAffinity::get_big_core_ids();
    static edge_ai::ThreadPool pool(big_cores);
    
    std::vector<std::future<void>> futures;
    futures.reserve((M / MACRO_TILE_M + 1) * (N / MACRO_TILE_N + 1));

    uint32_t K_packed = (K + 1) / 2;

    for (uint32_t m_idx = 0; m_idx < M; m_idx += MACRO_TILE_M) {
        for (uint32_t n_idx = 0; n_idx < N; n_idx += MACRO_TILE_N) {
            uint32_t current_m = std::min(MACRO_TILE_M, M - m_idx);
            uint32_t current_n = std::min(MACRO_TILE_N, N - n_idx);

            const uint8_t* a_offset = A_packed + (m_idx * K_packed);
            const float* b_offset = B + n_idx; 
            const float* scales_offset = scales + m_idx;
            const float* norm_offset = norm_weight + n_idx; // Assuming norm is per-feature (N)
            float* c_offset = C + (m_idx * N) + n_idx;

            uint32_t next_n_idx = n_idx + MACRO_TILE_N;
            if (next_n_idx < N) {
                __builtin_prefetch(B + next_n_idx, 0, 3);
            }

            futures.push_back(pool.enqueue([=]() {
                rust_sme_transformer_block_i4_f32(
                    a_offset, b_offset, scales_offset, norm_offset, c_offset, 
                    current_m, K, current_n, eps, alpha, beta
                );
            }));
        }
    }

    for (auto& fut : futures) {
        fut.get();
    }
}