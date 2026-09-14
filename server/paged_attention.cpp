// File: C:\Coding\edge_ai_project\server\paged_attention.cpp

#include "paged_attention.h"
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void compute_paged_attention(
    const std::vector<float>& batch_q,
    const std::vector<float>& k_cache_pool,
    const std::vector<float>& v_cache_pool,
    const std::vector<std::vector<int>>& batch_block_tables,
    const std::vector<int>& batch_seq_lens,
    std::vector<float>& batch_o,
    int hidden_dim,
    int block_size
) {
    int M = batch_seq_lens.size();
    
    // Parallelize across the batch dimension with dynamic scheduling for load balancing
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; ++m) {
        int seq_len = batch_seq_lens[m];
        if (seq_len == 0 || batch_block_tables[m].empty()) continue;
        
        float att_scale = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
        float inv_seq_len = 1.0f / std::max(1, seq_len);
        
        int tokens_processed = 0;
        const float* q_ptr = &batch_q[m * hidden_dim];
        float* o_ptr = &batch_o[m * hidden_dim];
        
        // Note: Ensure batch_o is zero-initialized by the caller before this function
        
        for (size_t b = 0; b < batch_block_tables[m].size(); ++b) {
            int phys_idx = batch_block_tables[m][b];
            int tokens_in_block = std::min(block_size, seq_len - tokens_processed);
            int block_base_offset = phys_idx * block_size * hidden_dim;
            
            // Hardware prefetch next block's K/V cache to hide memory latency
            if (b + 1 < batch_block_tables[m].size()) {
                int next_phys_idx = batch_block_tables[m][b + 1];
                int next_block_base_offset = next_phys_idx * block_size * hidden_dim;
                __builtin_prefetch(&k_cache_pool[next_block_base_offset], 0, 3); // Read, high temporal locality
                __builtin_prefetch(&v_cache_pool[next_block_base_offset], 0, 3);
            }
            
            for (int t = 0; t < tokens_in_block; ++t) {
                int token_offset = block_base_offset + t * hidden_dim;
                const float* k_ptr = &k_cache_pool[token_offset];
                const float* v_ptr = &v_cache_pool[token_offset];
                
                float score = 0.0f;
                
#ifdef __aarch64__
                // ARM Neon Vectorized Dot Product (Carried over from Day 11)
                int i = 0;
                float32x4_t acc_vec = vdupq_n_f32(0.0f);
                for (; i <= hidden_dim - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q_ptr + i);
                    float32x4_t k_vec = vld1q_f32(k_ptr + i);
                    acc_vec = vmlaq_f32(acc_vec, q_vec, k_vec);
                }
                score = vaddvq_f32(acc_vec); // Horizontal add
                
                // Scalar tail for non-multiples of 4
                for (; i < hidden_dim; ++i) {
                    score += q_ptr[i] * k_ptr[i];
                }
#else
                // Scalar fallback for x86 or generic builds
                for (int i = 0; i < hidden_dim; ++i) {
                    score += q_ptr[i] * k_ptr[i];
                }
#endif
                score *= att_scale;
                float weight = std::exp(score) * inv_seq_len; 
                
#ifdef __aarch64__
                // ARM Neon Vectorized V Accumulation (Carried over from Day 11)
                float32x4_t weight_vec = vdupq_n_f32(weight);
                int j = 0;
                for (; j <= hidden_dim - 4; j += 4) {
                    float32x4_t o_vec = vld1q_f32(o_ptr + j);
                    float32x4_t v_vec = vld1q_f32(v_ptr + j);
                    o_vec = vmlaq_f32(o_vec, weight_vec, v_vec);
                    vst1q_f32(o_ptr + j, o_vec);
                }
                // Scalar tail
                for (; j < hidden_dim; ++j) {
                    o_ptr[j] += weight * v_ptr[j];
                }
#else
                // Scalar fallback
                for (int j = 0; j < hidden_dim; ++j) {
                    o_ptr[j] += weight * v_ptr[j];
                }
#endif
            }
            tokens_processed += tokens_in_block;
        }
    }
}