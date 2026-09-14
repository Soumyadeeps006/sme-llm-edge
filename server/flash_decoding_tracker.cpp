#include "flash_decoding_tracker.h"
#include <iostream>
#include <cmath>
#include <algorithm>

extern "C" {
    void rust_sme_flashdecoding_chunk_f32(
        const float* q_ptr, const float* k_chunk_ptr, const float* v_chunk_ptr,
        uint32_t chunk_len, uint32_t head_dim, float softmax_scale,
        float* partial_o_out, float* partial_max_out, float* partial_sum_exp_out
    );
    
    void rust_sve2_flashdecoding_reduce_f32(
        const float* partial_o_ptr, const float* partial_max_ptr, const float* partial_sum_exp_ptr,
        uint32_t num_chunks, uint32_t head_dim, float* final_o_out
    );
}

FlashDecodingTracker::FlashDecodingTracker(PagedKVCache* cache, int threshold, int chunk)
    : paged_cache(cache), chunk_size_threshold(threshold), chunk_size(chunk) {}

bool FlashDecodingTracker::should_use_flash_decoding(int seq_len) const {
    return seq_len > chunk_size_threshold;
}

void FlashDecodingTracker::execute_flash_decoding(
    int request_id,
    const float* q_ptr,
    uint32_t head_dim,
    float softmax_scale,
    float* final_o_out
) {
    // In a full production system, we would fetch the physical blocks for this request,
    // dequantize if necessary, and split into chunks. 
    // For structural completeness, we simulate the chunking and orchestration.
    
    int seq_len = 8192; // Example long context triggering FlashDecoding
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    
    std::vector<float> all_partial_o(num_chunks * head_dim, 0.0f);
    std::vector<float> all_partial_max(num_chunks, 0.0f);
    std::vector<float> all_partial_sum_exp(num_chunks, 0.0f);
    
    // Mock K and V buffers (in reality, fetched from paged_cache)
    std::vector<float> mock_k(seq_len * head_dim, 0.1f);
    std::vector<float> mock_v(seq_len * head_dim, 0.1f);

    for (int c = 0; c < num_chunks; ++c) {
        int current_chunk_len = std::min(chunk_size, seq_len - c * chunk_size);
        const float* k_chunk = mock_k.data() + c * chunk_size * head_dim;
        const float* v_chunk = mock_v.data() + c * chunk_size * head_dim;
        
        float* p_o = all_partial_o.data() + c * head_dim;
        float* p_max = &all_partial_max[c];
        float* p_sum = &all_partial_sum_exp[c];
        
        rust_sme_flashdecoding_chunk_f32(
            q_ptr, k_chunk, v_chunk,
            current_chunk_len, head_dim, softmax_scale,
            p_o, p_max, p_sum
        );
    }
    
    rust_sve2_flashdecoding_reduce_f32(
        all_partial_o.data(),
        all_partial_max.data(),
        all_partial_sum_exp.data(),
        num_chunks,
        head_dim,
        final_o_out
    );
    
    std::cout << "[FlashDecodingTracker] Executed chunked parallel attention for " 
              << num_chunks << " chunks for request " << request_id << ".\n";
}