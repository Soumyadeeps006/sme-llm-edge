#include "eviction_tracker.h"
#include <iostream>
#include <algorithm>
#include <vector>

extern "C" {
    // =========================================================================
    // Day 38: FFI Boundary Audit
    // Memory Layout & Ownership Rules:
    // - Caller retains ownership of ALL pointers passed to this function.
    // - tokens_ptr: Must point to valid memory of size (num_tokens * hidden_dim) * sizeof(float).
    // - scores_out: Must point to valid memory of size num_tokens * sizeof(float).
    // - num_tokens must be > 0. Rust side MUST NOT panic (custom panic handler configured).
    // =========================================================================
    void rust_sve2_compute_token_importance(
        const float* tokens_ptr, uint32_t num_tokens, uint32_t hidden_dim, float* scores_out);

    // =========================================================================
    // - scores_ptr: Must point to valid memory of size num_tokens * sizeof(float).
    // - mask_out: Must point to valid memory of size ((num_tokens + 63) / 64) * sizeof(uint64_t).
    // - evict_count MUST be <= num_tokens.
    // =========================================================================
    void rust_sve2_generate_eviction_mask(
        const float* scores_ptr, uint32_t num_tokens, uint32_t evict_count, uint64_t* mask_out);

    // =========================================================================
    // - src_k_ptr, src_v_ptr, dst_k_ptr, dst_v_ptr: Valid memory of size (num_tokens * hidden_dim) * sizeof(float).
    // - mask_ptr: Valid memory of size ((num_tokens + 63) / 64) * sizeof(uint64_t).
    // =========================================================================
    void rust_sme_compact_kv_blocks(
        const float* src_k_ptr, const float* src_v_ptr,
        float* dst_k_ptr, float* dst_v_ptr,
        const uint64_t* mask_ptr, uint32_t num_tokens, uint32_t hidden_dim);
}

EvictionTracker::EvictionTracker(PagedKVCache* cache, float threshold, float ratio)
    : paged_cache(cache), eviction_threshold(threshold), evict_ratio(ratio) {}

bool EvictionTracker::check_and_evict(int request_id, int current_seq_len) {
    if (!paged_cache) {
        std::cerr << "[EvictionTracker] Error: paged_cache is null.\n";
        return false;
    }

    // Edge case guard: Do not evict if sequence length is too small.
    const int min_safe_tokens = 16; // Always keep at least a minimum safe threshold (e.g., prompt prefix)
    if (current_seq_len <= min_safe_tokens) {
        return false;
    }

    float utilization = paged_cache->get_utilization();
    if (utilization > eviction_threshold) {
        std::cout << "[EvictionTracker] Cache utilization at " << (utilization * 100.0f) 
                  << "%. Triggering SVE2 importance-based eviction.\n";
        
        int num_tokens_to_evict = static_cast<int>(current_seq_len * evict_ratio);
        
        // Edge case guard: Prevent eviction if evict_count >= num_tokens (always keep safe threshold)
        if (num_tokens_to_evict <= 0 || (current_seq_len - num_tokens_to_evict) < min_safe_tokens) {
            num_tokens_to_evict = current_seq_len - min_safe_tokens;
        }
        
        if (num_tokens_to_evict <= 0) {
            std::cout << "[EvictionTracker] Aborting eviction: not enough tokens to safely evict.\n";
            return false;
        }

        // Defensive guard: Ensure block alignment math does not result in off-by-one errors.
        int block_size = paged_cache->get_block_size();
        if (block_size <= 0) block_size = 1; // Fallback to prevent division by zero
        
        // Ceiling division to determine blocks to free
        int blocks_to_free = (num_tokens_to_evict + block_size - 1) / block_size; 
        
        // Day 38: Defensive guard before calling cache eviction
        if (blocks_to_free <= 0) {
            return false;
        }

        // Delegate to the cache's eviction method (which should also have internal guards)
        return paged_cache->evict_least_important_blocks(request_id, num_tokens_to_evict);
    }
    return false;
}