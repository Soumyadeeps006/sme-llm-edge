#pragma once
#include <vector>

// Computes block-wise paged attention for a batch of requests.
// batch_q: [M, hidden_dim]
// k_cache_pool, v_cache_pool: Flat arrays managed by PagedKVCache
// batch_block_tables: [M, num_blocks_per_req]
// batch_seq_lens: [M]
// batch_o: [M, hidden_dim] (Output, accumulated in-place or passed as mutable ref)
void compute_paged_attention(
    const std::vector<float>& batch_q,
    const std::vector<float>& k_cache_pool,
    const std::vector<float>& v_cache_pool,
    const std::vector<std::vector<int>>& batch_block_tables,
    const std::vector<int>& batch_seq_lens,
    std::vector<float>& batch_o,
    int hidden_dim,
    int block_size
);