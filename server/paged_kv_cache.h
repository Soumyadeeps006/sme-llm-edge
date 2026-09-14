#pragma once
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>

struct BlockTable {
    int request_id;
    std::vector<int> physical_blocks;
    int logical_length = 0;
};

struct QuantizedKVBlock {
    std::vector<int8_t> k_int8;
    std::vector<int8_t> v_int8;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
};

struct KVBlockPointers {
    const int8_t* k_int8_ptr;
    const int8_t* v_int8_ptr;
    const float* k_scales_ptr;
    const float* v_scales_ptr;
    int num_tokens_in_block;
};

class PagedKVCache {
private:
    int num_blocks;
    int block_size;
    int hidden_dim;
    
    std::vector<QuantizedKVBlock> kv_cache_blocks;
    std::vector<int> block_ref_counts;
    std::unordered_map<int, BlockTable> request_tables;
    std::mutex cache_mutex;

public:
    explicit PagedKVCache(int num_blocks, int block_size, int hidden_dim);
    
    int append_tokens(int request_id, const std::vector<float>& new_k, const std::vector<float>& new_v);
    void get_cache_for_batch(const std::vector<int>& request_ids, 
                             std::vector<std::vector<int>>& batch_block_tables,
                             std::vector<int>& batch_seq_lens);
    void free_request(int request_id);
    void share_prefix(int new_request_id, int source_request_id, int shared_length);
    
    // DAY 33: Free merged KV blocks
    void free_merged_blocks(int request_id, int num_blocks_to_free);

    // DAY 34: Eviction and Utilization tracking
    float get_utilization() const;
    bool evict_least_important_blocks(int request_id, int num_tokens_to_evict);

    void prefetch_kv_blocks(const std::vector<int>& physical_block_indices) const;

    const std::vector<QuantizedKVBlock>& get_kv_blocks() const { return kv_cache_blocks; }
    int get_block_size() const { return block_size; }
    int get_hidden_dim() const { return hidden_dim; }

    KVBlockPointers get_block_pointers(int physical_block_idx, int num_tokens_in_block) const;
};