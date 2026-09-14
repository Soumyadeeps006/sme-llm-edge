#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <mutex>

struct KVBlock {
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int occupied_tokens = 0;
};

class PagedKVCache {
private:
    int block_size;
    int hidden_dim;
    std::vector<KVBlock> physical_blocks;
    std::vector<int> free_blocks;
    std::unordered_map<int, std::vector<int>> logical_to_physical;
    std::mutex cache_mutex; // Protects concurrent access

public:
    PagedKVCache(int total_blocks, int block_size, int hidden_dim);
    void init_request(int request_id);
    void append(int request_id, const float* k_vec, const float* v_vec, int h_dim);
    void get_cache(int request_id, std::vector<float>& out_k, std::vector<float>& out_v);
    
    // Day 5: Batch-aware retrieval for continuous batching
    void get_batch_cache(const std::vector<int>& request_ids, std::vector<std::vector<float>>& out_k_batch, std::vector<std::vector<float>>& out_v_batch);
    
    void free_request(int request_id);
};