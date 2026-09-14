#include "kv_cache.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stdexcept>

PagedKVCache::PagedKVCache(int total_blocks, int block_size, int hidden_dim) 
    : block_size(block_size), hidden_dim(hidden_dim) {
    physical_blocks.resize(total_blocks);
    for (int i = 0; i < total_blocks; ++i) {
        physical_blocks[i].k_cache.resize(block_size * hidden_dim, 0.0f);
        physical_blocks[i].v_cache.resize(block_size * hidden_dim, 0.0f);
        free_blocks.push_back(i);
    }
    std::cout << "Initialized PagedKVCache: " << total_blocks << " blocks of size " << block_size << std::endl;
}

void PagedKVCache::init_request(int request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    logical_to_physical[request_id] = {};
}

void PagedKVCache::append(int request_id, const float* k_vec, const float* v_vec, int h_dim) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto& blocks = logical_to_physical[request_id];
    
    if (blocks.empty() || physical_blocks[blocks.back()].occupied_tokens == block_size) {
        if (free_blocks.empty()) {
            throw std::runtime_error("KV Cache OOM: No free blocks available!");
        }
        int new_block_idx = free_blocks.back();
        free_blocks.pop_back();
        blocks.push_back(new_block_idx);
        physical_blocks[new_block_idx].occupied_tokens = 0;
    }
    
    int current_block_idx = blocks.back();
    auto& block = physical_blocks[current_block_idx];
    int token_offset = block.occupied_tokens;
    
    std::memcpy(block.k_cache.data() + (token_offset * h_dim), k_vec, h_dim * sizeof(float));
    std::memcpy(block.v_cache.data() + (token_offset * h_dim), v_vec, h_dim * sizeof(float));
    block.occupied_tokens++;
}

void PagedKVCache::get_cache(int request_id, std::vector<float>& out_k, std::vector<float>& out_v) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto& blocks = logical_to_physical[request_id];
    int total_tokens = 0;
    for (int idx : blocks) total_tokens += physical_blocks[idx].occupied_tokens;
    
    out_k.resize(total_tokens * hidden_dim);
    out_v.resize(total_tokens * hidden_dim);
    
    int offset = 0;
    for (int idx : blocks) {
        const auto& block = physical_blocks[idx];
        int bytes_to_copy = block.occupied_tokens * hidden_dim * sizeof(float);
        std::memcpy(out_k.data() + offset, block.k_cache.data(), bytes_to_copy);
        std::memcpy(out_v.data() + offset, block.v_cache.data(), bytes_to_copy);
        offset += bytes_to_copy;
    }
}

// Day 5: Batch-aware retrieval
void PagedKVCache::get_batch_cache(const std::vector<int>& request_ids, std::vector<std::vector<float>>& out_k_batch, std::vector<std::vector<float>>& out_v_batch) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    out_k_batch.resize(request_ids.size());
    out_v_batch.resize(request_ids.size());
    
    for (size_t i = 0; i < request_ids.size(); ++i) {
        int request_id = request_ids[i];
        auto it = logical_to_physical.find(request_id);
        if (it == logical_to_physical.end()) continue;
        
        const auto& blocks = it->second;
        int total_tokens = 0;
        for (int idx : blocks) total_tokens += physical_blocks[idx].occupied_tokens;
        
        out_k_batch[i].resize(total_tokens * hidden_dim);
        out_v_batch[i].resize(total_tokens * hidden_dim);
        
        int offset = 0;
        for (int idx : blocks) {
            const auto& block = physical_blocks[idx];
            int bytes_to_copy = block.occupied_tokens * hidden_dim * sizeof(float);
            std::memcpy(out_k_batch[i].data() + offset, block.k_cache.data(), bytes_to_copy);
            std::memcpy(out_v_batch[i].data() + offset, block.v_cache.data(), bytes_to_copy);
            offset += bytes_to_copy;
        }
    }
}

void PagedKVCache::free_request(int request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = logical_to_physical.find(request_id);
    if (it != logical_to_physical.end()) {
        for (int idx : it->second) {
            physical_blocks[idx].occupied_tokens = 0;
            free_blocks.push_back(idx);
        }
        logical_to_physical.erase(it);
    }
}