#include "paged_kv_cache.h"
#include <algorithm>
#include <iostream>
#include <cmath>

extern "C" {
    void sme_sm_prefetch_kv_blocks(const int8_t** k_ptrs, const int8_t** v_ptrs, uint32_t num_blocks);
}

void quantize_to_int8(const std::vector<float>& src, 
                      std::vector<int8_t>& dst, 
                      std::vector<float>& scales,
                      int num_elements, int hidden_dim) {
    std::vector<float> abs_max(hidden_dim, 0.0f);
    for (int i = 0; i < num_elements; ++i) {
        for (int d = 0; d < hidden_dim; ++d) {
            float val = std::abs(src[i * hidden_dim + d]);
            if (val > abs_max[d]) abs_max[d] = val;
        }
    }
    
    const float eps = 1e-8f;
    for (int d = 0; d < hidden_dim; ++d) {
        scales[d] = abs_max[d] / 127.0f + eps;
    }
    
    for (int i = 0; i < num_elements; ++i) {
        for (int d = 0; d < hidden_dim; ++d) {
            float scaled = src[i * hidden_dim + d] / scales[d];
            int32_t clipped = static_cast<int32_t>(std::round(scaled));
            clipped = std::max(-128, std::min(127, clipped));
            dst[i * hidden_dim + d] = static_cast<int8_t>(clipped);
        }
    }
}

PagedKVCache::PagedKVCache(int num_blocks, int block_size, int hidden_dim) 
    : num_blocks(num_blocks), block_size(block_size), hidden_dim(hidden_dim) {
    kv_cache_blocks.resize(num_blocks);
    block_ref_counts.resize(num_blocks, 0);
    
    for (auto& block : kv_cache_blocks) {
        block.k_int8.resize(block_size * hidden_dim);
        block.v_int8.resize(block_size * hidden_dim);
        block.k_scales.resize(hidden_dim, 1.0f);
        block.v_scales.resize(hidden_dim, 1.0f);
    }
}

int PagedKVCache::append_tokens(int request_id, const std::vector<float>& new_k, const std::vector<float>& new_v) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    auto& table = request_tables[request_id];
    if (table.request_id == 0) {
        table.request_id = request_id;
    }
    
    int tokens_to_add = new_k.size() / hidden_dim;
    if (tokens_to_add == 0) return -1;
    
    int last_physical_block = -1;
    int token_idx = 0;
    
    while (token_idx < tokens_to_add) {
        int offset_in_block = table.logical_length % block_size;
        int current_block_idx = table.logical_length / block_size;
        
        if (current_block_idx >= table.physical_blocks.size()) {
            int free_block_idx = -1;
            for (int b = 0; b < num_blocks; ++b) {
                if (block_ref_counts[b] == 0) {
                    free_block_idx = b;
                    block_ref_counts[b] = 1;
                    break;
                }
            }
            
            if (free_block_idx == -1) {
                std::cerr << "[PagedKVCache] OOM: No free blocks available for request " << request_id << "\n";
                break;
            }
            table.physical_blocks.push_back(free_block_idx);
        }
        
        int phys_idx = table.physical_blocks[current_block_idx];
        int copy_len = std::min(block_size - offset_in_block, tokens_to_add - token_idx);
        
        std::vector<float> k_slice(new_k.begin() + token_idx * hidden_dim, 
                                   new_k.begin() + (token_idx + copy_len) * hidden_dim);
        quantize_to_int8(k_slice, kv_cache_blocks[phys_idx].k_int8, kv_cache_blocks[phys_idx].k_scales, copy_len, hidden_dim);
        
        std::vector<float> v_slice(new_v.begin() + token_idx * hidden_dim, 
                                   new_v.begin() + (token_idx + copy_len) * hidden_dim);
        quantize_to_int8(v_slice, kv_cache_blocks[phys_idx].v_int8, kv_cache_blocks[phys_idx].v_scales, copy_len, hidden_dim);
        
        token_idx += copy_len;
        table.logical_length += copy_len;
        last_physical_block = phys_idx;
    }
    
    return last_physical_block;
}

void PagedKVCache::get_cache_for_batch(const std::vector<int>& request_ids, 
                                       std::vector<std::vector<int>>& batch_block_tables,
                                       std::vector<int>& batch_seq_lens) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    batch_block_tables.clear();
    batch_seq_lens.clear();
    
    for (int req_id : request_ids) {
        auto it = request_tables.find(req_id);
        if (it != request_tables.end()) {
            batch_block_tables.push_back(it->second.physical_blocks);
            batch_seq_lens.push_back(it->second.logical_length);
        } else {
            batch_block_tables.push_back({});
            batch_seq_lens.push_back(0);
        }
    }
}

void PagedKVCache::free_request(int request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = request_tables.find(request_id);
    if (it != request_tables.end()) {
        for (int phys_idx : it->second.physical_blocks) {
            if (phys_idx >= 0 && phys_idx < num_blocks) {
                block_ref_counts[phys_idx]--;
            }
        }
        request_tables.erase(it);
    }
}

void PagedKVCache::share_prefix(int new_request_id, int source_request_id, int shared_length) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto source_it = request_tables.find(source_request_id);
    if (source_it == request_tables.end()) return;
    
    int blocks_to_share = (shared_length + block_size - 1) / block_size;
    
    BlockTable new_table;
    new_table.request_id = new_request_id;
    new_table.logical_length = shared_length;
    
    for (int i = 0; i < blocks_to_share && i < source_it->second.physical_blocks.size(); ++i) {
        int phys_idx = source_it->second.physical_blocks[i];
        new_table.physical_blocks.push_back(phys_idx);
        block_ref_counts[phys_idx]++;
    }
    
    request_tables[new_request_id] = new_table;
}

// Day 29: SME SM async prefetch trigger implementation
void PagedKVCache::prefetch_kv_blocks(const std::vector<int>& physical_block_indices) const {
    if (physical_block_indices.empty()) return;
    
    std::vector<const int8_t*> k_ptrs(physical_block_indices.size());
    std::vector<const int8_t*> v_ptrs(physical_block_indices.size());
    
    for (size_t i = 0; i < physical_block_indices.size(); ++i) {
        int phys_idx = physical_block_indices[i];
        if (phys_idx >= 0 && phys_idx < num_blocks) {
            k_ptrs[i] = kv_cache_blocks[phys_idx].k_int8.data();
            v_ptrs[i] = kv_cache_blocks[phys_idx].v_int8.data();
        } else {
            k_ptrs[i] = nullptr;
            v_ptrs[i] = nullptr;
        }
    }
    
    sme_sm_prefetch_kv_blocks(k_ptrs.data(), v_ptrs.data(), static_cast<uint32_t>(physical_block_indices.size()));
}

// Day 33: Free merged KV blocks back to the pool
void PagedKVCache::free_merged_blocks(int request_id, int num_blocks_to_free) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = request_tables.find(request_id);
    if (it != request_tables.end() && num_blocks_to_free > 0) {
        int freed = 0;
        while (!it->second.physical_blocks.empty() && freed < num_blocks_to_free) {
            int phys_idx = it->second.physical_blocks.back();
            it->second.physical_blocks.pop_back();
            if (phys_idx >= 0 && phys_idx < num_blocks) {
                block_ref_counts[phys_idx]--;
            }
            freed++;
        }
        it->second.logical_length -= num_blocks_to_free * block_size;
        if (it->second.logical_length < 0) it->second.logical_length = 0;
    }
}

// DAY 34: Get current cache utilization
float PagedKVCache::get_utilization() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    int used_blocks = 0;
    for (int count : block_ref_counts) {
        if (count > 0) used_blocks++;
    }
    return static_cast<float>(used_blocks) / num_blocks;
}

// DAY 34: Evict least important blocks and free them back to the pool
bool PagedKVCache::evict_least_important_blocks(int request_id, int num_tokens_to_evict) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = request_tables.find(request_id);
    if (it == request_tables.end() || num_tokens_to_evict <= 0) return false;

    int tokens_freed = 0;
    while (!it->second.physical_blocks.empty() && tokens_freed < num_tokens_to_evict) {
        int phys_idx = it->second.physical_blocks.back();
        it->second.physical_blocks.pop_back();
        if (phys_idx >= 0 && phys_idx < num_blocks) {
            block_ref_counts[phys_idx]--;
        }
        tokens_freed += block_size;
    }
    
    it->second.logical_length -= tokens_freed;
    if (it->second.logical_length < 0) it->second.logical_length = 0;
    
    std::cout << "[PagedKVCache] Evicted " << tokens_freed << " tokens (approx " 
              << (tokens_freed / block_size) << " blocks) for request " << request_id << "\n";
    return true;
}

KVBlockPointers PagedKVCache::get_block_pointers(int physical_block_idx, int num_tokens_in_block) const {
    if (physical_block_idx < 0 || physical_block_idx >= num_blocks) {
        return {nullptr, nullptr, nullptr, nullptr, 0};
    }
    const auto& block = kv_cache_blocks[physical_block_idx];
    return {
        block.k_int8.data(),
        block.v_int8.data(),
        block.k_scales.data(),
        block.v_scales.data(),
        num_tokens_in_block
    };
}