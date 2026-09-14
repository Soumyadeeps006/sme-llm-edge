#pragma once
#include <vector>
#include <cstdint>

class TomeTracker {
private:
    int hidden_dim;
    float similarity_threshold;
    int min_seq_len_for_tome;
    std::vector<uint64_t> merge_mask;
    std::vector<int> new_to_old_token_map;

public:
    TomeTracker(int hidden_dim, float sim_thresh = 0.95f, int min_len = 1024);
    
    const uint64_t* evaluate_and_generate_mask(const float* hidden_states, int seq_len);
    int apply_merge_and_get_freed_count(int seq_len, int block_size);
    
    const std::vector<int>& get_new_to_old_map() const { return new_to_old_token_map; }
    int get_merged_seq_len() const { return static_cast<int>(new_to_old_token_map.size()); }
};