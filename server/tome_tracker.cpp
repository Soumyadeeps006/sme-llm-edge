#include "tome_tracker.h"
#include <iostream>
#include <algorithm>

extern "C" {
    void rust_sve2_tome_similarity_mask(
        const float* hidden_states_ptr,
        uint64_t* mask_out_ptr,
        uint32_t seq_len,
        uint32_t hidden_dim,
        float similarity_threshold
    );
}

TomeTracker::TomeTracker(int hidden_dim, float sim_thresh, int min_len)
    : hidden_dim(hidden_dim), similarity_threshold(sim_thresh), min_seq_len_for_tome(min_len) {}

const uint64_t* TomeTracker::evaluate_and_generate_mask(const float* hidden_states, int seq_len) {
    if (seq_len < min_seq_len_for_tome) {
        int num_words = (seq_len + 63) / 64;
        merge_mask.assign(num_words, 0xFFFFFFFFFFFFFFFFULL);
        new_to_old_token_map.resize(seq_len);
        for (int i = 0; i < seq_len; ++i) new_to_old_token_map[i] = i;
        return merge_mask.data();
    }

    int num_words = (seq_len + 63) / 64;
    merge_mask.assign(num_words, 0xFFFFFFFFFFFFFFFFULL);

    rust_sve2_tome_similarity_mask(
        hidden_states,
        merge_mask.data(),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(hidden_dim),
        similarity_threshold
    );

    new_to_old_token_map.clear();
    for (int i = 0; i < seq_len; ++i) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        if ((merge_mask[word_idx] >> bit_idx) & 1) {
            new_to_old_token_map.push_back(i);
        }
    }

    return merge_mask.data();
}

int TomeTracker::apply_merge_and_get_freed_count(int seq_len, int block_size) {
    if (seq_len < min_seq_len_for_tome) return 0;
    
    int old_blocks = (seq_len + block_size - 1) / block_size;
    int new_blocks = (get_merged_seq_len() + block_size - 1) / block_size;
    return std::max(0, old_blocks - new_blocks);
}