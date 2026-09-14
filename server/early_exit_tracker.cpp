#include "early_exit_tracker.h"
#include <iostream>

// External Rust kernel declaration
extern "C" {
    void rust_sme_generate_early_exit_mask(
        const float* logits_ptr,
        uint64_t* mask_out_ptr,
        uint32_t batch_size,
        uint32_t vocab_size,
        uint32_t eos_token_id,
        float confidence_threshold
    );
}

EarlyExitTracker::EarlyExitTracker(int max_batch, int vocab_sz, int eos_id, float conf_thresh)
    : max_batch_size(max_batch), vocab_size(vocab_sz), eos_token_id(eos_id), 
      confidence_threshold(conf_thresh) {
    is_completed.resize(max_batch_size, false);
    active_token_mask.resize((max_batch_size + 63) / 64, 0xFFFFFFFFFFFFFFFFULL);
}

void EarlyExitTracker::reset_batch(int current_batch_size) {
    for (int i = 0; i < current_batch_size; ++i) {
        is_completed[i] = false;
    }
    int num_words = (current_batch_size + 63) / 64;
    for (int i = 0; i < num_words; ++i) {
        active_token_mask[i] = 0xFFFFFFFFFFFFFFFFULL;
    }
}

const uint64_t* EarlyExitTracker::evaluate_and_generate_mask(const float* logits, int batch_size, int seq_len) {
    reset_batch(batch_size);
    
    rust_sme_generate_early_exit_mask(
        logits,
        active_token_mask.data(),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(vocab_size),
        static_cast<uint32_t>(eos_token_id),
        confidence_threshold
    );
    
    // Update is_completed based on the generated mask
    for (int i = 0; i < batch_size; ++i) {
        int word_idx = i / 64;
        int bit_idx = i % 64;
        bool is_active = (active_token_mask[word_idx] >> bit_idx) & 1;
        is_completed[i] = !is_active;
    }
    
    return active_token_mask.data();
}

bool EarlyExitTracker::is_request_completed(int request_idx) const {
    if (request_idx < 0 || request_idx >= max_batch_size) return false;
    return is_completed[request_idx];
}

void EarlyExitTracker::mark_completed(int request_idx) {
    if (request_idx < 0 || request_idx >= max_batch_size) return;
    is_completed[request_idx] = true;
    int word_idx = request_idx / 64;
    int bit_idx = request_idx % 64;
    active_token_mask[word_idx] &= ~(1ULL << bit_idx);
}