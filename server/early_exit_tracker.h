#pragma once
#include <vector>
#include <cstdint>

class EarlyExitTracker {
private:
    int max_batch_size;
    int vocab_size;
    int eos_token_id;
    float confidence_threshold;
    std::vector<bool> is_completed;
    std::vector<uint64_t> active_token_mask; // SVE2 predicate mask representation

public:
    EarlyExitTracker(int max_batch, int vocab_sz, int eos_id, float conf_thresh = 0.99f);
    
    void reset_batch(int current_batch_size);
    
    // Evaluates logits and updates the active token mask
    // Returns pointer to the u64 array representing the SVE2 svbool_t predicate
    const uint64_t* evaluate_and_generate_mask(const float* logits, int batch_size, int seq_len);
    
    bool is_request_completed(int request_idx) const;
    void mark_completed(int request_idx);
    const std::vector<bool>& get_completed_status() const { return is_completed; }
};