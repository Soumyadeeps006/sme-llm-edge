#pragma once
#include <vector>
#include <cstdint>

extern "C" {
    void rust_sme_mtp_predict_f32(
        const float* hidden_state_ptr,
        const float* weights_ptr,
        float* logits_out_ptr,
        uint32_t hidden_dim,
        uint32_t vocab_size,
        uint32_t k_heads
    );

    void rust_sme_mtp_sample_f32(
        const float* logits_ptr,
        int32_t* draft_tokens_out,
        uint32_t vocab_size,
        uint32_t k_heads
    );
}

class MTPHeads {
private:
    uint32_t k_heads;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    std::vector<float> mtp_weights; // Flattened: K * hidden_dim * vocab_size
    std::vector<float> mtp_logits;  // Flattened: K * vocab_size
    std::vector<int32_t> draft_tokens; // Size: K

public:
    MTPHeads(uint32_t k_heads, uint32_t hidden_dim, uint32_t vocab_size);
    
    void load_weights(const float* weights_data);
    
    // Returns a vector of K draft tokens
    std::vector<int32_t> generate_drafts(const float* hidden_state);
    
    uint32_t get_k_heads() const { return k_heads; }
};