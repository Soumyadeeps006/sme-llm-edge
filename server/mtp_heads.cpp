#include "mtp_heads.h"
#include <iostream>
#include <cstring>
#include <vector>

// ============================================================================
// Day 38: FFI Boundary Audit
// Memory Layout & Ownership Rules:
// - Caller (C++) retains ownership of ALL pointers passed to these Rust functions.
// - Rust side MUST NOT panic across the FFI boundary. A custom #[panic_handler] 
//   is configured in Rust to loop {} to prevent undefined behavior (UB).
// - All pointers must be validated for null and correct slice bounds before calling.
// ============================================================================
extern "C" {
    // hidden_state_ptr: Must point to valid, contiguous memory of size hidden_dim * sizeof(float).
    // mtp_weights_ptr: Must point to valid memory of size (k_heads * hidden_dim * vocab_size) * sizeof(float).
    // mtp_logits_out: Must point to valid memory of size (k_heads * vocab_size) * sizeof(float).
    void rust_sme_mtp_predict_f32(
        const float* hidden_state_ptr,
        const float* mtp_weights_ptr,
        float* mtp_logits_out,
        uint32_t hidden_dim,
        uint32_t vocab_size,
        uint32_t k_heads
    );

    // mtp_logits_ptr: Must point to valid memory of size (k_heads * vocab_size) * sizeof(float).
    // draft_tokens_out: Must point to valid memory of size k_heads * sizeof(int32_t).
    void rust_sme_mtp_sample_f32(
        const float* mtp_logits_ptr,
        int32_t* draft_tokens_out,
        uint32_t vocab_size,
        uint32_t k_heads
    );
}

MTPHeads::MTPHeads(uint32_t k_heads, uint32_t hidden_dim, uint32_t vocab_size)
    : k_heads(k_heads), hidden_dim(hidden_dim), vocab_size(vocab_size) {
    mtp_weights.resize(k_heads * hidden_dim * vocab_size, 0.0f);
    mtp_logits.resize(k_heads * vocab_size, 0.0f);
    draft_tokens.resize(k_heads, -1);
}

void MTPHeads::load_weights(const float* weights_data) {
    // Day 38: FFI Guard
    if (!weights_data) {
        std::cerr << "[MTPHeads] Error: weights_data is null.\n";
        return;
    }
    std::memcpy(mtp_weights.data(), weights_data, mtp_weights.size() * sizeof(float));
}

std::vector<int32_t> MTPHeads::generate_drafts(const float* hidden_state) {
    // Day 38: FFI Guard - Ensure pointers and dimensions are valid before crossing the boundary
    if (!hidden_state || hidden_dim == 0 || vocab_size == 0 || k_heads == 0) {
        std::cerr << "[MTPHeads] Error: Invalid hidden_state pointer or zero dimensions.\n";
        return std::vector<int32_t>(k_heads, -1);
    }

    // Step 1: SVE2 Vectorized MTP Head GEMM
    rust_sme_mtp_predict_f32(
        hidden_state,
        mtp_weights.data(),
        mtp_logits.data(),
        hidden_dim,
        vocab_size,
        k_heads
    );

    // Step 2: SME ZA Fused MTP Logits Sampling
    rust_sme_mtp_sample_f32(
        mtp_logits.data(),
        draft_tokens.data(),
        vocab_size,
        k_heads
    );

    return std::vector<int32_t>(draft_tokens.begin(), draft_tokens.end());
}