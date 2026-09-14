#include "speculative_decoder.h"
#include <algorithm>
#include <iostream>

extern "C" {
    void sve2_speculative_verify_f32(
        const float* logits_ptr,
        const int32_t* draft_tokens_ptr,
        const int32_t* draft_lengths_ptr,
        uint32_t vocab_size,
        uint32_t max_draft_length,
        uint32_t batch_size,
        int32_t* accepted_counts_ptr,
        int32_t* replacement_tokens_ptr
    );
}

SpeculativeDecoder::SpeculativeDecoder(int max_draft) : max_draft_length(max_draft) {}

uint64_t SpeculativeDecoder::hash_ngram(int t1, int t2) const {
    return (static_cast<uint64_t>(static_cast<uint32_t>(t1)) << 32) | static_cast<uint32_t>(t2);
}

DraftResult SpeculativeDecoder::generate_draft(const std::vector<int>& recent_tokens) {
    DraftResult result;
    result.draft_length = 0;
    
    if (recent_tokens.size() < 2) {
        return result;
    }
    
    int current_t1 = recent_tokens[recent_tokens.size() - 2];
    int current_t2 = recent_tokens[recent_tokens.size() - 1];
    
    for (int i = 0; i < max_draft_length; ++i) {
        uint64_t key = hash_ngram(current_t1, current_t2);
        auto it = ngram_table.find(key);
        
        if (it != ngram_table.end()) {
            int next_token = it->second;
            result.draft_tokens.push_back(next_token);
            result.draft_length++;
            current_t1 = current_t2;
            current_t2 = next_token;
        } else {
            break;
        }
    }
    
    return result;
}

void SpeculativeDecoder::verify_drafts_sve2(const std::vector<float>& model_logits,
                                            const std::vector<int>& draft_tokens,
                                            const std::vector<int>& draft_lengths,
                                            int vocab_size,
                                            int batch_size,
                                            std::vector<int>& accepted_counts,
                                            std::vector<int>& replacement_tokens) {
    accepted_counts.assign(batch_size, 0);
    replacement_tokens.assign(batch_size, -1);
    
    if (model_logits.empty() || draft_tokens.empty() || draft_lengths.empty()) {
        return;
    }

    sve2_speculative_verify_f32(
        model_logits.data(),
        reinterpret_cast<const int32_t*>(draft_tokens.data()),
        reinterpret_cast<const int32_t*>(draft_lengths.data()),
        static_cast<uint32_t>(vocab_size),
        static_cast<uint32_t>(max_draft_length),
        static_cast<uint32_t>(batch_size),
        reinterpret_cast<int32_t*>(accepted_counts.data()),
        reinterpret_cast<int32_t*>(replacement_tokens.data())
    );
}

void SpeculativeDecoder::update_ngram(int t1, int t2, int next_token) {
    uint64_t key = hash_ngram(t1, t2);
    ngram_table[key] = next_token;
}