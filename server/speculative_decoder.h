#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

struct DraftResult {
    std::vector<int> draft_tokens;
    int draft_length;
};

class SpeculativeDecoder {
private:
    int max_draft_length;
    std::unordered_map<uint64_t, int> ngram_table; 
    uint64_t hash_ngram(int t1, int t2) const;

public:
    explicit SpeculativeDecoder(int max_draft = 3);
    
    DraftResult generate_draft(const std::vector<int>& recent_tokens);
    
    // Day 29: SVE2 accelerated verification
    void verify_drafts_sve2(const std::vector<float>& model_logits,
                            const std::vector<int>& draft_tokens,
                            const std::vector<int>& draft_lengths,
                            int vocab_size,
                            int batch_size,
                            std::vector<int>& accepted_counts,
                            std::vector<int>& replacement_tokens);
                     
    void update_ngram(int t1, int t2, int next_token);
    
    int get_max_draft_length() const { return max_draft_length; }
};