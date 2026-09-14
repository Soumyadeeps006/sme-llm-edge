#pragma once
#include <vector>
#include <algorithm>

struct ChunkedPrefillState {
    std::vector<int> prompt_tokens;
    int prefill_offset = 0;
    bool is_prefill_complete = false;
    int prefill_chunk_size;

    ChunkedPrefillState() : prefill_chunk_size(64) {}
    
    explicit ChunkedPrefillState(const std::vector<int>& tokens, int chunk_size = 64) 
        : prompt_tokens(tokens), prefill_chunk_size(chunk_size) {}

    // Returns the next chunk of tokens to process. Returns empty vector if complete.
    std::vector<int> get_next_chunk();
    
    // Returns the number of tokens left to prefill
    int get_remaining_tokens() const;
    
    // Resets the state for potential request reuse or testing
    void reset();
};