#include "chunked_prefill.h"
#include <iostream>

std::vector<int> ChunkedPrefillState::get_next_chunk() {
    if (is_prefill_complete || prompt_tokens.empty()) {
        return {};
    }
    
    int remaining = prompt_tokens.size() - prefill_offset;
    int chunk_len = std::min(remaining, prefill_chunk_size);
    
    std::vector<int> chunk(
        prompt_tokens.begin() + prefill_offset, 
        prompt_tokens.begin() + prefill_offset + chunk_len
    );
    
    prefill_offset += chunk_len;
    
    if (prefill_offset >= prompt_tokens.size()) {
        is_prefill_complete = true;
    }
    
    return chunk;
}

int ChunkedPrefillState::get_remaining_tokens() const {
    return prompt_tokens.empty() ? 0 : (prompt_tokens.size() - prefill_offset);
}

void ChunkedPrefillState::reset() {
    prefill_offset = 0;
    is_prefill_complete = false;
}