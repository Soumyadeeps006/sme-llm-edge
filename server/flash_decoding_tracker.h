#pragma once
#include "paged_kv_cache.h"
#include <vector>

class FlashDecodingTracker {
private:
    PagedKVCache* paged_cache;
    int chunk_size_threshold; // e.g., 4096
    int chunk_size;           // e.g., 512

public:
    FlashDecodingTracker(PagedKVCache* cache, int threshold = 4096, int chunk = 512);
    bool should_use_flash_decoding(int seq_len) const;
    void execute_flash_decoding(
        int request_id,
        const float* q_ptr,
        uint32_t head_dim,
        float softmax_scale,
        float* final_o_out
    );
};