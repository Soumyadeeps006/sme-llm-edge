#pragma once
#include "paged_kv_cache.h"
#include <vector>
#include <cstdint>

class EvictionTracker {
private:
    PagedKVCache* paged_cache;
    float eviction_threshold; // e.g., 0.90
    float evict_ratio;        // e.g., 0.20

public:
    EvictionTracker(PagedKVCache* cache, float threshold = 0.90f, float ratio = 0.20f);
    
    // Day 38: Hardened to gracefully handle edge cases (zero-token states, exact block boundaries)
    // without panicking or causing use-after-free errors.
    bool check_and_evict(int request_id, int current_seq_len);
};