#pragma once
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

struct RadixNode {
    int token_id;
    int source_request_id = -1; // The request that originally populated this path
    int ref_count = 0;
    std::unordered_map<int, std::unique_ptr<RadixNode>> children;
    
    explicit RadixNode(int token) : token_id(token) {}
};

class RadixKVCache {
private:
    std::unique_ptr<RadixNode> root;
    std::mutex cache_mutex;
    
    // Track full token sequence per request for accurate releasing
    std::unordered_map<int, std::vector<int>> request_token_history;

    void free_subtree(RadixNode* node);
    void garbage_collect();

public:
    RadixKVCache();
    ~RadixKVCache();

    // Returns {matched_token_count, source_request_id}
    std::pair<int, int> match_prefix(const std::vector<int>& tokens);
    
    // Registers a request's tokens into the trie after prefill is complete
    void insert_prefix(int request_id, const std::vector<int>& tokens);
    
    // Decrements ref counts along the path; triggers GC if ref_count == 0
    void release_prefix(int request_id);
};