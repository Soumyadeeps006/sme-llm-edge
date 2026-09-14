#include "radix_kv_cache.h"
#include <iostream>
#include <algorithm>

RadixKVCache::RadixKVCache() {
    root = std::make_unique<RadixNode>(-1); // Dummy root
    root->ref_count = 1; // Root is always alive
}

RadixKVCache::~RadixKVCache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    free_subtree(root.get());
}

void RadixKVCache::free_subtree(RadixNode* node) {
    if (!node) return;
    for (auto& child : node->children) {
        free_subtree(child.second.get());
    }
    node->children.clear();
}

void RadixKVCache::garbage_collect() {
    std::function<void(RadixNode*)> cleanup = [&](RadixNode* node) {
        for (auto it = node->children.begin(); it != node->children.end(); ) {
            RadixNode* child = it->second.get();
            if (child->ref_count == 0) {
                free_subtree(child);
                it = node->children.erase(it);
            } else {
                cleanup(child);
                ++it;
            }
        }
    };
    cleanup(root.get());
}

std::pair<int, int> RadixKVCache::match_prefix(const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    RadixNode* current = root.get();
    int matched_count = 0;
    int source_request_id = -1;
    
    for (int token : tokens) {
        auto it = current->children.find(token);
        if (it != current->children.end()) {
            current = it->second.get();
            matched_count++;
            if (current->source_request_id != -1) {
                source_request_id = current->source_request_id;
            }
        } else {
            break;
        }
    }
    return {matched_count, source_request_id};
}

void RadixKVCache::insert_prefix(int request_id, const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    request_token_history[request_id] = tokens;
    
    RadixNode* current = root.get();
    for (int token : tokens) {
        auto it = current->children.find(token);
        if (it == current->children.end()) {
            auto new_node = std::make_unique<RadixNode>(token);
            new_node->source_request_id = request_id;
            new_node->ref_count = 1;
            current->children[token] = std::move(new_node);
            current = current->children[token].get();
        } else {
            it->second->ref_count++;
            current = it->second.get();
        }
    }
}

void RadixKVCache::release_prefix(int request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = request_token_history.find(request_id);
    if (it == request_token_history.end()) return;
    
    const auto& history = it->second;
    RadixNode* current = root.get();
    
    for (int token : history) {
        auto child_it = current->children.find(token);
        if (child_it == current->children.end()) break;
        
        current = child_it->second.get();
        current->ref_count--;
    }
    
    request_token_history.erase(it);
    garbage_collect();
}