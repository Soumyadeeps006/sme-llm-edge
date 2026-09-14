#include "scheduler.h"
#include <iostream>
#include <algorithm>

Scheduler::Scheduler(RadixKVCache* cache, int max_batch, int target_ttft) 
    : kv_cache(cache), max_batch_size(max_batch), current_batch_size(1), target_ttft_ms(target_ttft) {}

void Scheduler::add_request(int request_id, const std::string& prompt, int max_tokens, bool stream, SOCKET client_fd) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    ActiveRequest req;
    req.request_id = request_id;
    req.prompt = prompt;
    req.max_tokens = max_tokens;
    req.stream = stream;
    req.client_fd = client_fd;
    req.state = RequestState::PENDING;
    req.current_step = 0;
    
    // Note: RadixKVCache initializes history automatically on the first append()
    
    requests.push_back(std::move(req));
    cv.notify_one();
}

void Scheduler::mark_completed(int request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto& req : requests) {
        if (req.request_id == request_id) {
            req.state = RequestState::COMPLETED;
            if (kv_cache) {
                kv_cache->release_prefix(request_id);
            }
            break;
        }
    }
}

void Scheduler::mark_failed(int request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto& req : requests) {
        if (req.request_id == request_id) {
            req.state = RequestState::FAILED;
            if (kv_cache) {
                kv_cache->release_prefix(request_id);
            }
            break;
        }
    }
}

ActiveRequest* Scheduler::get_request(int request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto& req : requests) {
        if (req.request_id == request_id) return &req;
    }
    return nullptr;
}

std::vector<ActiveRequest*> Scheduler::get_next_batch_adaptive() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    
    // Adaptive logic: scale up if queue is deep, scale down to prioritize latency
    int pending_count = 0;
    for (const auto& req : requests) {
        if (req.state == RequestState::PENDING) pending_count++;
    }
    
    if (pending_count > 4) {
        current_batch_size = std::min(max_batch_size, current_batch_size + 1);
    } else {
        current_batch_size = std::max(1, current_batch_size - 1);
    }

    std::vector<ActiveRequest*> batch;
    for (auto& req : requests) {
        if ((req.state == RequestState::PENDING || req.state == RequestState::RUNNING) && 
            req.current_step < req.max_tokens && batch.size() < static_cast<size_t>(current_batch_size)) {
            req.state = RequestState::RUNNING;
            batch.push_back(&req);
        }
    }
    return batch;
}