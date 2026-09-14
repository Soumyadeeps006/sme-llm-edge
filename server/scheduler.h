#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "radix_kv_cache.h"

#ifdef _WIN32
#include <winsock2.h>
typedef int SOCKET;
#else
#include <sys/socket.h>
typedef int SOCKET;
#endif

enum class RequestState {
    PENDING,    // Waiting to be processed
    RUNNING,    // Currently in a batch
    COMPLETED,  // Finished generation (</s> or max_tokens)
    FAILED      // Error occurred
};

struct ActiveRequest {
    int request_id;
    std::string prompt;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    std::string generated_text;
    int max_tokens;
    bool stream;
    RequestState state = RequestState::PENDING;
    int current_step = 0;
    SOCKET client_fd; // For streaming responses
    
    // Day 7: Speculative decoding metrics
    int speculative_accepted_count = 0;
    int speculative_total_drafts = 0;
};

class Scheduler {
private:
    std::vector<ActiveRequest> requests;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool running = true;
    RadixKVCache* kv_cache;
    
    // Adaptive Micro-Batching parameters
    int max_batch_size;
    int current_batch_size;
    int target_ttft_ms;

public:
    explicit Scheduler(RadixKVCache* cache, int max_batch = 8, int target_ttft = 100);
    
    void add_request(int request_id, const std::string& prompt, int max_tokens, bool stream, SOCKET client_fd);
    void mark_completed(int request_id);
    void mark_failed(int request_id);
    
    // Day 6: Dynamically calculates optimal batch size based on queue depth
    std::vector<ActiveRequest*> get_next_batch_adaptive();
    
    void stop() { 
        std::lock_guard<std::mutex> lock(queue_mutex);
        running = false; 
        cv.notify_all(); 
    }
    bool is_running() const { return running; }
    
    ActiveRequest* get_request(int request_id);
};