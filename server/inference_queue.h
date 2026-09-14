#pragma once
#include <thread>
#include <vector>
#include <chrono>
#include <cstdint>
#include <atomic>
#include "double_buffer.h"
#include "thread_pool.h"
#include "lock_free_queue.h"

namespace edge_ai {

class InferenceQueue {
public:
    InferenceQueue(ThreadPool& pool, DoubleBuffer& db, float eps, float alpha, float beta);
    ~InferenceQueue();

    void submit_request(const float* a_fp32, const float* b, const float* norm_weight, 
                        uint32_t m, uint32_t k, uint32_t n, const float* scales);

private:
    void pack_weights_i4_zero_alloc(const float* a_fp32, const float* scales, 
                                    uint8_t* a_packed_out, float* scales_out, 
                                    uint32_t m, uint32_t k);

    ThreadPool& thread_pool_;
    DoubleBuffer& double_buffer_;
    std::thread dispatcher_thread_;
    std::atomic<bool> stop_{false};
    float eps_, alpha_, beta_;

    // Micro-batching configuration
    uint32_t max_batch_m_;
    uint64_t batch_window_ns_; // e.g., 100,000 ns = 100 µs
    
    // Pending batch state
    uint32_t current_batch_m_;
    uint32_t batch_k_;
    uint32_t batch_n_;
    const float* batch_b_;
    const float* batch_norm_weight_;
    std::chrono::steady_clock::time_point batch_start_time_;

    // Lock-free task queue (Capacity 1024 = power of 2)
    LockFreeQueue<1024> task_queue_;
};

} // namespace edge_ai