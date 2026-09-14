#include "inference_queue.h"
#include <chrono>
#include <thread>
#include <iostream>

namespace edge_ai {

InferenceQueue::InferenceQueue(ThreadPool& pool, DoubleBuffer& db, float eps, float alpha, float beta)
    : thread_pool_(pool),
      double_buffer_(db),
      stop_(false),
      eps_(eps),
      alpha_(alpha),
      beta_(beta),
      max_batch_m_(16), // Tunable: Max micro-batch size
      batch_window_ns_(100000), // Tunable: 100 µs batching window
      current_batch_m_(0),
      batch_k_(0),
      batch_n_(0),
      batch_b_(nullptr),
      batch_norm_weight_(nullptr),
      batch_start_time_(std::chrono::steady_clock::now()) {
    
    dispatcher_thread_ = std::thread([this]() {
        while (!stop_.load(std::memory_order_acquire)) {
            InferenceTask task;
            if (task_queue_.pop(task)) {
                // Execute the batched compute task
                double_buffer_.commit_and_swap(task.batch_m);
                auto* read_buf = double_buffer_.get_read_buffer();
                
                thread_pool_.dispatch_transformer_chunk(
                    read_buf->a_packed.data(), task.b, read_buf->scales.data(),
                    task.norm_weight, read_buf->c_output.data(),
                    0, task.batch_m, task.k, task.n, eps_, alpha_, beta_,
                    (task.k + 1) / 2, task.n
                );
                double_buffer_.mark_read_complete();
            } else {
                // Low-latency yield instead of heavy OS condition_variable wait
                std::this_thread::yield();
            }
        }
    });
}

InferenceQueue::~InferenceQueue() {
    stop_.store(true, std::memory_order_release);
    if (dispatcher_thread_.joinable()) {
        dispatcher_thread_.join();
    }
}

void InferenceQueue::submit_request(const float* a_fp32, const float* b, const float* norm_weight, 
                                    uint32_t m, uint32_t k, uint32_t n, const float* scales) {
    if (current_batch_m_ == 0) {
        batch_k_ = k;
        batch_n_ = n;
        batch_b_ = b;
        batch_norm_weight_ = norm_weight;
        batch_start_time_ = std::chrono::steady_clock::now();
    }

    // NOTE: Day 23 zero-allocation SVE packing into double_buffer_.get_write_buffer() 
    // at offset current_batch_m_ happens here. (Omitted for brevity, assumes handled).

    current_batch_m_ += m;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - batch_start_time_).count();

    // If batch is full or timeout reached, push to lock-free queue
    if (current_batch_m_ >= max_batch_m_ || elapsed >= batch_window_ns_) {
        InferenceTask task{current_batch_m_, batch_k_, batch_n_, batch_b_, batch_norm_weight_, true};
        
        // Spin briefly if queue is full (backpressure mechanism)
        while (!task_queue_.push(task)) {
            std::this_thread::yield();
        }
        
        // Reset batch state for the next window
        current_batch_m_ = 0;
        batch_start_time_ = std::chrono::steady_clock::now();
    }
}

void InferenceQueue::pack_weights_i4_zero_alloc(const float* a_fp32, const float* scales, 
                                                uint8_t* a_packed_out, float* scales_out, 
                                                uint32_t m, uint32_t k) {
    // Placeholder: Day 23 SVE packing logic resides here
}

} // namespace edge_ai