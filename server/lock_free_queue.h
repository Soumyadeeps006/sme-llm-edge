#pragma once
#include <atomic>
#include <vector>
#include <cstdint>

namespace edge_ai {

struct InferenceTask {
    uint32_t batch_m;
    uint32_t k;
    uint32_t n;
    const float* b;
    const float* norm_weight;
    bool is_valid; // Acts as a simple validity flag for the slot
};

template <size_t Capacity>
class LockFreeQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr size_t MASK = Capacity - 1;

    // alignas(64) prevents false sharing between producer (tail) and consumer (head)
    struct alignas(64) AtomicIndex {
        std::atomic<size_t> index{0};
    };

    AtomicIndex head_; // Consumer reads here
    AtomicIndex tail_; // Producer writes here
    std::vector<InferenceTask> buffer_;

public:
    LockFreeQueue() : buffer_(Capacity) {
        for (auto& task : buffer_) {
            task.is_valid = false;
        }
    }

    // Returns true if successfully pushed, false if queue is full
    bool push(const InferenceTask& task) {
        size_t current_tail = tail_.index.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) & MASK;
        
        // Check if queue is full
        if (next_tail == (head_.index.load(std::memory_order_acquire) & MASK)) {
            return false; 
        }

        buffer_[current_tail] = task;
        buffer_[current_tail].is_valid = true;
        
        // Release semantics ensure the task write is visible before the tail index updates
        tail_.index.store(next_tail, std::memory_order_release);
        return true;
    }

    // Returns true if successfully popped, false if queue is empty
    bool pop(InferenceTask& task) {
        size_t current_head = head_.index.load(std::memory_order_relaxed);
        
        // Check if queue is empty
        if (current_head == (tail_.index.load(std::memory_order_acquire) & MASK)) {
            return false;
        }

        task = buffer_[current_head];
        buffer_[current_head].is_valid = false; // Reset for safety
        
        // Release semantics ensure the read is complete before head advances
        head_.index.store((current_head + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    bool is_empty() const {
        return (head_.index.load(std::memory_order_acquire) & MASK) == 
               (tail_.index.load(std::memory_order_acquire) & MASK);
    }
};

} // namespace edge_ai