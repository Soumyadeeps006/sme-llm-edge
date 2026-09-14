#include "double_buffer.h"
#include <algorithm>

namespace edge_ai {

DoubleBuffer::DoubleBuffer(uint32_t max_m, uint32_t k, uint32_t n, uint32_t k_packed_usize, uint32_t n_usize) 
    : k(k), n(n), k_packed_usize(k_packed_usize), n_usize(n_usize) {
    
    // Pre-allocate memory once. std::vector::resize value-initializes to 0.
    buf_0.a_packed.resize(max_m * k_packed_usize);
    buf_0.scales.resize(max_m);
    buf_0.c_output.resize(max_m * n_usize);
    buf_0.is_ready = false;

    buf_1.a_packed.resize(max_m * k_packed_usize);
    buf_1.scales.resize(max_m);
    buf_1.c_output.resize(max_m * n_usize);
    buf_1.is_ready = false;

    active_write = &buf_0;
    active_read = &buf_1;
}

InferenceBuffers* DoubleBuffer::get_write_buffer() {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_write;
}

void DoubleBuffer::commit_and_swap(uint32_t m) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_write->m = m;
    active_write->k = k;
    active_write->n = n;
    active_write->is_ready = true;
    
    // Ping-pong swap
    std::swap(active_write, active_read);
}

InferenceBuffers* DoubleBuffer::get_read_buffer() {
    return active_read;
}

void DoubleBuffer::mark_read_complete() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_read->is_ready = false;
}

} // namespace edge_ai