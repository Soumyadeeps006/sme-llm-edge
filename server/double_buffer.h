#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>

namespace edge_ai {

struct InferenceBuffers {
    std::vector<uint8_t> a_packed;
    std::vector<float> scales;
    std::vector<float> c_output;
    uint32_t m, k, n;
    bool is_ready;
};

class DoubleBuffer {
public:
    DoubleBuffer(uint32_t max_m, uint32_t k, uint32_t n, uint32_t k_packed_usize, uint32_t n_usize);
    ~DoubleBuffer() = default;

    InferenceBuffers* get_write_buffer();
    void commit_and_swap(uint32_t m);
    InferenceBuffers* get_read_buffer();
    void mark_read_complete();

private:
    InferenceBuffers buf_0;
    InferenceBuffers buf_1;
    InferenceBuffers* active_write;
    InferenceBuffers* active_read;
    std::mutex mutex_;
    uint32_t k, n, k_packed_usize, n_usize;
};

} // namespace edge_ai