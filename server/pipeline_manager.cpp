#include "pipeline_manager.h"
#include <iostream>

// External C function for SVE2 prefetch (from Day 29/31 Rust kernel)
extern "C" void sme_sm_prefetch_kv_blocks(
    const int8_t** k_ptrs,
    const int8_t** v_ptrs,
    uint32_t num_blocks
);

PipelineManager::PipelineManager(const PipelineConfig& config) : config_(config) {}

void PipelineManager::set_state(PipelineState state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_ = state;
}

PipelineState PipelineManager::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

void PipelineManager::prepare_next_step_prefetch(
    const std::vector<const float*>& k_ptrs,
    const std::vector<const float*>& v_ptrs,
    const std::vector<const float*>& expert_weights_ptrs
) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    next_k_ptrs_ = k_ptrs;
    next_v_ptrs_ = v_ptrs;
    next_expert_weights_ptrs_ = expert_weights_ptrs;
}

void PipelineManager::trigger_gqa_kv_prefetch() {
    if (next_k_ptrs_.empty() || next_v_ptrs_.empty()) return;
    
    // Cast float* to int8_t* to match the Rust kernel interface
    const int8_t** k_ptrs_i8 = reinterpret_cast<const int8_t**>(next_k_ptrs_.data());
    const int8_t** v_ptrs_i8 = reinterpret_cast<const int8_t**>(next_v_ptrs_.data());
    
    sme_sm_prefetch_kv_blocks(
        k_ptrs_i8,
        v_ptrs_i8,
        static_cast<uint32_t>(next_k_ptrs_.size())
    );
}

void PipelineManager::trigger_moe_nt_prefetch() {
    // The non-temporal load is handled directly inside the Rust kernel 
    // `rust_sme_moe_nt_streaming_f32` via svld1nt_f32. This C++ function 
    // serves as a synchronization/log point in the orchestration layer.
    if (!next_expert_weights_ptrs_.empty()) {
        // Hardware prefetcher is implicitly engaged by the NT loads in the kernel
    }
}