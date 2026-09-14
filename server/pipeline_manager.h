#pragma once
#include <vector>
#include <mutex>
#include <cstdint>

// DAY 31: Pipeline State Machine for Overlap Orchestration
enum class PipelineState {
    IDLE,
    FETCHING_WEIGHTS,
    COMPUTING_MOE,
    OVERLAPPING_GQA
};

struct PipelineConfig {
    uint32_t num_experts = 8;
    uint32_t top_k = 2;
    uint32_t hidden_dim = 4096;
    uint32_t expert_intermediate_dim = 14336;
    uint32_t num_kv_heads = 8;
    uint32_t head_dim = 128;
};

class PipelineManager {
private:
    PipelineConfig config_;
    PipelineState current_state_ = PipelineState::IDLE;
    std::mutex state_mutex_;
    
    // Prefetch queues for next-step KV cache and MoE weights
    std::vector<const float*> next_k_ptrs_;
    std::vector<const float*> next_v_ptrs_;
    std::vector<const float*> next_expert_weights_ptrs_;

public:
    explicit PipelineManager(const PipelineConfig& config);
    
    void set_state(PipelineState state);
    PipelineState get_state() const;
    
    // Prepare next-step prefetches (called at the end of current step)
    void prepare_next_step_prefetch(
        const std::vector<const float*>& k_ptrs,
        const std::vector<const float*>& v_ptrs,
        const std::vector<const float*>& expert_weights_ptrs
    );
    
    // Trigger hardware prefetches for GQA KV cache
    void trigger_gqa_kv_prefetch();
    
    // Trigger non-temporal load prefetch for MoE weights
    void trigger_moe_nt_prefetch();
    
    const PipelineConfig& get_config() const { return config_; }
};