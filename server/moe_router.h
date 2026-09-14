#pragma once
#include <vector>
#include <cstdint>

// Day 30: MoE Router Configuration and State Management
struct MoERouterConfig {
    uint32_t num_experts = 8;
    uint32_t top_k = 2;
    uint32_t hidden_dim = 4096;
    uint32_t expert_intermediate_dim = 14336;
    bool use_sparse_moe = true; // Toggle between Dense SwiGLU and Sparse MoE
};

class MoERouter {
private:
    MoERouterConfig config_;
    std::vector<float> router_weights_;      // [hidden_dim, num_experts]
    std::vector<float> expert_weights_;       // [num_experts, hidden_dim, expert_intermediate_dim]
    std::vector<int32_t> expert_load_counts_; // For load balancing tracking

public:
    explicit MoERouter(const MoERouterConfig& config);
    
    // Initialize router and expert weights (random or loaded from checkpoint)
    void initialize_weights();
    
    // Get configuration
    const MoERouterConfig& get_config() const { return config_; }
    
    // Get raw pointers for kernel calls
    const float* get_router_weights_ptr() const { return router_weights_.data(); }
    const float* get_expert_weights_ptr() const { return expert_weights_.data(); }
    
    // Execute sparse MoE routing + GEMM via SVE2/SME kernel
    void route_and_compute(
        const float* input_ptr,
        float* output_ptr,
        uint32_t num_tokens
    );
    
    // Load balancing statistics
    void reset_load_counts();
    const std::vector<int32_t>& get_load_counts() const { return expert_load_counts_; }
};