#include "moe_router.h"
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>

// External SVE2/SME MoE kernel declaration
extern "C" void rust_sme_moe_sparse_gemm_f32(
    const float* input_ptr,
    const float* router_weights_ptr,
    const float* expert_weights_ptr,
    float* output_ptr,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t expert_intermediate_dim
);

MoERouter::MoERouter(const MoERouterConfig& config)
    : config_(config),
      expert_load_counts_(config.num_experts, 0) {
    
    router_weights_.resize(config.hidden_dim * config.num_experts, 0.0f);
    expert_weights_.resize(
        config.num_experts * config.hidden_dim * config.expert_intermediate_dim, 
        0.0f
    );
    
    initialize_weights();
}

void MoERouter::initialize_weights() {
    std::mt19937 rng(42); // Deterministic seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    // Initialize router weights
    for (auto& w : router_weights_) {
        w = dist(rng);
    }
    
    // Initialize expert weights
    for (auto& w : expert_weights_) {
        w = dist(rng);
    }
    
    std::cout << "[MoERouter] Initialized " << config_.num_experts 
              << " experts, Top-" << config_.top_k 
              << ", hidden=" << config_.hidden_dim
              << ", intermediate=" << config_.expert_intermediate_dim << "\n";
}

void MoERouter::route_and_compute(
    const float* input_ptr,
    float* output_ptr,
    uint32_t num_tokens
) {
    if (!config_.use_sparse_moe) {
        std::cerr << "[MoERouter] ERROR: Dense mode should use SwiGLU kernel, not MoE router!\n";
        return;
    }
    
    // Call the fused SVE2/SME sparse routing + GEMM kernel
    rust_sme_moe_sparse_gemm_f32(
        input_ptr,
        router_weights_.data(),
        expert_weights_.data(),
        output_ptr,
        num_tokens,
        config_.hidden_dim,
        config_.num_experts,
        config_.top_k,
        config_.expert_intermediate_dim
    );
}

void MoERouter::reset_load_counts() {
    std::fill(expert_load_counts_.begin(), expert_load_counts_.end(), 0);
}