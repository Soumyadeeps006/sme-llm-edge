#pragma once
#include <cstdint>
#include <arm_sve.h>

namespace edge_ai {

// Scalar fallback for non-SVE environments or tail elements
void pack_weights_i4_scalar(const float* a_fp32, const float* scales, 
                            uint8_t* a_packed_out, float* scales_out, 
                            uint32_t m, uint32_t k);

// SVE-accelerated packing (assumes k is a multiple of SVE vector length for simplicity, with scalar tail handling)
void pack_weights_i4_sve(const float* a_fp32, const float* scales, 
                         uint8_t* a_packed_out, float* scales_out, 
                         uint32_t m, uint32_t k);

} // namespace edge_ai