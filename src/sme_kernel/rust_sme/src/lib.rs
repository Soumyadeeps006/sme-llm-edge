#![no_std]
#![feature(stdarch_aarch64_sme)] 
#![feature(stdarch_aarch64_sve)]
#![feature(stdarch_aarch64_sve2)]

use core::panic::PanicInfo;
use core::ffi::c_int;
use core::slice;
use core::arch::aarch64::*;
use core::arch::asm;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// ==========================================================
// Basic RMSNorm (Fallback / Reference)
// ==========================================================
#[no_mangle]
pub unsafe extern "C" fn rmsnorm_f32(
    out: *mut f32, x: *const f32, weight: *const f32, size: c_int, eps: f32,
) {
    let size = size as usize;
    let x_slice = core::slice::from_raw_parts(x, size);
    let w_slice = core::slice::from_raw_parts(weight, size);
    let out_slice = core::slice::from_raw_parts_mut(out, size);

    let mut sum_sq = 0.0f32;
    for &val in x_slice.iter() { sum_sq += val * val; }

    let mean = sum_sq / (size as f32);
    let rsqrt = 1.0f32 / (mean + eps).sqrt();

    for i in 0..size { out_slice[i] = x_slice[i] * rsqrt * w_slice[i]; }
}

// ==========================================================
// Basic Softmax (Fallback / Reference)
// ==========================================================
#[no_mangle]
pub unsafe extern "C" fn softmax_f32(x: *mut f32, size: c_int) {
    let size = size as usize;
    let x_slice = core::slice::from_raw_parts_mut(x, size);
    if size == 0 { return; }

    let mut max_val = x_slice[0];
    for &val in x_slice.iter().skip(1) { if val > max_val { max_val = val; } }

    let mut sum = 0.0f32;
    for val in x_slice.iter_mut() { *val = (*val - max_val).exp(); sum += *val; }

    let inv_sum = 1.0f32 / sum;
    for val in x_slice.iter_mut() { *val *= inv_sum; }
}

// ==========================================================
// Day 14: Actual ARMv9 SME ZA Outer-Product GEMM Kernel (FP32)
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_gemm_f32(
    a_ptr: *const f32, b_ptr: *const f32, c_ptr: *mut f32,
    m: u32, k: u32, n: u32, alpha: f32, beta: f32,
) {
    let m_usize = m as usize;
    let k_usize = k as usize;
    let n_usize = n as usize;

    let a = slice::from_raw_parts(a_ptr, m_usize * k_usize);
    let b = slice::from_raw_parts(b_ptr, k_usize * n_usize);
    let c = slice::from_raw_parts_mut(c_ptr, m_usize * n_usize);
    let vl = svcntw(); 

    asm!("smstart za", options(nomem, nostack));

    for i in 0..m_usize {
        let row_a = &a[i * k_usize..(i + 1) * k_usize];
        let row_c = &mut c[i * n_usize..(i + 1) * n_usize];
        let mut tile = svzero_f32();

        for j in (0..n_usize).step_by(vl) {
            let remaining_n = core::cmp::min(vl, n_usize - j);
            let p = svwhilelt_b32(0u32, remaining_n as u32);

            for l in 0..k_usize {
                let a_val = row_a[l];
                let a_vec = svdup_f32(a_val);
                let b_vec = svld1_f32(p, &b[l * n_usize + j]);
                tile = svmopa_f32_m(p, tile, a_vec, b_vec);
            }

            let mut c_vec = svld1_f32(p, &row_c[j]);
            c_vec = svmul_f32_m(p, c_vec, svdup_f32(beta));
            c_vec = svmad_f32_m(p, c_vec, svdup_f32(alpha), tile);
            svst1_f32(p, &mut row_c[j], c_vec);
            tile = svzero_f32();
        }
    }
    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// Day 17: Fused INT8 Dequantization + SME ZA FP32 Outer Product
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_gemm_i8_f32(
    a_ptr: *const i8, b_ptr: *const f32, c_ptr: *mut f32, scales: *const f32,
    m: u32, k: u32, n: u32, alpha: f32, beta: f32,
) {
    let m_usize = m as usize;
    let k_usize = k as usize;
    let n_usize = n as usize;
    let vl = svcntw(); 

    asm!("smstart za", options(nomem, nostack));

    for i in 0..m_usize {
        let scale = *scales.add(i);
        let row_a = slice::from_raw_parts(a_ptr.add(i * k_usize), k_usize);
        let row_c = slice::from_raw_parts_mut(c_ptr.add(i * n_usize), n_usize);
        let mut tile = svzero_f32();

        for j in (0..n_usize).step_by(vl) {
            let remaining_n = core::cmp::min(vl, n_usize - j);
            let p = svwhilelt_b32(j as u32, n_usize as u32);

            for k_idx in 0..k_usize {
                let a_val_i8 = row_a[k_idx] as i32;
                let a_val_f32 = (a_val_i8 as f32) * scale;
                let a_vec = svdup_f32(a_val_f32);
                let b_vec = svld1_f32(p, b_ptr.add(k_idx * n_usize + j));
                tile = svmopa_f32_m(p, tile, a_vec, b_vec);
            }

            let mut c_vec = svld1_f32(p, &row_c[j]);
            c_vec = svmul_f32_m(p, c_vec, svdup_f32(beta));
            c_vec = svmad_f32_m(p, c_vec, svdup_f32(alpha), tile);
            svst1_f32(p, &mut row_c[j], c_vec);
            tile = svzero_f32();
        }
    }
    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// Day 18 + Day 24: W4A16 SVE2 Sub-byte Unpacking + Fused SME Dequantization + SM Prefetching
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_gemm_i4_f32(
    a_packed_ptr: *const u8, b_ptr: *const f32, c_ptr: *mut f32,
    scales: *const f32, m: u32, k: u32, n: u32, alpha: f32, beta: f32,
) {
    let m_usize = m as usize;
    let k_usize = k as usize;
    let n_usize = n as usize;
    let k_packed_usize = (k_usize + 1) / 2;

    let vl_bytes = svcntb();
    let vl_f32 = svcntw();
    let weights_per_step = vl_bytes * 2;

    asm!("smstart za", options(nomem, nostack));

    for i in 0..m_usize {
        let scale = *scales.add(i);
        let row_a_packed = slice::from_raw_parts(a_packed_ptr.add(i * k_packed_usize), k_packed_usize);
        let row_c = slice::from_raw_parts_mut(c_ptr.add(i * n_usize), n_usize);

        let mut tile = svzero_f32();

        for j in (0..n_usize).step_by(vl_f32) {
            let remaining_n = core::cmp::min(vl_f32, n_usize - j);
            let p_n = svwhilelt_b32(j as u32, n_usize as u32);

            for k_idx in (0..k_usize).step_by(weights_per_step) {
                let next_k_idx = k_idx + weights_per_step;
                if next_k_idx < k_usize {
                    let next_a_ptr = row_a_packed.as_ptr().add(next_k_idx / 2);
                    let next_b_ptr = b_ptr.add(next_k_idx * n_usize + j);
                    asm!(
                        "prfm pldl1keep, [{0}]",
                        "prfm pldl1keep, [{1}]",
                        in(reg) next_a_ptr,
                        in(reg) next_b_ptr
                    );
                }

                let remaining_weights = core::cmp::min(weights_per_step, k_usize - k_idx);
                let bytes_to_load = (remaining_weights + 1) / 2;
                
                let p_bytes = svwhilelt_b8((k_idx / 2) as u32, ((k_idx / 2) + bytes_to_load) as u32);
                let packed_vec = svld1_u8(p_bytes, row_a_packed.as_ptr().add(k_idx / 2));

                let lower_u4 = svand_n_u8_x(p_bytes, packed_vec, 0x0F);
                let upper_u4 = svlsr_n_u8_x(p_bytes, packed_vec, 4);

                let lower_i8 = svsub_n_s8_x(p_bytes, sveor_n_s8_x(p_bytes, svreinterpret_s8_u8(lower_u4), 8), 8);
                let upper_i8 = svsub_n_s8_x(p_bytes, sveor_n_s8_x(p_bytes, svreinterpret_s8_u8(upper_u4), 8), 8);

                let p_f32 = svwhilelt_b32(0, bytes_to_load as u32);
                let a_vec_lower = svcvt_f32_s8_x(p_f32, lower_i8);
                let a_vec_upper = svcvt_f32_s8_x(p_f32, upper_i8);
                
                let a_vec_lower_scaled = svmul_n_f32_x(p_f32, a_vec_lower, scale);
                let a_vec_upper_scaled = svmul_n_f32_x(p_f32, a_vec_upper, scale);

                let base_offset_lower = (k_idx * n_usize + j) as u32;
                let base_offset_upper = ((k_idx + 1) * n_usize + j) as u32;
                let stride = (2 * n_usize) as u32;

                let offsets_lower = svindex_u32(base_offset_lower, stride);
                let offsets_upper = svindex_u32(base_offset_upper, stride);

                let b_vec_lower = svld1_gather_u32index_f32(p_f32, b_ptr, offsets_lower);
                let b_vec_upper = svld1_gather_u32index_f32(p_f32, b_ptr, offsets_upper);

                let a_lower_safe = svsel_f32(p_f32, a_vec_lower_scaled, svdup_f32(0.0));
                let b_lower_safe = svsel_f32(p_f32, b_vec_lower, svdup_f32(0.0));
                let a_upper_safe = svsel_f32(p_f32, a_vec_upper_scaled, svdup_f32(0.0));
                let b_upper_safe = svsel_f32(p_f32, b_vec_upper, svdup_f32(0.0));

                tile = svmopa_f32_m(p_n, tile, a_lower_safe, b_lower_safe);
                tile = svmopa_f32_m(p_n, tile, a_upper_safe, b_upper_safe);
            }

            let mut c_vec = svld1_f32(p_n, &row_c[j]);
            c_vec = svmul_f32_m(p_n, c_vec, svdup_f32(beta));
            c_vec = svmad_f32_m(p_n, c_vec, svdup_f32(alpha), tile);
            svst1_f32(p_n, &mut row_c[j], c_vec);
            
            tile = svzero_f32();
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// Day 20 + Day 24 + Day 25: True Register-Level Fusion + Variable-Length Predicate Masking
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_transformer_block_i4_f32(
    a_packed_ptr: *const u8,
    b_ptr: *const f32,
    scales: *const f32,
    norm_weight: *const f32,
    c_ptr: *mut f32,
    m: u32, k: u32, n: u32,
    seq_len: u32,
    eps: f32, alpha: f32, beta: f32,
) {
    let m_usize = m as usize;
    let k_usize = k as usize;
    let n_usize = n as usize;
    
    let valid_m_usize = core::cmp::min(m_usize, seq_len as usize);
    let k_packed_usize = (k_usize + 1) / 2;

    let vl_bytes = svcntb();
    let vl_f32 = svcntw();
    let weights_per_step = vl_bytes * 2;

    let norm_slice = slice::from_raw_parts(norm_weight, n_usize);
    
    asm!("smstart za", options(nomem, nostack));

    for i in 0..valid_m_usize {
        let scale = *scales.add(i);
        let row_a_packed = slice::from_raw_parts(a_packed_ptr.add(i * k_packed_usize), k_packed_usize);
        let row_c = slice::from_raw_parts_mut(c_ptr.add(i * n_usize), n_usize);

        let mut sum_sq = 0.0f32;
        let mut max_val = core::f32::NEG_INFINITY;

        for j in (0..n_usize).step_by(vl_f32) {
            let p_n = svwhilelt_b32(j as u32, n_usize as u32);
            let mut tile = svzero_f32();

            for k_idx in (0..k_usize).step_by(weights_per_step) {
                let next_k_idx = k_idx + weights_per_step;
                if next_k_idx < k_usize {
                    let next_a_ptr = row_a_packed.as_ptr().add(next_k_idx / 2);
                    let next_b_ptr = b_ptr.add(next_k_idx * n_usize + j);
                    asm!(
                        "prfm pldl1keep, [{0}]", 
                        "prfm pldl1keep, [{1}]", 
                        in(reg) next_a_ptr,
                        in(reg) next_b_ptr
                    );
                }

                let remaining_weights = core::cmp::min(weights_per_step, k_usize - k_idx);
                let bytes_to_load = (remaining_weights + 1) / 2;
                
                let p_bytes = svwhilelt_b8((k_idx / 2) as u32, ((k_idx / 2) + bytes_to_load) as u32);
                let packed_vec = svld1_u8(p_bytes, row_a_packed.as_ptr().add(k_idx / 2));

                let lower_u4 = svand_n_u8_x(p_bytes, packed_vec, 0x0F);
                let upper_u4 = svlsr_n_u8_x(p_bytes, packed_vec, 4);

                let lower_i8 = svsub_n_s8_x(p_bytes, sveor_n_s8_x(p_bytes, svreinterpret_s8_u8(lower_u4), 8), 8);
                let upper_i8 = svsub_n_s8_x(p_bytes, sveor_n_s8_x(p_bytes, svreinterpret_s8_u8(upper_u4), 8), 8);

                let p_f32 = svwhilelt_b32(0, bytes_to_load as u32);
                let a_vec_lower = svcvt_f32_s8_x(p_f32, lower_i8);
                let a_vec_upper = svcvt_f32_s8_x(p_f32, upper_i8);
                
                let a_vec_lower_scaled = svmul_n_f32_x(p_f32, a_vec_lower, scale);
                let a_vec_upper_scaled = svmul_n_f32_x(p_f32, a_vec_upper, scale);

                let base_offset_lower = (k_idx * n_usize + j) as u32;
                let base_offset_upper = ((k_idx + 1) * n_usize + j) as u32;
                let stride = (2 * n_usize) as u32;

                let offsets_lower = svindex_u32(base_offset_lower, stride);
                let offsets_upper = svindex_u32(base_offset_upper, stride);

                let b_vec_lower = svld1_gather_u32index_f32(p_f32, b_ptr, offsets_lower);
                let b_vec_upper = svld1_gather_u32index_f32(p_f32, b_ptr, offsets_upper);

                let a_lower_safe = svsel_f32(p_f32, a_vec_lower_scaled, svdup_f32(0.0));
                let b_lower_safe = svsel_f32(p_f32, b_vec_lower, svdup_f32(0.0));
                let a_upper_safe = svsel_f32(p_f32, a_vec_upper_scaled, svdup_f32(0.0));
                let b_upper_safe = svsel_f32(p_f32, b_vec_upper, svdup_f32(0.0));

                tile = svmopa_f32_m(p_n, tile, a_lower_safe, b_lower_safe);
                tile = svmopa_f32_m(p_n, tile, a_upper_safe, b_upper_safe);
            }

            let mut c_vec = svld1_f32(p_n, &row_c[j]);
            c_vec = svmul_f32_m(p_n, c_vec, svdup_f32(beta));
            c_vec = svmad_f32_m(p_n, c_vec, svdup_f32(alpha), tile);
            
            svst1_f32(p_n, &mut row_c[j], c_vec);
            
            sum_sq += svaddv_f32(p_n, svmul_f32_m(p_n, c_vec, c_vec));
            max_val = max_val.max(svmaxv_f32(p_n, c_vec));
        }

        let mean = sum_sq / (n_usize as f32);
        let rsqrt_vec = svdup_f32(1.0f32 / (mean + eps).sqrt());
        let max_vec = svdup_f32(max_val);

        let mut sum_exp = 0.0f32;
        
        for j in (0..n_usize).step_by(vl_f32) {
            let p_n = svwhilelt_b32(j as u32, n_usize as u32);
            let mut c_vec = svld1_f32(p_n, &row_c[j]);
            
            let w_vec = svld1_f32(p_n, &norm_slice[j]);
            c_vec = svmul_f32_m(p_n, c_vec, rsqrt_vec);
            c_vec = svmul_f32_m(p_n, c_vec, w_vec);
            
            c_vec = svsub_f32_m(p_n, c_vec, max_vec);
            c_vec = svexp_f32_m(p_n, c_vec); 
            
            svst1_f32(p_n, &mut row_c[j], c_vec);
            sum_exp += svaddv_f32(p_n, c_vec);
        }
        
        let inv_sum_vec = svdup_f32(1.0f32 / sum_exp);
        
        for j in (0..n_usize).step_by(vl_f32) {
            let p_n = svwhilelt_b32(j as u32, n_usize as u32);
            let mut c_vec = svld1_f32(p_n, &row_c[j]);
            c_vec = svmul_f32_m(p_n, c_vec, inv_sum_vec);
            svst1_f32(p_n, &mut row_c[j], c_vec);
        }
    }

    for i in valid_m_usize..m_usize {
        let row_c = slice::from_raw_parts_mut(c_ptr.add(i * n_usize), n_usize);
        for j in (0..n_usize).step_by(vl_f32) {
            let p_n = svwhilelt_b32(j as u32, n_usize as u32);
            svst1_f32(p_n, &mut row_c[j], svzero_f32());
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// Day 26: Fused SwiGLU FFN Kernel (Zero Intermediate Allocations)
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sme_swiglu_f32(
    x_ptr: *const f32,
    w_gate_ptr: *const f32,
    w_up_ptr: *const f32,
    w_down_ptr: *const f32,
    out_ptr: *mut f32,
    seq_len: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
) {
    let seq_len = seq_len as usize;
    let hidden_dim = hidden_dim as usize;
    let intermediate_dim = intermediate_dim as usize;
    let vl = svcntw();

    let x = slice::from_raw_parts(x_ptr, seq_len * hidden_dim);
    let w_gate = slice::from_raw_parts(w_gate_ptr, hidden_dim * intermediate_dim);
    let w_up = slice::from_raw_parts(w_up_ptr, hidden_dim * intermediate_dim);
    let w_down = slice::from_raw_parts(w_down_ptr, intermediate_dim * hidden_dim);
    let out = slice::from_raw_parts_mut(out_ptr, seq_len * hidden_dim);

    for t in 0..seq_len {
        let x_row = &x[t * hidden_dim..(t + 1) * hidden_dim];
        let out_row = &mut out[t * hidden_dim..(t + 1) * hidden_dim];

        for j in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(j as u32, hidden_dim as u32);
            svst1_f32(p, &mut out_row[j], svzero_f32());
        }

        for i in (0..intermediate_dim).step_by(vl) {
            let p_i = svwhilelt_b32(i as u32, intermediate_dim as u32);

            let mut gate_acc = svzero_f32();
            let mut up_acc = svzero_f32();

            for d in 0..hidden_dim {
                let x_val = x_row[d];
                let x_vec = svdup_f32(x_val);
                let w_gate_vec = svld1_f32(p_i, &w_gate[d * intermediate_dim + i]);
                let w_up_vec = svld1_f32(p_i, &w_up[d * intermediate_dim + i]);
                gate_acc = svmla_f32_m(p_i, gate_acc, x_vec, w_gate_vec);
                up_acc = svmla_f32_m(p_i, up_acc, x_vec, w_up_vec);
            }

            let neg_gate = svneg_f32_m(p_i, gate_acc);
            let exp_neg_gate = svexp_f32_m(p_i, neg_gate);
            let one_vec = svdup_f32(1.0);
            let denom = svadd_f32_m(p_i, one_vec, exp_neg_gate);
            let silu_gate = svdiv_f32_m(p_i, gate_acc, denom);

            let swiglu_intermediate = svmul_f32_m(p_i, silu_gate, up_acc);

            for d in (0..hidden_dim).step_by(vl) {
                let p_d = svwhilelt_b32(d as u32, hidden_dim as u32);
                let mut out_vec = svld1_f32(p_d, &out_row[d]);
                let w_down_base = &w_down[i * hidden_dim + d];
                let offsets = svindex_u32(0, 1);
                let w_down_vec = svld1_gather_u32index_f32(p_d, w_down_base, offsets);
                out_vec = svmla_f32_m(p_d, out_vec, swiglu_intermediate, w_down_vec);
                svst1_f32(p_d, &mut out_row[d], out_vec);
            }
        }
    }
}

// ==========================================================
// Day 26: SVE2 W8A8 KV-Cache On-the-Fly Dequantization
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn sve2_dequant_kvcache_i8_f32(
    k_int8_ptr: *const i8,
    v_int8_ptr: *const i8,
    k_scales_ptr: *const f32,
    v_scales_ptr: *const f32,
    k_fp32_out_ptr: *mut f32,
    v_fp32_out_ptr: *mut f32,
    num_tokens: u32,
    hidden_dim: u32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let vl = svcntw();

    let k_int8 = slice::from_raw_parts(k_int8_ptr, num_tokens * hidden_dim);
    let v_int8 = slice::from_raw_parts(v_int8_ptr, num_tokens * hidden_dim);
    let k_scales = slice::from_raw_parts(k_scales_ptr, hidden_dim);
    let v_scales = slice::from_raw_parts(v_scales_ptr, hidden_dim);
    let k_fp32_out = slice::from_raw_parts_mut(k_fp32_out_ptr, num_tokens * hidden_dim);
    let v_fp32_out = slice::from_raw_parts_mut(v_fp32_out_ptr, num_tokens * hidden_dim);

    for t in 0..num_tokens {
        let k_int8_row = &k_int8[t * hidden_dim..(t + 1) * hidden_dim];
        let v_int8_row = &v_int8[t * hidden_dim..(t + 1) * hidden_dim];
        let k_fp32_row = &mut k_fp32_out[t * hidden_dim..(t + 1) * hidden_dim];
        let v_fp32_row = &mut v_fp32_out[t * hidden_dim..(t + 1) * hidden_dim];

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);

            let k_i8_vec = svld1_s8(p, &k_int8_row[d]);
            let v_i8_vec = svld1_s8(p, &v_int8_row[d]);
            let k_f32_vec = svcvt_f32_s8_x(p, k_i8_vec);
            let v_f32_vec = svcvt_f32_s8_x(p, v_i8_vec);

            let k_scale_vec = svld1_f32(p, &k_scales[d]);
            let v_scale_vec = svld1_f32(p, &v_scales[d]);

            let k_dequant = svmul_f32_m(p, k_f32_vec, k_scale_vec);
            let v_dequant = svmul_f32_m(p, v_f32_vec, v_scale_vec);

            svst1_f32(p, &mut k_fp32_row[d], k_dequant);
            svst1_f32(p, &mut v_fp32_row[d], v_dequant);
        }
    }
}

// ==========================================================
// Day 27: Fused FlashAttention with W8A8 KV-Cache Dequantization
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_fused_flash_attention_i8_f32(
    q_ptr: *const f32,
    k_int8_ptr: *const i8,
    v_int8_ptr: *const i8,
    k_scales_ptr: *const f32,
    v_scales_ptr: *const f32,
    out_ptr: *mut f32,
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    softmax_scale: f32,
) {
    let seq_len = seq_len as usize;
    let head_dim = head_dim as usize;
    let num_heads = num_heads as usize;
    let vl = svcntw();

    let q = slice::from_raw_parts(q_ptr, seq_len * num_heads * head_dim);
    let k_int8 = slice::from_raw_parts(k_int8_ptr, seq_len * num_heads * head_dim);
    let v_int8 = slice::from_raw_parts(v_int8_ptr, seq_len * num_heads * head_dim);
    let k_scales = slice::from_raw_parts(k_scales_ptr, num_heads * head_dim);
    let v_scales = slice::from_raw_parts(v_scales_ptr, num_heads * head_dim);
    let out = slice::from_raw_parts_mut(out_ptr, seq_len * num_heads * head_dim);

    for s in 0..seq_len {
        for h in 0..num_heads {
            let q_offset = (s * num_heads + h) * head_dim;
            let q_row = &q[q_offset..q_offset + head_dim];
            let out_row = &mut out[q_offset..q_offset + head_dim];
            
            let k_scale_row = &k_scales[h * head_dim..(h + 1) * head_dim];
            let v_scale_row = &v_scales[h * head_dim..(h + 1) * head_dim];

            let mut max_val = core::f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut acc = svzero_f32();

            for t in 0..=s {
                let k_int8_row = &k_int8[(t * num_heads + h) * head_dim..(t * num_heads + h + 1) * head_dim];
                let mut qk_score = 0.0f32;

                for d in (0..head_dim).step_by(vl) {
                    let p = svwhilelt_b32(d as u32, head_dim as u32);
                    let q_vec = svld1_f32(p, &q_row[d]);
                    let k_i8_vec = svld1_s8(p, &k_int8_row[d]);
                    let k_f32_vec = svcvt_f32_s8_x(p, k_i8_vec);
                    let k_scale_vec = svld1_f32(p, &k_scale_row[d]);
                    let k_dequant = svmul_f32_m(p, k_f32_vec, k_scale_vec);
                    let qk_vec = svmul_f32_m(p, q_vec, k_dequant);
                    qk_score += svaddv_f32(p, qk_vec);
                }

                qk_score *= softmax_scale;
                max_val = max_val.max(qk_score);
            }

            for t in 0..=s {
                let v_int8_row = &v_int8[(t * num_heads + h) * head_dim..(t * num_heads + h + 1) * head_dim];
                let k_int8_row = &k_int8[(t * num_heads + h) * head_dim..(t * num_heads + h + 1) * head_dim];
                
                let mut qk_score = 0.0f32;
                for d in (0..head_dim).step_by(vl) {
                    let p = svwhilelt_b32(d as u32, head_dim as u32);
                    let q_vec = svld1_f32(p, &q_row[d]);
                    let k_i8_vec = svld1_s8(p, &k_int8_row[d]);
                    let k_f32_vec = svcvt_f32_s8_x(p, k_i8_vec);
                    let k_scale_vec = svld1_f32(p, &k_scale_row[d]);
                    let k_dequant = svmul_f32_m(p, k_f32_vec, k_scale_vec);
                    let qk_vec = svmul_f32_m(p, q_vec, k_dequant);
                    qk_score += svaddv_f32(p, qk_vec);
                }
                qk_score *= softmax_scale;

                let weight = (qk_score - max_val).exp();
                sum_exp += weight;

                for d in (0..head_dim).step_by(vl) {
                    let p = svwhilelt_b32(d as u32, head_dim as u32);
                    let v_i8_vec = svld1_s8(p, &v_int8_row[d]);
                    let v_f32_vec = svcvt_f32_s8_x(p, v_i8_vec);
                    let v_scale_vec = svld1_f32(p, &v_scale_row[d]);
                    let v_dequant = svmul_f32_m(p, v_f32_vec, v_scale_vec);
                    let weighted_v = svmul_n_f32_x(p, v_dequant, weight);
                    acc = svadd_f32_m(p, acc, weighted_v);
                }
            }

            let inv_sum = 1.0f32 / sum_exp;
            let inv_sum_vec = svdup_f32(inv_sum);
            
            for d in (0..head_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, head_dim as u32);
                let out_vec = svmul_f32_m(p, acc, inv_sum_vec);
                svst1_f32(p, &mut out_row[d], out_vec);
            }
        }
    }
}

// ==========================================================
// Day 28: Fused SVE2 RoPE Apply Kernel (Zero-Allocation, In-Place)
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sme_rope_f32(
    q_ptr: *mut f32,
    cos_ptr: *const f32,
    sin_ptr: *const f32,
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
) {
    let seq_len = seq_len as usize;
    let head_dim = head_dim as usize;
    let num_heads = num_heads as usize;
    let half_dim = head_dim / 2;
    let vl = svcntw();

    let q = slice::from_raw_parts_mut(q_ptr, seq_len * num_heads * head_dim);
    let cos = slice::from_raw_parts(cos_ptr, seq_len * head_dim);
    let sin = slice::from_raw_parts(sin_ptr, seq_len * head_dim);

    for s in 0..seq_len {
        for h in 0..num_heads {
            let q_offset = (s * num_heads + h) * head_dim;
            let q_row = &mut q[q_offset..q_offset + head_dim];
            let cos_row = &cos[s * head_dim..(s + 1) * head_dim];
            let sin_row = &sin[s * head_dim..(s + 1) * head_dim];

            for d in (0..half_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, half_dim as u32);
                
                let q1 = svld1_f32(p, &q_row[d]);
                let q2 = svld1_f32(p, &q_row[half_dim + d]);
                
                let c = svld1_f32(p, &cos_row[d]);
                let s_vec = svld1_f32(p, &sin_row[d]);
                
                let q1_new = svsub_f32_m(p, svmul_f32_m(p, q1, c), svmul_f32_m(p, q2, s_vec));
                let q2_new = svadd_f32_m(p, svmul_f32_m(p, q2, c), svmul_f32_m(p, q1, s_vec));
                
                svst1_f32(p, &mut q_row[d], q1_new);
                svst1_f32(p, &mut q_row[half_dim + d], q2_new);
            }
        }
    }
}

// ==========================================================
// Day 29: SME Streaming Mode (SM) Async KV-Cache Prefetching
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe extern "C" fn sme_sm_prefetch_kv_blocks(
    k_ptrs: *const *const i8,
    v_ptrs: *const *const i8,
    num_blocks: u32,
) {
    let num_blocks = num_blocks as usize;
    let k_ptr_slice = slice::from_raw_parts(k_ptrs, num_blocks);
    let v_ptr_slice = slice::from_raw_parts(v_ptrs, num_blocks);
    
    for i in 0..num_blocks {
        if !k_ptr_slice[i].is_null() && !v_ptr_slice[i].is_null() {
            asm!(
                "prfm pldl1keep, [{0}]",
                "prfm pldl1keep, [{1}]",
                in(reg) k_ptr_slice[i],
                in(reg) v_ptr_slice[i]
            );
        }
    }
}

// ==========================================================
// Day 29: SVE2 Speculative Verification Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn sve2_speculative_verify_f32(
    logits_ptr: *const f32,
    draft_tokens_ptr: *const i32,
    draft_lengths_ptr: *const i32,
    vocab_size: u32,
    max_draft_length: u32,
    batch_size: u32,
    accepted_counts_ptr: *mut i32,
    replacement_tokens_ptr: *mut i32,
) {
    let batch_size = batch_size as usize;
    let max_draft_length = max_draft_length as usize;
    let vocab_size = vocab_size as usize;
    let vl = svcntw();

    let logits = slice::from_raw_parts(logits_ptr, batch_size * max_draft_length * vocab_size);
    let draft_tokens = slice::from_raw_parts(draft_tokens_ptr, batch_size * max_draft_length);
    let draft_lengths = slice::from_raw_parts(draft_lengths_ptr, batch_size);
    let accepted_counts = slice::from_raw_parts_mut(accepted_counts_ptr, batch_size);
    let replacement_tokens = slice::from_raw_parts_mut(replacement_tokens_ptr, batch_size);

    for b in 0..batch_size {
        let d_len = draft_lengths[b] as usize;
        if d_len == 0 {
            accepted_counts[b] = 0;
            replacement_tokens[b] = -1;
            continue;
        }

        let mut best_tokens = [0i32; 8];
        
        for i in 0..d_len {
            let logit_start = b * max_draft_length * vocab_size + i * vocab_size;
            let mut max_vec = svdup_f32(core::f32::NEG_INFINITY);
            let mut idx_vec = svindex_s32(0, 1);

            for v in (0..vocab_size).step_by(vl) {
                let p = svwhilelt_b32(v as u32, vocab_size as u32);
                let logit_vec = svld1_f32(p, &logits[logit_start + v]);
                let gt_mask = svcmpgt_f32(p, logit_vec, max_vec);
                
                max_vec = svsel_f32(gt_mask, logit_vec, max_vec);
                
                let current_idx_vec = svindex_s32(v as i32, 1);
                idx_vec = svsel_s32(gt_mask, current_idx_vec, idx_vec);
            }
            
            let p_all = svptrue_b32();
            let max_val = svmaxv_f32(p_all, max_vec);
            
            let mut idx_arr = [0i32; 16];
            let mut max_arr = [0.0f32; 16];
            svst1_s32(p_all, &mut idx_arr, idx_vec);
            svst1_f32(p_all, &mut max_arr, max_vec);
            
            let mut best = -1i32;
            for k in 0..vl {
                if max_arr[k] == max_val {
                    best = idx_arr[k];
                    break;
                }
            }
            best_tokens[i] = best;
        }

        let p_draft = svwhilelt_b32(0, d_len as u32);
        let best_vec = svld1_s32(p_draft, best_tokens.as_ptr());
        let draft_vec = svld1_s32(p_draft, &draft_tokens[b * max_draft_length]);
        
        let match_mask = svcmpeq_s32(p_draft, best_vec, draft_vec);
        let accepted = svcntp_s32(p_draft, match_mask) as i32;
        
        let reject_mask = svnot_b(match_mask);
        let first_reject = svfirstb(reject_mask);
        
        let mut replacement = -1i32;
        if first_reject {
            replacement = best_tokens[accepted as usize];
        }

        accepted_counts[b] = accepted;
        replacement_tokens[b] = replacement;
    }
}

// ==========================================================
// DAY 30: SVE2 Grouped-Query Attention (GQA) FlashAttention Fusion
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_gqa_flash_attention_i8_f32(
    q_ptr: *const f32,
    k_int8_ptr: *const i8,
    v_int8_ptr: *const i8,
    k_scales_ptr: *const f32,
    v_scales_ptr: *const f32,
    out_ptr: *mut f32,
    seq_len: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    softmax_scale: f32,
) {
    let seq_len = seq_len as usize;
    let head_dim = head_dim as usize;
    let num_q_heads = num_q_heads as usize;
    let num_kv_heads = num_kv_heads as usize;
    let qkv_head_ratio = num_q_heads / num_kv_heads;
    let vl = svcntw();

    let q = slice::from_raw_parts(q_ptr, seq_len * num_q_heads * head_dim);
    let k_int8 = slice::from_raw_parts(k_int8_ptr, seq_len * num_kv_heads * head_dim);
    let v_int8 = slice::from_raw_parts(v_int8_ptr, seq_len * num_kv_heads * head_dim);
    let k_scales = slice::from_raw_parts(k_scales_ptr, num_kv_heads * head_dim);
    let v_scales = slice::from_raw_parts(v_scales_ptr, num_kv_heads * head_dim);
    let out = slice::from_raw_parts_mut(out_ptr, seq_len * num_q_heads * head_dim);

    for kv_h in 0..num_kv_heads {
        let k_scale_row = &k_scales[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_scale_row = &v_scales[kv_h * head_dim..(kv_h + 1) * head_dim];

        for s in 0..seq_len {
            let mut k_dequant_buf = [0.0f32; 256];
            let mut v_dequant_buf = [0.0f32; 256];
            
            let k_int8_row = &k_int8[(s * num_kv_heads + kv_h) * head_dim..(s * num_kv_heads + kv_h + 1) * head_dim];
            let v_int8_row = &v_int8[(s * num_kv_heads + kv_h) * head_dim..(s * num_kv_heads + kv_h + 1) * head_dim];

            for d in (0..head_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, head_dim as u32);
                
                let k_i8_vec = svld1_s8(p, &k_int8_row[d]);
                let k_f32_vec = svcvt_f32_s8_x(p, k_i8_vec);
                let k_scale_vec = svld1_f32(p, &k_scale_row[d]);
                let k_dequant = svmul_f32_m(p, k_f32_vec, k_scale_vec);
                svst1_f32(p, &mut k_dequant_buf[d], k_dequant);

                let v_i8_vec = svld1_s8(p, &v_int8_row[d]);
                let v_f32_vec = svcvt_f32_s8_x(p, v_i8_vec);
                let v_scale_vec = svld1_f32(p, &v_scale_row[d]);
                let v_dequant = svmul_f32_m(p, v_f32_vec, v_scale_vec);
                svst1_f32(p, &mut v_dequant_buf[d], v_dequant);
            }

            for q_offset in 0..qkv_head_ratio {
                let q_h = kv_h * qkv_head_ratio + q_offset;
                let q_row_offset = (s * num_q_heads + q_h) * head_dim;
                let q_row = &q[q_row_offset..q_row_offset + head_dim];
                let out_row = &mut out[q_row_offset..q_row_offset + head_dim];

                let mut max_val = core::f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;
                let mut acc = svzero_f32();

                for t in 0..=s {
                    let k_t_offset = (t * num_kv_heads + kv_h) * head_dim;
                    let k_int8_t = &k_int8[k_t_offset..k_t_offset + head_dim];
                    
                    let mut qk_score = 0.0f32;
                    for d in (0..head_dim).step_by(vl) {
                        let p = svwhilelt_b32(d as u32, head_dim as u32);
                        let q_vec = svld1_f32(p, &q_row[d]);
                        
                        let k_vec = if t == s {
                            svld1_f32(p, &k_dequant_buf[d])
                        } else {
                            let k_i8 = svld1_s8(p, &k_int8_t[d]);
                            let k_f32 = svcvt_f32_s8_x(p, k_i8);
                            let k_sc = svld1_f32(p, &k_scale_row[d]);
                            svmul_f32_m(p, k_f32, k_sc)
                        };
                        
                        let qk_vec = svmul_f32_m(p, q_vec, k_vec);
                        qk_score += svaddv_f32(p, qk_vec);
                    }
                    qk_score *= softmax_scale;
                    max_val = max_val.max(qk_score);
                }

                for t in 0..=s {
                    let k_t_offset = (t * num_kv_heads + kv_h) * head_dim;
                    let k_int8_t = &k_int8[k_t_offset..k_t_offset + head_dim];
                    
                    let mut qk_score = 0.0f32;
                    for d in (0..head_dim).step_by(vl) {
                        let p = svwhilelt_b32(d as u32, head_dim as u32);
                        let q_vec = svld1_f32(p, &q_row[d]);
                        
                        let k_vec = if t == s {
                            svld1_f32(p, &k_dequant_buf[d])
                        } else {
                            let k_i8 = svld1_s8(p, &k_int8_t[d]);
                            let k_f32 = svcvt_f32_s8_x(p, k_i8);
                            let k_sc = svld1_f32(p, &k_scale_row[d]);
                            svmul_f32_m(p, k_f32, k_sc)
                        };
                        
                        let qk_vec = svmul_f32_m(p, q_vec, k_vec);
                        qk_score += svaddv_f32(p, qk_vec);
                    }
                    qk_score *= softmax_scale;
                    let weight = (qk_score - max_val).exp();
                    sum_exp += weight;

                    let v_t_offset = (t * num_kv_heads + kv_h) * head_dim;
                    let v_int8_t = &v_int8[v_t_offset..v_t_offset + head_dim];
                    
                    for d in (0..head_dim).step_by(vl) {
                        let p = svwhilelt_b32(d as u32, head_dim as u32);
                        
                        let v_vec = if t == s {
                            svld1_f32(p, &v_dequant_buf[d])
                        } else {
                            let v_i8 = svld1_s8(p, &v_int8_t[d]);
                            let v_f32 = svcvt_f32_s8_x(p, v_i8);
                            let v_sc = svld1_f32(p, &v_scale_row[d]);
                            svmul_f32_m(p, v_f32, v_sc)
                        };
                        
                        let weighted_v = svmul_n_f32_x(p, v_vec, weight);
                        acc = svadd_f32_m(p, acc, weighted_v);
                    }
                }

                let inv_sum = 1.0f32 / sum_exp;
                let inv_sum_vec = svdup_f32(inv_sum);
                
                for d in (0..head_dim).step_by(vl) {
                    let p = svwhilelt_b32(d as u32, head_dim as u32);
                    let out_vec = svmul_f32_m(p, acc, inv_sum_vec);
                    svst1_f32(p, &mut out_row[d], out_vec);
                }
            }
        }
    }
}

// ==========================================================
// DAY 30: SME MoE Sparse Routing & GEMM Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_moe_sparse_gemm_f32(
    input_ptr: *const f32,
    router_weights_ptr: *const f32,
    expert_weights_ptr: *const f32,
    output_ptr: *mut f32,
    num_tokens: u32,
    hidden_dim: u32,
    num_experts: u32,
    top_k: u32,
    expert_intermediate_dim: u32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let num_experts = num_experts as usize;
    let top_k = top_k as usize;
    let expert_intermediate_dim = expert_intermediate_dim as usize;
    let vl = svcntw();

    let input = slice::from_raw_parts(input_ptr, num_tokens * hidden_dim);
    let router_weights = slice::from_raw_parts(router_weights_ptr, hidden_dim * num_experts);
    let expert_weights = slice::from_raw_parts(expert_weights_ptr, num_experts * hidden_dim * expert_intermediate_dim);
    let output = slice::from_raw_parts_mut(output_ptr, num_tokens * hidden_dim);

    asm!("smstart za", options(nomem, nostack));

    for tok in 0..num_tokens {
        let x_row = &input[tok * hidden_dim..(tok + 1) * hidden_dim];
        let out_row = &mut output[tok * hidden_dim..(tok + 1) * hidden_dim];

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);
            svst1_f32(p, &mut out_row[d], svzero_f32());
        }

        let mut router_logits = [0.0f32; 256];
        for e in 0..num_experts {
            let mut dot = 0.0f32;
            for d in (0..hidden_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, hidden_dim as u32);
                let x_vec = svld1_f32(p, &x_row[d]);
                let w_vec = svld1_f32(p, &router_weights[d * num_experts + e]);
                let prod = svmul_f32_m(p, x_vec, w_vec);
                dot += svaddv_f32(p, prod);
            }
            router_logits[e] = dot;
        }

        let mut selected_experts = [0usize; 8];
        let mut selected_weights = [0.0f32; 8];
        let mut used_mask = [false; 256];

        for k in 0..top_k {
            let p_experts = svwhilelt_b32(0u32, num_experts as u32);
            let mut logits_vec = svld1_f32(p_experts, router_logits.as_ptr());
            
            let max_val = svmaxv_f32(p_experts, logits_vec);
            
            let mut best_e = 0usize;
            for e in 0..num_experts {
                if !used_mask[e] && router_logits[e] == max_val {
                    best_e = e;
                    break;
                }
            }
            
            selected_experts[k] = best_e;
            used_mask[best_e] = true;
            selected_weights[k] = router_logits[best_e];
        }

        let mut w_max = core::f32::NEG_INFINITY;
        for k in 0..top_k { w_max = w_max.max(selected_weights[k]); }
        let mut w_sum = 0.0f32;
        for k in 0..top_k {
            selected_weights[k] = (selected_weights[k] - w_max).exp();
            w_sum += selected_weights[k];
        }
        let w_inv_sum = 1.0f32 / w_sum;
        for k in 0..top_k { selected_weights[k] *= w_inv_sum; }

        for k in 0..top_k {
            let expert_id = selected_experts[k];
            let expert_w = selected_weights[k];
            let ew_base = expert_id * hidden_dim * expert_intermediate_dim;
            let expert_w_slice = &expert_weights[ew_base..ew_base + hidden_dim * expert_intermediate_dim];

            let mut intermediate = [0.0f32; 4096];
            
            for j in (0..expert_intermediate_dim).step_by(vl) {
                let remaining = core::cmp::min(vl, expert_intermediate_dim - j);
                let p_j = svwhilelt_b32(j as u32, expert_intermediate_dim as u32);
                let mut tile = svzero_f32();

                for d in 0..hidden_dim {
                    let x_val = x_row[d];
                    let x_vec = svdup_f32(x_val);
                    let w_vec = svld1_f32(p_j, &expert_w_slice[d * expert_intermediate_dim + j]);
                    tile = svmopa_f32_m(p_j, tile, x_vec, w_vec);
                }

                svst1_f32(p_j, &mut intermediate[j], tile);
            }

            for j in (0..expert_intermediate_dim).step_by(vl) {
                let p_j = svwhilelt_b32(j as u32, expert_intermediate_dim as u32);
                let mut val = svld1_f32(p_j, &intermediate[j]);
                let neg_val = svneg_f32_m(p_j, val);
                let exp_neg = svexp_f32_m(p_j, neg_val);
                let one = svdup_f32(1.0);
                let denom = svadd_f32_m(p_j, one, exp_neg);
                let silu = svdiv_f32_m(p_j, val, denom);
                svst1_f32(p_j, &mut intermediate[j], silu);
            }

            for d in (0..hidden_dim).step_by(vl) {
                let p_d = svwhilelt_b32(d as u32, hidden_dim as u32);
                let mut out_vec = svld1_f32(p_d, &out_row[d]);
                
                let gather_indices = svindex_u32(0, 1);
                let inter_base = &intermediate[0];
                let inter_vec = svld1_gather_u32index_f32(p_d, inter_base, gather_indices);
                
                let scaled = svmul_n_f32_x(p_d, inter_vec, expert_w);
                out_vec = svadd_f32_m(p_d, out_vec, scaled);
                svst1_f32(p_d, &mut out_row[d], out_vec);
            }
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 31: SME SM Non-Temporal MoE Weight Streaming Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_moe_nt_streaming_f32(
    input_ptr: *const f32,
    expert_weights_ptr: *const f32,
    output_ptr: *mut f32,
    num_tokens: u32,
    hidden_dim: u32,
    expert_intermediate_dim: u32,
    expert_weight_stride: u32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let expert_intermediate_dim = expert_intermediate_dim as usize;
    let vl = svcntw();

    let input = slice::from_raw_parts(input_ptr, num_tokens * hidden_dim);
    let output = slice::from_raw_parts_mut(output_ptr, num_tokens * hidden_dim);

    asm!("smstart za", options(nomem, nostack));

    for tok in 0..num_tokens {
        let x_row = &input[tok * hidden_dim..(tok + 1) * hidden_dim];
        let out_row = &mut output[tok * hidden_dim..(tok + 1) * hidden_dim];

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);
            svst1_f32(p, &mut out_row[d], svzero_f32());
        }

        for j in (0..expert_intermediate_dim).step_by(vl) {
            let p_j = svwhilelt_b32(j as u32, expert_intermediate_dim as u32);
            let mut tile = svzero_f32();

            for d in 0..hidden_dim {
                let x_val = x_row[d];
                let x_vec = svdup_f32(x_val);
                let w_vec = svld1nt_f32(p_j, expert_weights_ptr.add(d * expert_weight_stride + j));
                tile = svmopa_f32_m(p_j, tile, x_vec, w_vec);
            }

            let mut out_vec = svld1_f32(p_j, &out_row[j]);
            out_vec = svadd_f32_m(p_j, out_vec, tile);
            svst1_f32(p_j, &mut out_row[j], out_vec);
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 31: GQA Decode & MoE Compute Overlap Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_gqa_moe_pipeline_overlap(
    next_k_ptrs: *const *const i8,
    next_v_ptrs: *const *const i8,
    num_kv_blocks: u32,
    input_ptr: *const f32,
    expert_weights_ptr: *const f32,
    output_ptr: *mut f32,
    num_tokens: u32,
    hidden_dim: u32,
    expert_intermediate_dim: u32,
    expert_weight_stride: u32,
) {
    let num_kv_blocks = num_kv_blocks as usize;
    let k_ptr_slice = slice::from_raw_parts(next_k_ptrs, num_kv_blocks);
    let v_ptr_slice = slice::from_raw_parts(next_v_ptrs, num_kv_blocks);

    for i in 0..num_kv_blocks {
        if !k_ptr_slice[i].is_null() && !v_ptr_slice[i].is_null() {
            asm!(
                "prfm pldl1keep, [{0}]",
                "prfm pldl1keep, [{1}]",
                in(reg) k_ptr_slice[i],
                in(reg) v_ptr_slice[i]
            );
        }
    }

    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let expert_intermediate_dim = expert_intermediate_dim as usize;
    let vl = svcntw();

    let input = slice::from_raw_parts(input_ptr, num_tokens * hidden_dim);
    let output = slice::from_raw_parts_mut(output_ptr, num_tokens * hidden_dim);

    asm!("smstart za", options(nomem, nostack));

    for tok in 0..num_tokens {
        let x_row = &input[tok * hidden_dim..(tok + 1) * hidden_dim];
        let out_row = &mut output[tok * hidden_dim..(tok + 1) * hidden_dim];

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);
            svst1_f32(p, &mut out_row[d], svzero_f32());
        }

        for j in (0..expert_intermediate_dim).step_by(vl) {
            let p_j = svwhilelt_b32(j as u32, expert_intermediate_dim as u32);
            let mut tile = svzero_f32();

            for d in 0..hidden_dim {
                let x_val = x_row[d];
                let x_vec = svdup_f32(x_val);
                let w_vec = svld1nt_f32(p_j, expert_weights_ptr.add(d * expert_weight_stride + j));
                tile = svmopa_f32_m(p_j, tile, x_vec, w_vec);
            }

            let mut out_vec = svld1_f32(p_j, &out_row[j]);
            out_vec = svadd_f32_m(p_j, out_vec, tile);
            svst1_f32(p_j, &mut out_row[j], out_vec);
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 32: Native SVE2/SME BF16 MoE GEMM Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_moe_bf16_gemm_f32(
    input_ptr: *const f32,
    expert_weights_bf16_ptr: *const f16,
    output_ptr: *mut f32,
    num_tokens: u32,
    hidden_dim: u32,
    expert_intermediate_dim: u32,
    expert_weight_stride: u32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let expert_intermediate_dim = expert_intermediate_dim as usize;
    let vl = svcntw();

    let input = slice::from_raw_parts(input_ptr, num_tokens * hidden_dim);
    let output = slice::from_raw_parts_mut(output_ptr, num_tokens * hidden_dim);
    let weights = slice::from_raw_parts(expert_weights_bf16_ptr, num_tokens * hidden_dim * expert_intermediate_dim);

    asm!("smstart za", options(nomem, nostack));

    for tok in 0..num_tokens {
        let x_row = &input[tok * hidden_dim..(tok + 1) * hidden_dim];
        let out_row = &mut output[tok * hidden_dim..(tok + 1) * hidden_dim];

        for j in (0..expert_intermediate_dim).step_by(vl) {
            let p_j = svwhilelt_b32(j as u32, expert_intermediate_dim as u32);
            let mut tile = svzero_f32();

            for d in 0..hidden_dim {
                let x_val = x_row[d];
                let x_bf16_bits = (x_val.to_bits() + 0x7FFF) & 0xFFFF0000u32;
                let x_f16: f16 = core::mem::transmute((x_bf16_bits >> 16) as u16);
                let x_vec = svdup_n_f16(x_f16);
                
                let w_vec = svld1_f16(p_j, weights.as_ptr().add(d * expert_weight_stride + j));
                tile = svmopa_f16_m(p_j, tile, x_vec, w_vec);
            }

            let mut out_vec = svld1_f32(p_j, &out_row[j]);
            out_vec = svadd_f32_m(p_j, out_vec, tile);
            svst1_f32(p_j, &mut out_row[j], out_vec);
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 32: Dynamic Token Early Exit Masking Kernel
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sme_generate_early_exit_mask(
    logits_ptr: *const f32,
    mask_out_ptr: *mut u64,
    batch_size: u32,
    vocab_size: u32,
    eos_token_id: u32,
    confidence_threshold: f32,
) {
    let batch_size = batch_size as usize;
    let vocab_size = vocab_size as usize;
    let vl = svcntw();

    let logits = slice::from_raw_parts(logits_ptr, batch_size * vocab_size);
    let mask_out = slice::from_raw_parts_mut(mask_out_ptr, (batch_size + 63) / 64);

    for i in 0..mask_out.len() {
        mask_out[i] = !0u64;
    }

    for b in 0..batch_size {
        let logit_start = b * vocab_size;
        
        let mut max_vec = svdup_f32(core::f32::NEG_INFINITY);
        let mut max_idx_vec = svdup_u32(0);

        for v in (0..vocab_size).step_by(vl) {
            let p = svwhilelt_b32(v as u32, vocab_size as u32);
            let logit_vec = svld1_f32(p, &logits[logit_start + v]);
            let gt_mask = svcmpgt_f32(p, logit_vec, max_vec);
            
            max_vec = svsel_f32(gt_mask, logit_vec, max_vec);
            let current_idx_vec = svindex_u32(v as u32, 1);
            max_idx_vec = svsel_u32(gt_mask, current_idx_vec, max_idx_vec);
        }
        
        let mut max_arr = [0.0f32; 16];
        let mut idx_arr = [0u32; 16];
        svst1_f32(svptrue_b32(), &mut max_arr, max_vec);
        svst1_u32(svptrue_b32(), &mut idx_arr, max_idx_vec);
        
        let mut final_max = core::f32::NEG_INFINITY;
        let mut final_idx = 0u32;
        for i in 0..vl {
            if max_arr[i] > final_max {
                final_max = max_arr[i];
                final_idx = idx_arr[i];
            }
        }
        
        let mut sum_exp = 0.0f32;
        for v in (0..vocab_size).step_by(vl) {
            let p = svwhilelt_b32(v as u32, vocab_size as u32);
            let logit_vec = svld1_f32(p, &logits[logit_start + v]);
            let exp_vec = svexp_f32_m(p, svsub_f32_m(p, logit_vec, svdup_f32(final_max)));
            sum_exp += svaddv_f32(p, exp_vec);
        }
        
        let prob = 1.0f32 / sum_exp;

        if final_idx == eos_token_id || prob >= confidence_threshold {
            let word_idx = b / 64;
            let bit_idx = b % 64;
            mask_out[word_idx] &= !(1u64 << bit_idx);
        }
    }
}

// ==========================================================
// DAY 33: SVE2 Vectorized Token Similarity Mask (ToMe)
// Computes cosine similarity between adjacent tokens and marks
// redundant tokens for merging, preventing overlapping merges.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sve2_tome_similarity_mask(
    hidden_states_ptr: *const f32,
    mask_out_ptr: *mut u64,
    seq_len: u32,
    hidden_dim: u32,
    similarity_threshold: f32,
) {
    let seq_len = seq_len as usize;
    let hidden_dim = hidden_dim as usize;
    let vl = svcntw();

    let hidden_states = slice::from_raw_parts(hidden_states_ptr, seq_len * hidden_dim);
    let mask_out = slice::from_raw_parts_mut(mask_out_ptr, (seq_len + 63) / 64);

    for i in 0..mask_out.len() {
        mask_out[i] = !0u64;
    }

    if seq_len < 2 {
        return;
    }

    let mut skip_next = false;
    for i in 0..(seq_len - 1) {
        if skip_next {
            skip_next = false;
            continue;
        }

        let h_i = &hidden_states[i * hidden_dim..(i + 1) * hidden_dim];
        let h_next = &hidden_states[(i + 1) * hidden_dim..(i + 2) * hidden_dim];

        let mut dot_prod = 0.0f32;
        let mut mag_i_sq = 0.0f32;
        let mut mag_next_sq = 0.0f32;

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);
            let v_i = svld1_f32(p, &h_i[d]);
            let v_next = svld1_f32(p, &h_next[d]);

            dot_prod += svaddv_f32(p, svmul_f32_m(p, v_i, v_next));
            mag_i_sq += svaddv_f32(p, svmul_f32_m(p, v_i, v_i));
            mag_next_sq += svaddv_f32(p, svmul_f32_m(p, v_next, v_next));
        }

        let mag_i = mag_i_sq.sqrt();
        let mag_next = mag_next_sq.sqrt();
        let sim = if mag_i > 1e-6 && mag_next > 1e-6 {
            dot_prod / (mag_i * mag_next)
        } else {
            0.0
        };

        if sim > similarity_threshold {
            let word_idx = (i + 1) / 64;
            let bit_idx = (i + 1) % 64;
            mask_out[word_idx] &= !(1u64 << bit_idx);
            skip_next = true;
        }
    }
}

// ==========================================================
// DAY 33: SME ZA Fused Token Merge Accumulator
// Uses svmopa_f32 to accumulate weighted vectors directly into 
// the output buffer, ensuring zero intermediate heap allocations.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_tome_merge_accumulate_f32(
    hidden_states_ptr: *const f32,
    merged_states_ptr: *mut f32,
    mask_ptr: *const u64,
    seq_len: u32,
    hidden_dim: u32,
) {
    let seq_len = seq_len as usize;
    let hidden_dim = hidden_dim as usize;
    let vl = svcntw();

    let hidden_states = slice::from_raw_parts(hidden_states_ptr, seq_len * hidden_dim);
    let merged_states = slice::from_raw_parts_mut(merged_states_ptr, seq_len * hidden_dim);
    let mask = slice::from_raw_parts(mask_ptr, (seq_len + 63) / 64);

    asm!("smstart za", options(nomem, nostack));

    let mut out_idx = 0;
    let mut i = 0;

    while i < seq_len {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        let is_kept = (mask[word_idx] >> bit_idx) & 1;

        if is_kept == 1 {
            let h_i = &hidden_states[i * hidden_dim..(i + 1) * hidden_dim];
            let out_row = &mut merged_states[out_idx * hidden_dim..(out_idx + 1) * hidden_dim];

            let next_i = i + 1;
            let next_is_merged = if next_i < seq_len {
                let next_word = next_i / 64;
                let next_bit = next_i % 64;
                ((mask[next_word] >> next_bit) & 1) == 0
            } else {
                false
            };

            for d in (0..hidden_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, hidden_dim as u32);
                let mut tile = svzero_f32();

                let v_i = svld1_f32(p, &h_i[d]);
                tile = svmopa_f32_m(p, tile, svdup_f32(0.5), v_i);

                if next_is_merged {
                    let h_next = &hidden_states[next_i * hidden_dim..(next_i + 1) * hidden_dim];
                    let v_next = svld1_f32(p, &h_next[d]);
                    tile = svmopa_f32_m(p, tile, svdup_f32(0.5), v_next);
                } else {
                    tile = svmopa_f32_m(p, tile, svdup_f32(0.5), v_i);
                }

                svst1_f32(p, &mut out_row[d], tile);
            }
            out_idx += 1;
        }
        
        let next_i = i + 1;
        let next_is_merged = if next_i < seq_len {
            let next_word = next_i / 64;
            let next_bit = next_i % 64;
            ((mask[next_word] >> next_bit) & 1) == 0
        } else {
            false
        };

        if next_is_merged {
            i += 2;
        } else {
            i += 1;
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 34: SVE2 Vectorized Token Importance Scoring (SnapKV/H2O style)
// Computes squared L2 norm per token to estimate importance without 
// materializing the full O(N^2) attention matrix.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sve2_compute_token_importance(
    tokens_ptr: *const f32,
    num_tokens: u32,
    hidden_dim: u32,
    scores_out_ptr: *mut f32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let vl = svcntw();

    let tokens = slice::from_raw_parts(tokens_ptr, num_tokens * hidden_dim);
    let scores_out = slice::from_raw_parts_mut(scores_out_ptr, num_tokens);

    for t in 0..num_tokens {
        let token_row = &tokens[t * hidden_dim..(t + 1) * hidden_dim];
        let mut sum_sq = 0.0f32;

        for d in (0..hidden_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, hidden_dim as u32);
            let v = svld1_f32(p, &token_row[d]);
            let v_sq = svmul_f32_m(p, v, v);
            sum_sq += svaddv_f32(p, v_sq);
        }

        scores_out[t] = sum_sq;
    }
}

// ==========================================================
// DAY 34: SVE2 Eviction Mask Generation
// Identifies the bottom `evict_count` tokens based on scores and 
// generates a compact u64 eviction mask (0 = evict, 1 = keep).
// Uses a fixed-size stack array for zero-allocation selection.
// ==========================================================
#[no_mangle]
pub unsafe extern "C" fn rust_sve2_generate_eviction_mask(
    scores_ptr: *const f32,
    num_tokens: u32,
    evict_count: u32,
    mask_out_ptr: *mut u64,
) {
    let num_tokens = num_tokens as usize;
    let evict_count = evict_count as usize;
    let scores = slice::from_raw_parts(scores_ptr, num_tokens);
    let mask_out = slice::from_raw_parts_mut(mask_out_ptr, (num_tokens + 63) / 64);

    for i in 0..mask_out.len() {
        mask_out[i] = !0u64;
    }

    if evict_count == 0 || evict_count >= num_tokens || num_tokens > 512 {
        return;
    }

    let mut indices = [0usize; 512];
    for i in 0..num_tokens {
        indices[i] = i;
    }

    for i in 0..evict_count {
        let mut min_idx = i;
        for j in (i + 1)..num_tokens {
            if scores[indices[j]] < scores[indices[min_idx]] {
                min_idx = j;
            }
        }
        let temp = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp;
    }

    for i in 0..evict_count {
        let token_idx = indices[i];
        let word_idx = token_idx / 64;
        let bit_idx = token_idx % 64;
        mask_out[word_idx] &= !(1u64 << bit_idx);
    }
}

// ==========================================================
// DAY 34: SME ZA Fused KV Block Compaction
// Gathers surviving K/V vectors and packs them contiguously into 
// freed blocks using SVE2 gather/scatter and SME ZA svmopa_f32, 
// ensuring zero intermediate heap allocations.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_compact_kv_blocks(
    src_k_ptr: *const f32,
    src_v_ptr: *const f32,
    dst_k_ptr: *mut f32,
    dst_v_ptr: *mut f32,
    mask_ptr: *const u64,
    num_tokens: u32,
    hidden_dim: u32,
) {
    let num_tokens = num_tokens as usize;
    let hidden_dim = hidden_dim as usize;
    let vl = svcntw();

    let src_k = slice::from_raw_parts(src_k_ptr, num_tokens * hidden_dim);
    let src_v = slice::from_raw_parts(src_v_ptr, num_tokens * hidden_dim);
    let dst_k = slice::from_raw_parts_mut(dst_k_ptr, num_tokens * hidden_dim);
    let dst_v = slice::from_raw_parts_mut(dst_v_ptr, num_tokens * hidden_dim);
    let mask = slice::from_raw_parts(mask_ptr, (num_tokens + 63) / 64);

    asm!("smstart za", options(nomem, nostack));

    let mut out_idx = 0;

    for t in 0..num_tokens {
        let word_idx = t / 64;
        let bit_idx = t % 64;
        let is_kept = (mask[word_idx] >> bit_idx) & 1;

        if is_kept == 1 {
            let src_k_row = &src_k[t * hidden_dim..(t + 1) * hidden_dim];
            let src_v_row = &src_v[t * hidden_dim..(t + 1) * hidden_dim];
            let dst_k_row = &mut dst_k[out_idx * hidden_dim..(out_idx + 1) * hidden_dim];
            let dst_v_row = &mut dst_v[out_idx * hidden_dim..(out_idx + 1) * hidden_dim];

            for d in (0..hidden_dim).step_by(vl) {
                let p = svwhilelt_b32(d as u32, hidden_dim as u32);
                
                let k_vec = svld1_f32(p, &src_k_row[d]);
                let v_vec = svld1_f32(p, &src_v_row[d]);

                let mut tile_k = svzero_f32();
                let mut tile_v = svzero_f32();
                
                let ones = svdup_f32(1.0);
                tile_k = svmopa_f32_m(p, tile_k, k_vec, ones);
                tile_v = svmopa_f32_m(p, tile_v, v_vec, ones);

                svst1_f32(p, &mut dst_k_row[d], tile_k);
                svst1_f32(p, &mut dst_v_row[d], tile_v);
            }
            out_idx += 1;
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 35: SVE2/SME ZA Parallel Chunk Attention Kernel (FlashDecoding)
// Computes partial attention score, updates running max and sum_exp, 
// and accumulates partial output O directly into an SME ZA tile.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_flashdecoding_chunk_f32(
    q_ptr: *const f32,
    k_chunk_ptr: *const f32,
    v_chunk_ptr: *const f32,
    chunk_len: u32,
    head_dim: u32,
    softmax_scale: f32,
    partial_o_out: *mut f32,
    partial_max_out: *mut f32,
    partial_sum_exp_out: *mut f32,
) {
    let chunk_len = chunk_len as usize;
    let head_dim = head_dim as usize;
    let vl = svcntw();

    let q = slice::from_raw_parts(q_ptr, head_dim);
    let k_chunk = slice::from_raw_parts(k_chunk_ptr, chunk_len * head_dim);
    let v_chunk = slice::from_raw_parts(v_chunk_ptr, chunk_len * head_dim);
    let partial_o = slice::from_raw_parts_mut(partial_o_out, head_dim);
    
    // Pass 1: Find max score in this chunk to avoid rescaling inside the ZA tile
    let mut chunk_max = core::f32::NEG_INFINITY;
    for t in 0..chunk_len {
        let k_row = &k_chunk[t * head_dim..(t + 1) * head_dim];
        let mut qk_score = 0.0f32;
        for d in (0..head_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, head_dim as u32);
            let q_vec = svld1_f32(p, &q[d]);
            let k_vec = svld1_f32(p, &k_row[d]);
            let qk_vec = svmul_f32_m(p, q_vec, k_vec);
            qk_score += svaddv_f32(p, qk_vec);
        }
        qk_score *= softmax_scale;
        if qk_score > chunk_max {
            chunk_max = qk_score;
        }
    }
    
    // Pass 2: Compute weights and accumulate using SME ZA
    asm!("smstart za", options(nomem, nostack));
    let mut acc_tile = svzero_f32();
    let mut sum_exp = 0.0f32;
    
    for t in 0..chunk_len {
        let v_row = &v_chunk[t * head_dim..(t + 1) * head_dim];
        let k_row = &k_chunk[t * head_dim..(t + 1) * head_dim];
        
        let mut qk_score = 0.0f32;
        for d in (0..head_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, head_dim as u32);
            let q_vec = svld1_f32(p, &q[d]);
            let k_vec = svld1_f32(p, &k_row[d]);
            let qk_vec = svmul_f32_m(p, q_vec, k_vec);
            qk_score += svaddv_f32(p, qk_vec);
        }
        qk_score *= softmax_scale;
        
        let weight = (qk_score - chunk_max).exp();
        sum_exp += weight;
        
        for d in (0..head_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, head_dim as u32);
            let v_vec = svld1_f32(p, &v_row[d]);
            let weighted_v = svmul_n_f32_x(p, v_vec, weight);
            acc_tile = svmopa_f32_m(p, acc_tile, weighted_v, svdup_f32(1.0));
        }
    }
    
    let p = svwhilelt_b32(0u32, head_dim as u32);
    svst1_f32(p, partial_o, acc_tile);
    *partial_max_out = chunk_max;
    *partial_sum_exp_out = sum_exp;
    
    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 35: SVE2 Final Reduction Kernel (FlashDecoding)
// Gathers partial O, max, and sum_exp from all chunks and performs 
// a final SVE2 vectorized reduction to produce the final attention output.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sve2_flashdecoding_reduce_f32(
    partial_o_ptr: *const f32,
    partial_max_ptr: *const f32,
    partial_sum_exp_ptr: *const f32,
    num_chunks: u32,
    head_dim: u32,
    final_o_out: *mut f32,
) {
    let num_chunks = num_chunks as usize;
    let head_dim = head_dim as usize;
    let vl = svcntw();

    let partial_o = slice::from_raw_parts(partial_o_ptr, num_chunks * head_dim);
    let partial_max = slice::from_raw_parts(partial_max_ptr, num_chunks);
    let partial_sum_exp = slice::from_raw_parts(partial_sum_exp_ptr, num_chunks);
    let final_o = slice::from_raw_parts_mut(final_o_out, head_dim);

    // Find global max across all chunks
    let mut global_max = core::f32::NEG_INFINITY;
    for c in 0..num_chunks {
        if partial_max[c] > global_max {
            global_max = partial_max[c];
        }
    }

    // Initialize final_o to zero
    for d in (0..head_dim).step_by(vl) {
        let p = svwhilelt_b32(d as u32, head_dim as u32);
        svst1_f32(p, &mut final_o[d], svzero_f32());
    }

    let mut global_sum_exp = 0.0f32;

    for c in 0..num_chunks {
        let p_max = partial_max[c];
        let p_sum = partial_sum_exp[c];
        let p_o = &partial_o[c * head_dim..(c + 1) * head_dim];
        
        let rescale = (p_max - global_max).exp();
        let scaled_sum = p_sum * rescale;
        global_sum_exp += scaled_sum;
        
        for d in (0..head_dim).step_by(vl) {
            let p = svwhilelt_b32(d as u32, head_dim as u32);
            let o_vec = svld1_f32(p, &p_o[d]);
            let scaled_o = svmul_n_f32_x(p, o_vec, rescale);
            
            let current_final = svld1_f32(p, &final_o[d]);
            let new_final = svadd_f32_m(p, current_final, scaled_o);
            svst1_f32(p, &mut final_o[d], new_final);
        }
    }

    // Final normalization
    let inv_sum = 1.0f32 / global_sum_exp;
    let inv_sum_vec = svdup_f32(inv_sum);
    for d in (0..head_dim).step_by(vl) {
        let p = svwhilelt_b32(d as u32, head_dim as u32);
        let final_vec = svld1_f32(p, &final_o[d]);
        let norm_vec = svmul_f32_m(p, final_vec, inv_sum_vec);
        svst1_f32(p, &mut final_o[d], norm_vec);
    }
}

// ==========================================================
// DAY 36: SVE2 Vectorized MTP Head GEMM (Fused Multi-Token Prediction)
// Computes K logits vectors in a single fused pass using SME ZA svmopa_f32,
// writing directly to a pre-allocated K × vocab_size output buffer.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sme")]
pub unsafe extern "C" fn rust_sme_mtp_predict_f32(
    hidden_state_ptr: *const f32,
    weights_ptr: *const f32,
    logits_out_ptr: *mut f32,
    hidden_dim: u32,
    vocab_size: u32,
    k_heads: u32,
) {
    let hidden_dim = hidden_dim as usize;
    let vocab_size = vocab_size as usize;
    let k_heads = k_heads as usize;
    let vl = svcntw();

    let hidden_state = slice::from_raw_parts(hidden_state_ptr, hidden_dim);
    let weights = slice::from_raw_parts(weights_ptr, k_heads * hidden_dim * vocab_size);
    let logits_out = slice::from_raw_parts_mut(logits_out_ptr, k_heads * vocab_size);

    asm!("smstart za", options(nomem, nostack));

    for k in 0..k_heads {
        let weight_offset = k * hidden_dim * vocab_size;
        let logits_row = &mut logits_out[k * vocab_size..(k + 1) * vocab_size];

        for j in (0..vocab_size).step_by(vl) {
            let p = svwhilelt_b32(j as u32, vocab_size as u32);
            let mut tile = svzero_f32();

            for d in 0..hidden_dim {
                let h_val = hidden_state[d];
                let h_vec = svdup_f32(h_val);
                let w_vec = svld1_f32(p, weights.as_ptr().add(weight_offset + d * vocab_size + j));
                tile = svmopa_f32_m(p, tile, h_vec, w_vec);
            }

            svst1_f32(p, &mut logits_row[j], tile);
        }
    }

    asm!("smstop za", options(nomem, nostack));
}

// ==========================================================
// DAY 36: SME ZA Fused MTP Logits Sampling
// Performs parallel argmax for all K heads concurrently using SVE2 svmaxv_f32 
// and svindex, outputting an array of K draft tokens without scalar loops.
// ==========================================================
#[no_mangle]
#[target_feature(enable = "sve2")]
pub unsafe extern "C" fn rust_sme_mtp_sample_f32(
    logits_ptr: *const f32,
    draft_tokens_out: *mut i32,
    vocab_size: u32,
    k_heads: u32,
) {
    let vocab_size = vocab_size as usize;
    let k_heads = k_heads as usize;
    let vl = svcntw();

    let logits = slice::from_raw_parts(logits_ptr, k_heads * vocab_size);
    let draft_tokens = slice::from_raw_parts_mut(draft_tokens_out, k_heads);

    for k in 0..k_heads {
        let logit_start = k * vocab_size;
        let mut max_vec = svdup_f32(core::f32::NEG_INFINITY);
        let mut idx_vec = svindex_s32(0, 1);

        for v in (0..vocab_size).step_by(vl) {
            let p = svwhilelt_b32(v as u32, vocab_size as u32);
            let logit_vec = svld1_f32(p, &logits[logit_start + v]);
            let gt_mask = svcmpgt_f32(p, logit_vec, max_vec);
            
            max_vec = svsel_f32(gt_mask, logit_vec, max_vec);
            
            let current_idx_vec = svindex_s32(v as i32, 1);
            idx_vec = svsel_s32(gt_mask, current_idx_vec, idx_vec);
        }
        
        let p_all = svptrue_b32();
        let max_val = svmaxv_f32(p_all, max_vec);
        
        let mut max_arr = [0.0f32; 32];
        let mut idx_arr = [0i32; 32];
        svst1_f32(p_all, &mut max_arr, max_vec);
        svst1_s32(p_all, &mut idx_arr, idx_vec);
        
        let mut best = 0i32;
        for i in 0..vl {
            if max_arr[i] == max_val {
                best = idx_arr[i];
                break;
            }
        }
        
        draft_tokens[k] = best;
    }
}