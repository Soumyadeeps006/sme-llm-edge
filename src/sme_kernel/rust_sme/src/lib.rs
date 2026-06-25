#![no_std]
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Computes RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
///
/// # Safety
/// All pointer arguments must be valid, non-null, and point to allocated memory
/// containing the appropriate number of elements.
#[no_mangle]
pub unsafe extern "C" fn rmsnorm_f32(
    out: *mut f32,
    x: *const f32,
    weight: *const f32,
    size: libc::c_int,
    eps: f32,
) {
    let size = size as usize;
    let x_slice = core::slice::from_raw_parts(x, size);
    let w_slice = core::slice::from_raw_parts(weight, size);
    let out_slice = core::slice::from_raw_parts_mut(out, size);

    // Calculate sum of squares
    let mut sum_sq = 0.0f32;
    for &val in x_slice.iter() {
        sum_sq += val * val;
    }

    let mean = sum_sq / (size as f32);
    let rsqrt = 1.0f32 / (mean + eps).sqrt();

    for i in 0..size {
        out_slice[i] = x_slice[i] * rsqrt * w_slice[i];
    }
}

/// Computes Softmax inplace: x = exp(x - max) / sum(exp(x - max))
///
/// # Safety
/// The pointer argument must be valid, non-null, and point to size elements.
#[no_mangle]
pub unsafe extern "C" fn softmax_f32(x: *mut f32, size: libc::c_int) {
    let size = size as usize;
    let x_slice = core::slice::from_raw_parts_mut(x, size);

    if size == 0 {
        return;
    }

    // Find max value for numerical stability
    let mut max_val = x_slice[0];
    for &val in x_slice.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    // Compute exponentials and sum
    let mut sum = 0.0f32;
    for val in x_slice.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    let inv_sum = 1.0f32 / sum;
    for val in x_slice.iter_mut() {
        *val *= inv_sum;
    }
}
