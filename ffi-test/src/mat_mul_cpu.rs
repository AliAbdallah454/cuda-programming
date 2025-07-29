pub fn cpu_mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");
    
    let mut c = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    c
}

/// Cache-optimized matrix multiplication on CPU
/// 
/// Uses loop tiling/blocking to improve cache performance
/// for larger matrices.
/// 
/// # Arguments
/// * `a` - Matrix A in row-major order
/// * `b` - Matrix B in row-major order
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A and rows in B
/// * `n` - Number of columns in B and C
/// * `block_size` - Tile size for cache blocking
/// 
/// # Returns
/// Result matrix C in row-major order
pub fn cpu_mat_mul_blocked(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, block_size: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");
    
    let mut c = vec![0.0f32; m * n];
    
    // Blocked matrix multiplication for better cache locality
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                // Process block
                let i_end = std::cmp::min(ii + block_size, m);
                let j_end = std::cmp::min(jj + block_size, n);
                let k_end = std::cmp::min(kk + block_size, k);
                
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i * n + j]; // Accumulate to existing value
                        for l in kk..k_end {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
    
    c
}