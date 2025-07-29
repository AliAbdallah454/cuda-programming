use libc::c_int;

#[link(name = "mat_mul_cuda", kind = "static")]
extern "C" {
    /// CUDA matrix multiplication function
    /// 
    /// # Arguments
    /// * `a` - Pointer to matrix A (m x k)
    /// * `b` - Pointer to matrix B (k x n) 
    /// * `c` - Pointer to result matrix C (m x n)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    pub fn launch_mat_mul(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

/// Safe wrapper for CUDA matrix multiplication
pub fn cuda_mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");
    
    let mut c = vec![0.0f32; m * n];
    
    unsafe {
        launch_mat_mul(
            a.as_ptr() as *mut f32,
            b.as_ptr() as *mut f32,
            c.as_mut_ptr(),
            m as c_int,
            k as c_int,
            n as c_int,
        );
    }
    
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_multiplication() {
        let a = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        
        let b = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        
        let result = cuda_mat_mul(&a, &b, 4, 4, 4);
        
        // Result should be the same as matrix A (A * I = A)
        for i in 0..16 {
            assert!((result[i] - a[i]).abs() < 1e-6, 
                    "Mismatch at index {}: expected {}, got {}", i, a[i], result[i]);
        }
    }
}