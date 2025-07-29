use ffi_test::cuda_mat_mul;
use ffi_test::mat_mul_cpu::cpu_mat_mul;

use std::time::Instant;

fn create_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    let mut matrix = Vec::with_capacity(rows * cols);
    
    for _ in 0..rows * cols {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        matrix.push((rng as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    
    matrix
}

fn main() {

    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 256;
    
    let matrix_a = create_random_matrix(M, K, 12345);
    let matrix_b = create_random_matrix(K, N, 67890);
    let mut c = vec![0.0f32; M * N];

    // Warming GPU up
    let matrix_a_warm_up = create_random_matrix(M, K, 123454);
    let matrix_b_warm_up = create_random_matrix(K, N, 67809);
    let mut c_warm_up = vec![0.0f32; M * N];
    cuda_mat_mul(&matrix_a_warm_up, &matrix_b_warm_up, &mut c_warm_up, M, K, N);

    let start = Instant::now();
    let _ = cpu_mat_mul(&matrix_a, &matrix_b, M, K, N);
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;
    
    let start = Instant::now();
    cuda_mat_mul(&matrix_a, &matrix_b, &mut c, M, K, N);
    let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("CPU time: {:.2} ms", cpu_time);
    println!("GPU time: {:.2} ms", gpu_time);

    let speedup = cpu_time / gpu_time;
    println!("GPU speed up: {:.2}x", speedup);
}
