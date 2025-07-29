
use ffi_test::cuda_mat_mul;

fn print_matrix(name: &str, matrix: &[f32], rows: usize, cols: usize) {
    println!("{}:", name);
    for i in 0..rows {
        for j in 0..cols {
            print!("{:8.2} ", matrix[i * cols + j]);
        }
        println!();
    }
    println!();
}

fn main() {
    println!("CUDA Matrix Multiplication Test");
    println!("===============================");
    
    // Test 1: Identity matrix multiplication
    const M: usize = 4;
    const K: usize = 4;
    const N: usize = 4;

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

    print_matrix("Matrix A (4x4)", &a, M, K);
    print_matrix("Matrix B (4x4) - Identity", &b, K, N);

    let result = cuda_mat_mul(&a, &b, M, K, N);
    print_matrix("Result C = A * B", &result, M, N);
    
    // Verify result (A * I should equal A)
    let matches = a.iter().zip(result.iter()).all(|(a, r)| (a - r).abs() < 1e-6);
    println!("Identity test: {}", if matches { "✅ PASSED" } else { "❌ FAILED" });
    
    // Test 2: Simple 2x2 multiplication
    println!("\n--- Test 2: 2x2 Matrix Multiplication ---");
    let a2 = vec![1.0, 2.0, 3.0, 4.0];  // 2x2
    let b2 = vec![2.0, 0.0, 1.0, 2.0];  // 2x2
    
    print_matrix("Matrix A (2x2)", &a2, 2, 2);
    print_matrix("Matrix B (2x2)", &b2, 2, 2);
    
    let result2 = cuda_mat_mul(&a2, &b2, 2, 2, 2);
    print_matrix("Result C = A * B", &result2, 2, 2);
    
    // Expected result: [[4, 4], [10, 8]]
    let expected = vec![4.0, 4.0, 10.0, 8.0];
    let matches2 = expected.iter().zip(result2.iter()).all(|(e, r)| (e - r).abs() < 1e-6);
    println!("2x2 test: {}", if matches2 { "✅ PASSED" } else { "❌ FAILED" });
    
    println!("\nAll tests completed!");
}
