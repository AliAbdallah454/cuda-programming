use libc::c_int;

pub mod mat_mul_cpu;

pub use mat_mul_cpu::cpu_mat_mul;

#[link(name = "mat_mul_cuda", kind = "static")]
extern "C" {
    pub fn launch_mat_mul(
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

pub fn cuda_mat_mul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");
    
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
}