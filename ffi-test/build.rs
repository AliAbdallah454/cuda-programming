use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {

    println!("cargo:rerun-if-changed=cuda/mat_mul_kernel.cu");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);
    
    // Compile .cu to .o
    let obj_file = out_path.join("mat_mul_kernel.o");
    let status = Command::new("nvcc")
        .args(&[
            "-c", "cuda/mat_mul_kernel.cu",
            "-o", obj_file.to_str().unwrap(),
            "-Xcompiler", "-fPIC",
        ])
        .status()
        .expect("Failed to run nvcc. Make sure CUDA is installed and nvcc is in PATH");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

    // Archive the .o into a .a (static library)
    let lib_file = out_path.join("libmat_mul_cuda.a");
    let status = Command::new("ar")
        .args(&["rcs", lib_file.to_str().unwrap(), obj_file.to_str().unwrap()])
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar archiving failed");
    }

    // Tell Cargo where to find the library and link it
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=mat_mul_cuda");
    
    // Link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
