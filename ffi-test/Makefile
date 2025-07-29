# CUDA Rust FFI Makefile
# Requires: CUDA toolkit, Rust toolchain

.PHONY: all build run test clean check setup help

# Default target
all: build

# Setup environment and check dependencies
setup:
	@echo "🔧 Checking dependencies..."
	@which nvcc > /dev/null || (echo "❌ CUDA compiler (nvcc) not found. Please install CUDA toolkit." && exit 1)
	@which cargo > /dev/null || (echo "❌ Rust (cargo) not found. Please install Rust toolchain." && exit 1)
	@echo "✅ All dependencies found"
	@echo "CUDA version:"
	@nvcc --version | head -4
	@echo "Rust version:"
	@rustc --version

# Build the project
build: setup
	@echo "🏗️  Building CUDA Rust FFI project..."
	cargo build --release

# Build in debug mode
debug: setup
	@echo "🏗️  Building in debug mode..."
	cargo build

# Run the example
run: build
	@echo "🚀 Running CUDA matrix multiplication example..."
	cargo run --release

# Run in debug mode
run-debug: debug
	@echo "🚀 Running in debug mode..."
	cargo run

# Run tests
test: build
	@echo "🧪 Running tests..."
	cargo test --release

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -f *.o *.a libmat_mul_rust_ffi_kernel.a mat_mul_rust_ffi_kernel.o

# Check code without building
check:
	@echo "✅ Checking code..."
	cargo check

# Format code
fmt:
	@echo "🎨 Formatting code..."
	cargo fmt

# Run clippy lints
clippy:
	@echo "📎 Running Clippy lints..."
	cargo clippy -- -D warnings

# Full check (format, clippy, test)
ci: fmt clippy test
	@echo "✅ All CI checks passed!"

# Install CUDA (Ubuntu/Debian)
install-cuda-ubuntu:
	@echo "📦 Installing CUDA on Ubuntu/Debian..."
	@echo "This will add NVIDIA package repository and install CUDA toolkit"
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
	sudo dpkg -i cuda-keyring_1.0-1_all.deb
	sudo apt-get update
	sudo apt-get -y install cuda-toolkit-12-2
	@echo "⚠️  You may need to add CUDA to your PATH:"
	@echo "export PATH=/usr/local/cuda/bin:\$$PATH"
	@echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$$LD_LIBRARY_PATH"

# Show help
help:
	@echo "CUDA Rust FFI Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Check dependencies and show versions"
	@echo "  build          - Build the project in release mode"
	@echo "  debug          - Build the project in debug mode"
	@echo "  run            - Build and run the example (release)"
	@echo "  run-debug      - Build and run the example (debug)"
	@echo "  test           - Run all tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  check          - Check code without building"
	@echo "  fmt            - Format code with rustfmt"
	@echo "  clippy         - Run clippy lints"
	@echo "  ci             - Run full CI pipeline (fmt, clippy, test)"
	@echo "  install-cuda-ubuntu - Install CUDA toolkit on Ubuntu/Debian"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - CUDA Toolkit (nvcc compiler)"
	@echo "  - Rust toolchain (cargo, rustc)"
	@echo "  - Compatible NVIDIA GPU for running"