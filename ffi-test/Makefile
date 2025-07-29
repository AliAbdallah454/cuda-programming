# CUDA Rust FFI Makefile
# Requires: CUDA toolkit, Rust toolchain

.PHONY: all build run test clean check setup help

all: build

build: setup
	@echo " Building CUDA Rust FFI project..."
	cargo build --release

debug: setup
	@echo " Building in debug mode..."
	cargo build

run: build
	@echo "Running CUDA matrix multiplication example..."
	cargo run --release

run-debug: debug
	@echo "Running in debug mode..."
	cargo run

test: build
	@echo "Running tests..."
	cargo test --release

clean:
	@echo "Cleaning build artifacts..."
	cargo clean

check:
	@echo "Checking code..."
	cargo check

fmt:
	@echo "Formatting code..."
	cargo fmt

clippy:
	@echo "Running Clippy lints..."
	cargo clippy -- -D warnings

ci: fmt clippy test
	@echo "All CI checks passed!"