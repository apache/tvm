MODEL ?= resnet
NUM_THREADS ?= 4
BATCH_SIZE ?= 64
TRAINING ?= true
DEBUG ?= false

build_dir := ../build

ifeq ($(DEBUG), false)
	debug := release
	xargo_args := --release
else
	debug := debug
endif

target/x86_64-unknown-linux-sgx/$(debug)/libmodel-enclave.a: $(build_dir)/libmodel.a **/*
	RUST_TARGET_PATH=$(shell pwd) \
		RUST_TARGET_DIR=$(shell pwd)/target \
		RUSTFLAGS="-Z force-unstable-if-unmarked" \
		TVM_NUM_THREADS=$(NUM_THREADS) \
		BUILD_DIR=../build \
		xargo build --target x86_64-unknown-linux-sgx $(xargo_args) -q

$(build_dir)/libmodel.a: $(build_dir)/model.o
	llvm-ar cr $@ $^

$(build_dir)/model.o: $(build_dir)/model.bc
	clang -c $< -o $@ -fPIC -O3
	objcopy --globalize-symbol __tvm_module_startup $@

$(build_dir)/model.bc: src/build_model.py
	python3 $< -o $(build_dir)

clean:
	xargo clean
