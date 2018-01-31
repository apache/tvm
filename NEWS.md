TVM Change Log
==============

This file records the changes in TVM library in reverse chronological order.

## On-going version

Refer to the Roadmap issue for complete list on on-going version features.
If you check in something that is not reflected in Roadmap issue, please reply
to that issue so it can get added.

## 0.2

This release comes with a complete set of TOPI support for NNVM compiler, which allows compilation of end to end workloads.
We also make major improvements in supporting new backends: ROCm for AMDGPUs and ARM GPU.

- Backend support
   - Support LLVM mainline(4.0, 5.0, 6.0)
   - Support ROCM stack for AMD GPUs
   - More robust OpenCL support for ARM GPUs
- Android RPC runtime
- Multi-threading optimization for ARM
   - multi-threaded depthwise
   - multi-threaded conv2d
- New schedule primitives
   - storage_align for shared memory alignment
   - double_buffer
- UnrollLoop : more robust version of unroll loop, count maximum steps that can be unrolled.
- Full set of TOPI operators
   - Introduce tvm.target to specify target options for compilation better.
   - broadcast/ reduction operators
   - pooling and global pooling
   - Generic target support for topi
   - schedule with external libraries
- End to end deep learning pipelines for CPU, GPU, ARM GPU
- Tutorials
  - How to load compiled module in any language runtime
  -  How to use java runtime
- Contrib library: MIOpen, CuDNN
- Ongoing items that contains functioning pieces
  - WebGL backend
  - C++ compiler support
  - MPS DNN
  - low bit support, introduced popcount


## 0.1

- Language runtime
    - python
    - javascript
    - java
    - c++
- Backend
    - arm, x86
    - javascript, wasm
    - CUDA
    - opencl
    - Metal
- DNN Library integration
- RPC  runtime
- TOPI operator pipeline python
- TOPI operator pipeline in C++
- Rough perf of the TOPI GPU pipeline
- Rough pref of TOPI CPU pipeline
- End to end graph executors


## Initial version

- Pack libary into shared library.
- External function and contrib libraries
- DLPack integration support
- AOT and module system
- Basic code structure ready.
