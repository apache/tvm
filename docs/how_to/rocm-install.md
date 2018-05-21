Installation Guide for ROCM
==================
This page contains instructions about building TVM for ROCm backend. Currently, ROCm is supported only on linux, so all the instructions are written with linux in mind.

## Depedencies
1. HIP runtime from ROCm. Make sure the installation system has ROCm installed in it.
2. Latest stable version of LLVM (v6.0.1), and LLD.

## Installation
TVM for ROCm can be built with Makefile infrastructure. 
```
git clone --recursive https://github.com/dmlc/tvm.git
cd tvm
make ROCM_PATH=/opt/rocm LLVM_CONFIG=<path to llvm-config>
```

## Testing
Inorder to test multiple python scripts without installing tvm (useful for tvm development), use the following command:
```
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:${PYTHONPATH}
```
