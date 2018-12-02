#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disbale CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA OFF)

# Whether enable ROCM runtime
#
# Possible values:
# - ON: enable ROCM with cmake's auto search
# - OFF: disbale ROCM
# - /path/to/rocm: use specific path to rocm
set(USE_ROCM OFF)

# Whether enable SDAccel runtime
set(USE_SDACCEL OFF)

# Whether enable Intel FPGA SDK for OpenCL (AOCL) runtime
set(USE_AOCL OFF)

# Whether enable OpenCL runtime
set(USE_OPENCL OFF)

# Whether enable Metal runtime
set(USE_METAL OFF)

# Whether enable Vulkan runtime
#
# Possible values:
# - ON: enable Vulkan with cmake's auto search
# - OFF: disbale vulkan
# - /path/to/vulkan-sdk: use specific path to vulkan-sdk
set(USE_VULKAN OFF)

# Whether enable OpenGL runtime
set(USE_OPENGL OFF)

# Whether to enable SGX runtime
#
# Possible values for USE_SGX:
# - /path/to/sgxsdk: path to Intel SGX SDK
# - OFF: disable SGX
#
# SGX_MODE := HW|SIM
set(USE_SGX OFF)
set(SGX_MODE "SIM")
set(RUST_SGX_SDK "/path/to/rust-sgx-sdk")

# Whether enable RPC runtime
set(USE_RPC ON)

# Whether embed stackvm into the runtime
set(USE_STACKVM_RUNTIME OFF)

# Whether enable tiny embedded graph runtime.
set(USE_GRAPH_RUNTIME ON)

# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG OFF)

# Whether build with LLVM support
# Requires LLVM version >= 4.0
#
# Possible values:
# - ON: enable llvm with cmake's find search
# - OFF: disbale llvm
# - /path/to/llvm-config: enable specific LLVM when multiple llvm-dev is available.
set(USE_LLVM OFF)

#---------------------------------------------
# Contrib libraries
#---------------------------------------------
# Whether use BLAS, choices: openblas, mkl, atlas, apple
set(USE_BLAS none)

# /path/to/mkl: mkl root path when use mkl blas library
# set(USE_MKL_PATH /opt/intel/mkl) for UNIX
# set(USE_MKL_PATH ../IntelSWTools/compilers_and_libraries_2018/windows/mkl) for WIN32
set(USE_MKL_PATH none)

# Whether use contrib.random in runtime
set(USE_RANDOM OFF)

# Whether use NNPack
set(USE_NNPACK OFF)

# Whether use CuDNN
set(USE_CUDNN OFF)

# Whether use cuBLAS
set(USE_CUBLAS OFF)

# Whether use MIOpen
set(USE_MIOPEN OFF)

# Whether use MPS
set(USE_MPS OFF)

# Whether use rocBlas
set(USE_ROCBLAS OFF)

# Whether use contrib sort
set(USE_SORT OFF)

# Build ANTLR parser for Relay text format
set(USE_ANTLR OFF)
