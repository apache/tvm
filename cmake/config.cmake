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
# whether enable CUDA during compile
set(USE_CUDA OFF)

# ROCM
set(USE_ROCM OFF)
set(ROCM_PATH "/opt/rocm")

# Whether enable OpenCL runtime
set(USE_OPENCL OFF)

# Whether enable Metal runtime
set(USE_METAL OFF)

# Whether enable Vulkan runtime
set(USE_VULKAN OFF)

# Whether enable OpenGL runtime
set(USE_OPENGL OFF)

# Whether enable RPC runtime
set(USE_RPC ON)

# Whether enable tiny embedded graph runtime.
set(USE_GRAPH_RUNTIME ON)

# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG OFF)

# Whether build with LLVM support
# Requires LLVM version >= 4.0
#
# Possible values:
# - ON: enable llvm with cmake's find llvm
# - OFF: disbale llvm
# - /path/to/llvm-config enable specific LLVM when multiple llvm-dev is available.
set(USE_LLVM OFF)

#---------------------------------------------
# Contrib libraries
#---------------------------------------------
# Whether use BLAS, choices: openblas, atlas, blas, apple
set(USE_BLAS none)

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
