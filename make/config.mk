#-------------------------------------------------------------------------------
#  Template configuration for compiling
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

# whether compile with debug
DEBUG = 0

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------
# whether enable CUDA during compile
USE_CUDA = 0

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable.
# CUDA_PATH = /usr/local/cuda

# ROCM
USE_ROCM = 0

# whether enable OpenCL during compile
USE_OPENCL = 0

# whether enable Metal during compile
USE_METAL = 0

# whether enable SGX during compile
USE_SGX = 0
SGX_SDK = /opt/sgxsdk

# Whether enable RPC during compile
USE_RPC = 1

# Whether enable tiny embedded graph runtime.
USE_GRAPH_RUNTIME = 1

#  Whether enable additional graph debug functions
USE_GRAPH_RUNTIME_DEBUG = 0

# whether build with LLVM support
# Requires LLVM version >= 4.0
# Set LLVM_CONFIG to your version, uncomment to build with llvm support
#
# LLVM_CONFIG = llvm-config

#---------------------------------------------
# Contrib optional libraries.
#---------------------------------------------
# Whether use BLAS, choices: openblas, atlas, blas, apple
USE_BLAS = none

# Whether use contrib.random in runtime
USE_RANDOM = 0

# Whether use NNPack
USE_NNPACK = 0
# NNPACK_PATH = none

# Whether use CuDNN
USE_CUDNN = 0

# Whether use MIOpen
USE_MIOPEN = 0

# Whether use MPS
USE_MPS = 0

# Whether use cuBLAS
USE_CUBLAS = 0

# Whether use rocBlas
USE_ROCBLAS = 0
