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

#---------------------
# choice of compiler
#--------------------
export NVCC = nvcc

# whether compile with debug
DEBUG = 0

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = 1

# whether use OpenCL during compile
USE_OPENCL = 0

# whether build with LLVM support
# This requires llvm-config to be in your PATH
# Requires LLVM version >= 4.0
USE_LLVM = 0

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable.
# CUDA_PATH = /usr/local/cuda
