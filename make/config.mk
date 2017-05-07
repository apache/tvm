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
USE_CUDA = 1

# whether enable OpenCL during compile
USE_OPENCL = 0

# whether enable Metal during compile
USE_METAL = 0

# whether build with LLVM support
# Requires LLVM version >= 4.0
# Set LLVM_CONFIG to your version
# LLVM_CONFIG = llvm-config-4.0
USE_LLVM = 0

#---------------------------------------------
# Contrib optional libraries.
#---------------------------------------------
# Whether use BLAS, choices: openblas, atlas, blas, apple
USE_BLAS = none

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable.
# CUDA_PATH = /usr/local/cuda
