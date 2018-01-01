#-------------------------------------------------------------------------------
#  Template configuration for compiling nnvm
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of nnvm. First copy the this
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

# choice of archiver
export AR = ar

# the additional link flags you want to add
ADD_LDFLAGS=

# the additional compile flags you want to add
ADD_CFLAGS=

# path to dmlc-core module 
#DMLC_CORE_PATH=

#----------------------------
# plugins
#----------------------------

# whether to use fusion integration. This requires installing cuda.
# ifndef CUDA_PATH
# 	CUDA_PATH = /usr/local/cuda
# endif
# NNVM_FUSION_PATH = plugin/nnvm-fusion
# NNVM_PLUGINS += $(NNVM_FUSION_PATH)/nnvm-fusion.mk
