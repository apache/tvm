# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
