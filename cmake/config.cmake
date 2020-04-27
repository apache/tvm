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
#  Then build in parallel with 8 threads
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
# - OFF: disable CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA OFF)

# Whether enable ROCM runtime
#
# Possible values:
# - ON: enable ROCM with cmake's auto search
# - OFF: disable ROCM
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
# - OFF: disable vulkan
# - /path/to/vulkan-sdk: use specific path to vulkan-sdk
set(USE_VULKAN OFF)

# Whether enable OpenGL runtime
set(USE_OPENGL OFF)

# Whether enable MicroTVM runtime
set(USE_MICRO OFF)

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

# Whether enable additional vm profiler functions
set(USE_VM_PROFILER OFF)

# Whether enable uTVM standalone runtime
set(USE_MICRO_STANDALONE_RUNTIME OFF)

# Whether build with LLVM support
# Requires LLVM version >= 4.0
#
# Possible values:
# - ON: enable llvm with cmake's find search
# - OFF: disable llvm
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
# set(USE_MKL_PATH <path to venv or site-packages directory>) if using `pip install mkl`
set(USE_MKL_PATH none)

# Whether use MKLDNN library, choices: ON, OFF, path to mkldnn library
set(USE_MKLDNN OFF)

# Whether use OpenMP thread pool, choices: gnu, intel
# Note: "gnu" uses gomp library, "intel" uses iomp5 library
set(USE_OPENMP none)

# Whether use contrib.random in runtime
set(USE_RANDOM OFF)

# Whether use NNPack
set(USE_NNPACK OFF)

# Possible values:
# - ON: enable tflite with cmake's find search
# - OFF: disable tflite
# - /path/to/libtensorflow-lite.a: use specific path to tensorflow lite library
set(USE_TFLITE OFF)

# /path/to/tensorflow: tensorflow root path when use tflite library
set(USE_TENSORFLOW_PATH none)

# Required for full builds with TFLite. Not needed for runtime with TFLite.
# /path/to/flatbuffers: flatbuffers root path when using tflite library
set(USE_FLATBUFFERS_PATH none)

# Possible values:
# - OFF: disable tflite support for edgetpu
# - /path/to/edgetpu: use specific path to edgetpu library
set(USE_EDGETPU OFF)

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
set(USE_SORT ON)

# Whether use MKL-DNN (DNNL) codegen
set(USE_DNNL_CODEGEN OFF)

# Build ANTLR parser for Relay text format
# Possible values:
# - ON: enable ANTLR by searching default locations (cmake find_program for antlr4 and /usr/local for jar)
# - OFF: disable ANTLR
# - /path/to/antlr-*-complete.jar: path to specific ANTLR jar file
set(USE_ANTLR OFF)

# Whether use Relay debug mode
set(USE_RELAY_DEBUG OFF)

# Whether to build fast VTA simulator driver
set(USE_VTA_FSIM OFF)

# Whether to build cycle-accurate VTA simulator driver
set(USE_VTA_TSIM OFF)

# Whether to build VTA FPGA driver (device side only)
set(USE_VTA_FPGA OFF)

# Whether to build the example external runtime module
set(USE_EXAMPLE_EXT_RUNTIME OFF)

# Whether use Thrust
set(USE_THRUST OFF)

# Whether to build the TensorFlow TVMDSOOp module
set(USE_TF_TVMDSOOP OFF)

# Whether to use hexagon device
set(USE_HEXAGON_DEVICE OFF)
set(USE_HEXAGON_SDK /path/to/sdk)
