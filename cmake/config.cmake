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
#
# Possible values:
# - ON: enable OpenCL with cmake's auto search
# - OFF: disable OpenCL
# - /path/to/opencl-sdk: use specific path to opencl-sdk
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

# Whether enable RPC runtime
set(USE_RPC ON)

# Whether to build the C++ RPC server binary
set(USE_CPP_RPC OFF)

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
# Whether to build with BYODT software emulated posit custom datatype
# 
# Possible values:
# - ON: enable BYODT posit, requires setting UNIVERSAL_PATH
# - OFF: disable BYODT posit
#
# set(UNIVERSAL_PATH /path/to/stillwater-universal) for ON
set(USE_BYODT_POSIT OFF)

# Whether use BLAS, choices: openblas, atlas, apple
set(USE_BLAS none)

# Whether to use MKL
# Possible values:
# - ON: Enable MKL
# - /path/to/mkl: mkl root path
# - OFF: Disable MKL
# set(USE_MKL /opt/intel/mkl) for UNIX
# set(USE_MKL ../IntelSWTools/compilers_and_libraries_2018/windows/mkl) for WIN32
# set(USE_MKL <path to venv or site-packages directory>) if using `pip install mkl`
set(USE_MKL OFF)

# Whether use MKLDNN library, choices: ON, OFF, path to mkldnn library
set(USE_MKLDNN OFF)

# Whether use OpenMP thread pool, choices: gnu, intel
# Note: "gnu" uses gomp library, "intel" uses iomp5 library
set(USE_OPENMP none)

# Whether use contrib.random in runtime
set(USE_RANDOM ON)

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

# Whether to use Arm Compute Library (ACL) codegen
# We provide 2 separate flags since we cannot build the ACL runtime on x86.
# This is useful for cases where you want to cross-compile a relay graph
# on x86 then run on AArch.
#
# An example of how to use this can be found here: docs/deploy/arm_compute_lib.rst.
#
# USE_ARM_COMPUTE_LIB - Support for compiling a relay graph offloading supported
#                       operators to Arm Compute Library. OFF/ON
# USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME - Run Arm Compute Library annotated functions via the ACL
#                                     runtime. OFF/ON/"path/to/ACL"
set(USE_ARM_COMPUTE_LIB OFF)
set(USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME OFF)

# Whether to build with Arm Ethos-N support
# Possible values:
# - OFF: disable Arm Ethos-N support
# - path/to/arm-ethos-N-stack: use a specific version of the
#   Ethos-N driver stack
set(USE_ETHOSN OFF)
# If USE_ETHOSN is enabled, use ETHOSN_HW (ON) if Ethos-N hardware is available on this machine
# otherwise use ETHOSN_HW (OFF) to use the software test infrastructure
set(USE_ETHOSN_HW OFF)

# Whether to build with TensorRT codegen or runtime
# Examples are available here: docs/deploy/tensorrt.rst.
#
# USE_TENSORRT_CODEGEN - Support for compiling a relay graph where supported operators are
#                        offloaded to TensorRT. OFF/ON
# USE_TENSORRT_RUNTIME - Support for running TensorRT compiled modules, requires presense of
#                        TensorRT library. OFF/ON/"path/to/TensorRT"
set(USE_TENSORRT_CODEGEN OFF)
set(USE_TENSORRT_RUNTIME OFF)

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

# Whether use Thrust
set(USE_THRUST OFF)

# Whether to build the TensorFlow TVMDSOOp module
set(USE_TF_TVMDSOOP OFF)

# Whether to use STL's std::unordered_map or TVM's POD compatible Map
set(USE_FALLBACK_STL_MAP OFF)

# Whether to use hexagon device
set(USE_HEXAGON_DEVICE OFF)
set(USE_HEXAGON_SDK /path/to/sdk)

# Whether to use ONNX codegen
set(USE_TARGET_ONNX OFF)

# Whether to compile the standalone C runtime.
set(USE_STANDALONE_CRT ON)

