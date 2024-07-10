#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=1
export CCACHE_DIR=/ccache-tvm

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/
cp $WORKSPACE_CWD/cmake/config.cmake $WORKSPACE_CWD/build/ -f

echo set\(USE_OPENCL ON\) >> config.cmake
echo set\(USE_LLVM ON\) >> config.cmake
if [[ ${GPU} == cuda* ]]; then
    echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
    echo set\(CMAKE_CUDA_ARCHITECTURES "80;90"\) >>config.cmake
    echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
    echo set\(USE_CUDA ON\) >> config.cmake
fi
if [[ -f ${ADRENOACCL_SDK}/include/adrenoaccl.h ]]; then
    echo set\(USE_ADRENO_ACCL \"${ADRENOACCL_SDK}\"\) >> config.cmake
fi
if [[ -f ${ADRENO_OPENCL}/CL/cl.h ]]; then
    echo set\(USE_CLML \"${ADRENO_OPENCL}\"\) >> config.cmake
fi

cmake .. && make -j`nproc`
