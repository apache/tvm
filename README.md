<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

TVM.TL
==============================================
**Not the official and Final Release, but you can use tl via [BitBLAS](https://github.com/microsoft/BitBLAS)**

TVM.TL is an extention of TVMScript to write simple and high performance GPU kernels with tensorcores. TVM.TL is currently supported on CUDA deivces with Ampere (sm_80+), Turing (sm_75) and Volta(sm_70).

Let's get started with a simple GEMM example.
```python
import tvm.tl.language as T
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype = "float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
        bias: T.Buffer([N], dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            bias_local = T.alloc_fragment((block_N,), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(bias[bx * block_N], bias_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias_local[j]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```
Despite this simple examples, tvm.tl can be used to write more complicated examples including convolutions, flash-attention-v2 (fwd & bwd), normalizations, these examples can be found under folder tl_scripts.

 The performance of our flash-attention is comparable to the manually implementation. (see [Link](https://github.com/nox-410/tvm.tl/blob/tl/tl_doc/flash_perf.md)).

## Install

Install is similar to tvm. First, fill in USE_CUDA and USE_LLVM in cmake/config.cmake, like this:
```bash
set(USE_LLVM "/path/to/llvm-config --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CUDA /usr/local/cuda)
```
Then build tvm
```bash
mkdir -p build && cd build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -
export PYTHONPATH="$PYTHONPATH:$PWD/python"
# some python package required by tvm
pip install torch attrs cloudpickle decorator psutil synr tornado xgboost
```
We also need to prepare the cutlass headers, the default version of cutlass in TVM does not work correctly
```bash
git clone https://github.com/NVIDIA/cutlass.git -b v3.2.2
export TL_CUTLASS_PATH=/path/to/cutlass/include
```
Note 1: It is recommeneded to use the latest cuda toolkit, because we requires nvcc to jit compile the generated CUDA code.

Note 2: Don't forget to clone the submodules.

## Language reference
Still in progress.

See tl_doc/language_ref.md
