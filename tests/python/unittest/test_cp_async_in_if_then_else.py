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
"""test the correctness of inject async memory copy from an if_then_else load"""
import tvm
import numpy as np

from tvm.script import tir as T
import tvm.testing

expected_cuda_script = r"""
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(16) main_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  __shared__ float A_shared[64];
  __shared__ float B_shared[64];
  A_shared[((int)threadIdx.x)] = 0.000000e+00f;
  B_shared[((int)threadIdx.x)] = 0.000000e+00f;
__asm__ __volatile__("cp.async.commit_group;");


  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((int)threadIdx.x) + 16)))
    );
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2;"
       :: "r"(addr), "l"((void*)(A + (((int)threadIdx.x) * 14))), "n"(4)
    );
  }

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((int)threadIdx.x) + 16)))
    );
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2;"
       :: "r"(addr), "l"((void*)(B + (((int)threadIdx.x) * 14))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");


  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((int)threadIdx.x) + 32)))
    );
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2;"
       :: "r"(addr), "l"((void*)(A + ((((int)threadIdx.x) * 14) + 1))), "n"(4)
    );
  }

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((int)threadIdx.x) + 32)))
    );
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2;"
       :: "r"(addr), "l"((void*)(B + ((((int)threadIdx.x) * 14) + 1))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int i = 0; i < 13; ++i) {
    bool cse_var_1 = (i < 12);

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((i + 3) & 3) * 16) + ((int)threadIdx.x))))
    );
    int src_bytes = cse_var_1 ? 4 : 0;
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2, %3;"
       :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.x) * 14) + i) + 2))), "n"(4), "r"(src_bytes)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 5;");

    __syncthreads();
    C[((((int)threadIdx.x) * 16) + i)] = (A_shared[(((i & 3) * 16) + ((int)threadIdx.x))] + B_shared[(((i & 3) * 16) + ((int)threadIdx.x))]);
    __syncthreads();

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((i + 3) & 3) * 16) + ((int)threadIdx.x))))
    );
    int src_bytes = cse_var_1 ? 4 : 0;
    __asm__ __volatile__(
      "cp.async.ca.shared.global [%0], [%1], %2, %3;"
       :: "r"(addr), "l"((void*)(B + (((((int)threadIdx.x) * 14) + i) + 2))), "n"(4), "r"(src_bytes)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  C[((((int)threadIdx.x) * 16) + 13)] = (A_shared[(((int)threadIdx.x) + 16)] + B_shared[(((int)threadIdx.x) + 16)]);
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  C[((((int)threadIdx.x) * 16) + 14)] = (A_shared[(((int)threadIdx.x) + 32)] + B_shared[(((int)threadIdx.x) + 32)]);
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  C[((((int)threadIdx.x) * 16) + 15)] = (A_shared[(((int)threadIdx.x) + 48)] + B_shared[(((int)threadIdx.x) + 48)]);
}

"""


generated_code = ""
support_async = True


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    global generated_code
    global support_async
    generated_code = code
    # return a dummy code so that device < sm80 could build correctly
    if not support_async:
        ret = ""
        for line in code.split("\n"):
            ret += line + "\n"
            if line.startswith('extern "C" __global__'):
                break
        ret += "}"
        return ret
    return code


@tvm.testing.requires_cuda
def test_cp_async_in_if_then_else():
    global support_async
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    if major < 8:
        # At least sm80 is required
        support_async = False

    @T.prim_func
    def simple_compute(
        A: T.Buffer((16, 14), "float32"),
        B: T.Buffer((16, 14), "float32"),
        C: T.Buffer((16, 16), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                16,
                annotations={
                    "software_pipeline_stage": [0, 0, 3],
                    "software_pipeline_order": [0, 2, 1],
                    "software_pipeline_async_stages": [0],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i])
                    T.writes(C[tx, i])
                    A_shared = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                    B_shared = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = T.if_then_else(
                            1 <= i and i < 15, A[tx, i - 1], T.float32(0), dtype="float32"
                        )
                    with T.block():
                        T.reads(B[tx, i])
                        T.writes(B_shared[tx, 0])
                        B_shared[tx, 0] = T.if_then_else(
                            1 <= i and i < 15, B[tx, i - 1], T.float32(0), dtype="float32"
                        )
                    with T.block():
                        T.reads(A_shared[tx, 0], B_shared[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm.IRModule.from_expr(simple_compute)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        tvm.build(mod, target="cuda")

    assert generated_code == expected_cuda_script

    if not support_async:
        # avoid return dummy code to other tests
        support_async = True


if __name__ == "__main__":
    test_cp_async_in_if_then_else()
