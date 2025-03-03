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

import tvm
import tvm.testing
from tvm.script import tir as T

import pytest
import numpy as np


def count_cp_async(stmt):
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Call) and n.op.name == "tir.ptx_cp_async":
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    return num_alloc[0]


def generate_global_to_shared_vectorized_copy(dtype, vector_size):
    num_iters = 128 // vector_size
    vector_size_expr = tvm.runtime.convert(vector_size)

    @T.prim_func
    def ptx_global_to_shared_copy(
        A: T.Buffer((32, 128), dtype), B: T.Buffer((32, 128), dtype)
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        bx = T.env_thread("blockIdx.x")
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(bx, 1)
        T.launch_thread(tx, 32)
        with T.block():
            A_shared = T.alloc_buffer([32, 128], dtype, scope="shared")
            T.reads(A[0:32, 0:128])
            T.writes(B[0:32, 0:128])

            T.attr("default", "async_scope", 1)
            for i in T.serial(num_iters):
                for j in T.vectorized(vector_size):
                    A_shared[tx, i * vector_size_expr + j] = A[tx, i * vector_size_expr + j]

            T.evaluate(T.ptx_commit_group(dtype=""))
            T.evaluate(T.ptx_wait_group(0, dtype=""))

            for i in range(128):
                B[tx, i] = A_shared[tx, i]

    return ptx_global_to_shared_copy


@T.prim_func
def ptx_global_to_shared_copy_fp32x1(
    A: T.Buffer((32, 128), "float32"), B: T.Buffer((32, 128), "float32")
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float32", scope="shared")
        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        T.attr("default", "async_scope", 1)
        for i in T.serial(128):
            A_shared[tx, i] = A[tx, i]

        T.evaluate(T.ptx_commit_group(dtype=""))
        T.evaluate(T.ptx_wait_group(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@T.prim_func
def ptx_global_to_shared_dyn_copy_fp16x8(
    A: T.Buffer((32, 128), "float16"),
    B: T.Buffer((32, 128), "float16"),
    C: T.Buffer((32, 128), "float16"),
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer([32, 128], "float16", scope="shared.dyn")
        T.reads(A[0:32, 0:128], B[0:32, 0:128])
        T.writes(C[0:32, 0:128])

        T.attr("default", "async_scope", 1)
        for i in T.serial(16):
            for j in T.vectorized(8):
                A_shared[tx, i * 8 + j] = A[tx, i * 8 + j]
                B_shared[tx, i * 8 + j] = B[tx, i * 8 + j]

        T.evaluate(T.ptx_commit_group(dtype=""))
        T.evaluate(T.ptx_wait_group(0, dtype=""))

        for i in range(128):
            C[tx, i] = A_shared[tx, i] + B_shared[tx, i]


@tvm.testing.requires_cuda
def test_inject_async_copy():
    for dtype, vec_size in [("float16", 8), ("float16", 4), ("float32", 4), ("float32", 1)]:
        if vec_size == 1:
            f = ptx_global_to_shared_copy_fp32x1
        else:
            f = generate_global_to_shared_vectorized_copy(dtype, vec_size)

        mod = tvm.IRModule.from_expr(f)
        mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
        mod = tvm.tir.transform.FlattenBuffer()(mod)
        if vec_size > 1:
            mod = tvm.tir.transform.VectorizeLoop()(mod)
        mod = tvm.tir.transform.InjectPTXAsyncCopy()(mod)

        assert count_cp_async(mod["main"].body) == 1

        if not tvm.testing.is_ampere_or_newer():
            continue

        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            mod = tvm.build(tvm.IRModule.from_expr(f), target="cuda")

        A_np = np.random.rand(32, 128).astype(dtype)
        B_np = np.zeros((32, 128)).astype(dtype)
        dev = tvm.cuda(0)
        A_nd = tvm.nd.array(A_np, device=dev)
        B_nd = tvm.nd.array(B_np, device=dev)
        mod(A_nd, B_nd)
        tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@tvm.testing.requires_cuda
def test_inject_async_copy_shared_dyn():
    f = ptx_global_to_shared_dyn_copy_fp16x8

    mod = tvm.IRModule.from_expr(f)
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.VectorizeLoop()(mod)
    mod = tvm.tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tvm.tir.transform.InjectPTXAsyncCopy()(mod)

    assert count_cp_async(mod["main"].body) == 2

    if not tvm.testing.is_ampere_or_newer():
        return

    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        mod = tvm.build(tvm.IRModule.from_expr(f), target="cuda")

    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.random.rand(32, 128).astype("float16")
    C_np = np.zeros((32, 128)).astype("float16")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    C_nd = tvm.nd.array(C_np, device=dev)
    mod(A_nd, B_nd, C_nd)
    tvm.testing.assert_allclose(C_nd.numpy(), A_np + B_np)


@T.prim_func
def ptx_global_to_shared_copy_fp32x1_barrier(
    A: T.Buffer((32, 128), "float32"), B: T.Buffer((32, 128), "float32")
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float32", scope="shared")

        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        T.evaluate(T.create_barriers(1, dtype=""))
        T.evaluate(T.ptx_init_barrier_thread_count(0, 32, dtype=""))

        T.attr("default", "async_scope", 1)
        for i in T.serial(128):
            A_shared[tx, i] = A[tx, i]

        T.evaluate(T.ptx_cp_async_barrier(0, dtype=""))
        T.evaluate(T.ptx_arrive_barrier(0, dtype=""))
        T.evaluate(T.ptx_wait_barrier(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@tvm.testing.requires_cuda
def test_inject_async_copy_barrier():
    dtype = "float32"
    vec_size = 1
    f = ptx_global_to_shared_copy_fp32x1_barrier

    mod = tvm.IRModule.from_expr(f)
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.InjectPTXAsyncCopy()(mod)

    assert count_cp_async(mod["main"].body) == 1

    if tvm.testing.is_ampere_or_newer():
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            mod = tvm.build(tvm.IRModule.from_expr(f), target="cuda")

        A_np = np.random.rand(32, 128).astype(dtype)
        B_np = np.zeros((32, 128)).astype(dtype)
        dev = tvm.cuda(0)
        A_nd = tvm.nd.array(A_np, device=dev)
        B_nd = tvm.nd.array(B_np, device=dev)
        mod(A_nd, B_nd)
        tvm.testing.assert_allclose(B_nd.numpy(), A_np)


expected_cuda_script = r"""__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

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
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  __shared__ float A_shared[64];
  __shared__ float B_shared[64];
  A_shared[((int)threadIdx.x)] = 0.000000e+00f;
  B_shared[((int)threadIdx.x)] = 0.000000e+00f;
__asm__ __volatile__("cp.async.commit_group;");


  {
    unsigned int addr = cast_smem_ptr_to_int(A_shared + (((int)threadIdx.x) + 16));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((int)threadIdx.x) * 14))), "n"(4)
    );
  }

  {
    unsigned int addr = cast_smem_ptr_to_int(B_shared + (((int)threadIdx.x) + 16));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((int)threadIdx.x) * 14))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");


  {
    unsigned int addr = cast_smem_ptr_to_int(A_shared + (((int)threadIdx.x) + 32));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((int)threadIdx.x) * 14) + 1))), "n"(4)
    );
  }

  {
    unsigned int addr = cast_smem_ptr_to_int(B_shared + (((int)threadIdx.x) + 32));
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((int)threadIdx.x) * 14) + 1))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int i = 0; i < 13; ++i) {
    bool cse_var_1 = (i < 12);

  {
    unsigned int addr = cast_smem_ptr_to_int(A_shared + ((((i + 3) & 3) * 16) + ((int)threadIdx.x)));
    int pred_guard = (int)cse_var_1;
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.ca.shared.global [%1], [%2], %3;"
      #endif
      "  @!p st.shared.u32 [%1], {%4};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)(A + (((((int)threadIdx.x) * 14) + i) + 2))), "n"(4), "r"(0)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 5;");

    __syncthreads();
    C[((((int)threadIdx.x) * 16) + i)] = (A_shared[(((i & 3) * 16) + ((int)threadIdx.x))] + B_shared[(((i & 3) * 16) + ((int)threadIdx.x))]);
    __syncthreads();

  {
    unsigned int addr = cast_smem_ptr_to_int(B_shared + ((((i + 3) & 3) * 16) + ((int)threadIdx.x)));
    int pred_guard = (int)cse_var_1;
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.ca.shared.global [%1], [%2], %3;"
      #endif
      "  @!p st.shared.u32 [%1], {%4};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)(B + (((((int)threadIdx.x) * 14) + i) + 2))), "n"(4), "r"(0)
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


@pytest.fixture
def postproc_if_missing_async_support():
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    support_async = major >= 8

    func_name = "tvm_callback_cuda_postproc"
    prev_postproc = tvm.get_global_func(func_name, allow_missing=True)

    # Store the generated code prior to the post-processing.  This
    # way, even though the generated code doesn't compile on platforms
    # that do not support async, the comparison against an expected
    # output can still be performed.  We cannot use
    # `mod.get_source()`, as that contains the source after all
    # post-processing.
    original_code = None

    def get_original_code():
        nonlocal original_code
        return original_code

    @tvm.register_func(func_name, override=True)
    def tvm_callback_cuda_postproc(code, _):
        nonlocal original_code
        original_code = code
        if support_async:
            return code
        else:
            ret = []
            for line in code.split("\n"):
                ret.append(line)
                ret.append("\n")
                if line.startswith('extern "C" __global__') and line.endswith("{"):
                    break
            ret.append("}")
            return "".join(ret)

    yield get_original_code

    # Restore previous postproc func to avoid impacting other tests
    if prev_postproc is None:
        tvm._ffi.registry.remove_global_func(func_name)
    else:
        tvm.register_func(func_name, prev_postproc, override=True)


@tvm.testing.requires_cuda
def test_cp_async_in_if_then_else(postproc_if_missing_async_support):
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
    generated_code = postproc_if_missing_async_support()
    assert generated_code == expected_cuda_script


@pytest.mark.skip(
    reason="This test fails due to an ordering issue with MergeSharedMemoryAllocations "
    "in device_driver_api.cc. However, fixing this causes failures in MLC. "
    "This bug should be addressed. See discussion in https://github.com/apache/tvm/pull/16769 "
    "and https://github.com/apache/tvm/pull/16569#issuecomment-1992720448"
)
@tvm.testing.requires_cuda
def test_vectorize_cp_async_in_if_then_else(postproc_if_missing_async_support):
    @T.prim_func
    def complex_compute(
        A: T.Buffer((2, 16, 16, 1280), "float16"),
        W: T.Buffer((1280, 3, 3, 1280), "float16"),
        Conv: T.Buffer((512, 1280), "float16"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        data_im2col_reindex_shared_dyn = T.alloc_buffer((512, 11520), "float16", scope="shared.dyn")
        data_im2col_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer(
            (512, 11520), "float16", scope="wmma.matrix_a"
        )
        weight_flatten_reindex_shared_dyn = T.alloc_buffer(
            (1280, 11520), "float16", scope="shared.dyn"
        )
        weight_flatten_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer(
            (1280, 11520), "float16", scope="wmma.matrix_b"
        )
        Conv_reindex_wmma_accumulator = T.alloc_buffer(
            (512, 1280), "float16", scope="wmma.accumulator"
        )
        for x_0_0 in T.thread_binding(8, thread="blockIdx.y"):
            for y_0_0 in T.thread_binding(20, thread="blockIdx.x"):
                for x_0_1 in T.thread_binding(2, thread="threadIdx.y"):
                    for y_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for x_0_2_init, y_0_2_init in T.grid(2, 2):
                            with T.block("Conv_init_o"):
                                v_x_o = T.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + x_0_2_init)
                                v_y_o = T.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + y_0_2_init)
                                T.reads()
                                T.writes(
                                    Conv_reindex_wmma_accumulator[
                                        v_x_o * 16 : v_x_o * 16 + 16, v_y_o * 16 : v_y_o * 16 + 16
                                    ]
                                )
                                C_s0 = T.int32()
                                C_s1 = T.int32()
                                C = T.match_buffer(
                                    Conv_reindex_wmma_accumulator[
                                        v_x_o * 16 : v_x_o * 16 + 16, v_y_o * 16 : v_y_o * 16 + 16
                                    ],
                                    (16, 16),
                                    "float16",
                                    strides=(C_s0, C_s1),
                                    scope="wmma.accumulator",
                                    offset_factor=16,
                                )
                                T.tvm_fill_fragment(
                                    C.data,
                                    16,
                                    16,
                                    16,
                                    C.elem_offset // C_s0 // 16 * (C_s0 // 16)
                                    + C.elem_offset % C_s0 // 16,
                                    T.float32(0),
                                )
                        for k_0_0 in T.serial(
                            180,
                            annotations={
                                "software_pipeline_stage": [0, 0, 1],
                                "software_pipeline_order": [0, 1, 2],
                                "software_pipeline_async_stages": [0],
                            },
                        ):
                            for ax0_ax1_0_fused_0 in range(4):
                                for ax0_ax1_0_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_0_fused_2 in T.thread_binding(
                                        2, thread="threadIdx.y"
                                    ):
                                        for ax0_ax1_0_fused_3 in T.thread_binding(
                                            32, thread="threadIdx.x"
                                        ):
                                            with T.block("data_im2col_reindex_shared.dyn_o"):
                                                v0 = T.axis.spatial(
                                                    512,
                                                    x_0_0 * 64
                                                    + (
                                                        ax0_ax1_0_fused_0 * 128
                                                        + ax0_ax1_0_fused_1 * 64
                                                        + ax0_ax1_0_fused_2 * 32
                                                        + ax0_ax1_0_fused_3
                                                    )
                                                    // 8,
                                                )
                                                v1_o = T.axis.spatial(
                                                    1440,
                                                    k_0_0 * 8
                                                    + (
                                                        ax0_ax1_0_fused_0 * 128
                                                        + ax0_ax1_0_fused_1 * 64
                                                        + ax0_ax1_0_fused_2 * 32
                                                        + ax0_ax1_0_fused_3
                                                    )
                                                    % 8,
                                                )
                                                T.reads(
                                                    A[
                                                        v0 // 256,
                                                        v1_o // 480 + v0 % 256 // 16 - 1,
                                                        v1_o % 480 // 160 + v0 % 16 - 1,
                                                        v1_o % 160 * 8 : v1_o % 160 * 8 + 8,
                                                    ]
                                                )
                                                T.writes(
                                                    data_im2col_reindex_shared_dyn[
                                                        v0, v1_o * 8 : v1_o * 8 + 8
                                                    ]
                                                )
                                                for ax1_1 in T.vectorized(8):
                                                    with T.block("data_im2col_reindex_shared.dyn"):
                                                        v1_i = T.axis.spatial(8, ax1_1)
                                                        T.reads(
                                                            A[
                                                                v0 // 256,
                                                                v1_o // 480 + v0 % 256 // 16 - 1,
                                                                v1_o % 480 // 160 + v0 % 16 - 1,
                                                                v1_o % 160 * 8 + v1_i,
                                                            ]
                                                        )
                                                        T.writes(
                                                            data_im2col_reindex_shared_dyn[
                                                                v0, v1_o * 8 + v1_i
                                                            ]
                                                        )
                                                        T.block_attr(
                                                            {"buffer_dim_align": [[0, 0, 32, 8]]}
                                                        )
                                                        data_im2col_reindex_shared_dyn[
                                                            v0, v1_o * 8 + v1_i
                                                        ] = T.if_then_else(
                                                            1 <= v1_o // 480 + v0 % 256 // 16
                                                            and v1_o // 480 + v0 % 256 // 16 < 17
                                                            and 1 <= v1_o % 480 // 160 + v0 % 16
                                                            and v1_o % 480 // 160 + v0 % 16 < 17,
                                                            A[
                                                                v0 // 256,
                                                                v1_o // 480 + v0 % 256 // 16 - 1,
                                                                v1_o % 480 // 160 + v0 % 16 - 1,
                                                                v1_o % 160 * 8 + v1_i,
                                                            ],
                                                            T.float16(0),
                                                        )
                            for ax0_ax1_0_fused_0 in range(4):
                                for ax0_ax1_0_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_0_fused_2 in T.thread_binding(
                                        2, thread="threadIdx.y"
                                    ):
                                        for ax0_ax1_0_fused_3 in T.thread_binding(
                                            32, thread="threadIdx.x"
                                        ):
                                            for ax1_1 in T.vectorized(8):
                                                with T.block("weight_flatten_reindex_shared.dyn"):
                                                    v0 = T.axis.spatial(
                                                        1280,
                                                        y_0_0 * 64
                                                        + (
                                                            ax0_ax1_0_fused_0 * 128
                                                            + ax0_ax1_0_fused_1 * 64
                                                            + ax0_ax1_0_fused_2 * 32
                                                            + ax0_ax1_0_fused_3
                                                        )
                                                        // 8,
                                                    )
                                                    v1 = T.axis.spatial(
                                                        11520,
                                                        k_0_0 * 64
                                                        + (
                                                            ax0_ax1_0_fused_0 * 128
                                                            + ax0_ax1_0_fused_1 * 64
                                                            + ax0_ax1_0_fused_2 * 32
                                                            + ax0_ax1_0_fused_3
                                                        )
                                                        % 8
                                                        * 8
                                                        + ax1_1,
                                                    )
                                                    T.reads(
                                                        W[
                                                            v0,
                                                            v1 // 3840,
                                                            v1 % 3840 // 1280,
                                                            v1 % 1280,
                                                        ]
                                                    )
                                                    T.writes(
                                                        weight_flatten_reindex_shared_dyn[v0, v1]
                                                    )
                                                    T.block_attr(
                                                        {"buffer_dim_align": [[0, 0, 32, 8]]}
                                                    )
                                                    weight_flatten_reindex_shared_dyn[v0, v1] = W[
                                                        v0,
                                                        v1 // 1280 // 3,
                                                        v1 // 1280 % 3,
                                                        v1 % 1280,
                                                    ]
                            for k_0_1 in range(4):
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with T.block("data_im2col_reindex_shared.dyn_wmma.matrix_a_o"):
                                        v0_o = T.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + ax0_0)
                                        v1_o = T.axis.spatial(720, k_0_0 * 4 + k_0_1 + ax1_0)
                                        T.reads(
                                            data_im2col_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        T.writes(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(
                                            data_im2col_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(A_s0, A_s1),
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C = T.match_buffer(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(C_s0, C_s1),
                                            scope="wmma.matrix_a",
                                            offset_factor=16,
                                        )
                                        T.tvm_load_matrix_sync(
                                            C.data,
                                            16,
                                            16,
                                            16,
                                            C.elem_offset // C_s0 // 16 * (C_s0 // 16)
                                            + C.elem_offset % C_s0 // 16,
                                            T.tvm_access_ptr(
                                                T.type_annotation("float16"),
                                                A_1.data,
                                                A_1.elem_offset,
                                                A_s0 * 16,
                                                1,
                                            ),
                                            A_s0,
                                            "row_major",
                                        )
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with T.block(
                                        "weight_flatten_reindex_shared.dyn_wmma.matrix_b_o"
                                    ):
                                        v0_o = T.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + ax0_0)
                                        v1_o = T.axis.spatial(720, k_0_0 * 4 + k_0_1 + ax1_0)
                                        T.reads(
                                            weight_flatten_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        T.writes(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ]
                                        )
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(
                                            weight_flatten_reindex_shared_dyn[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(A_s0, A_s1),
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C = T.match_buffer(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v0_o * 16 : v0_o * 16 + 16,
                                                v1_o * 16 : v1_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(C_s0, C_s1),
                                            scope="wmma.matrix_b",
                                            offset_factor=16,
                                        )
                                        T.tvm_load_matrix_sync(
                                            C.data,
                                            16,
                                            16,
                                            16,
                                            C.elem_offset // C_s0 // 16 * (C_s0 // 16)
                                            + C.elem_offset % C_s0 // 16,
                                            T.tvm_access_ptr(
                                                T.type_annotation("float16"),
                                                A_1.data,
                                                A_1.elem_offset,
                                                A_s0 * 16,
                                                1,
                                            ),
                                            A_s0,
                                            "col_major",
                                        )
                                for x_0_2, y_0_2 in T.grid(2, 2):
                                    with T.block("Conv_update_o"):
                                        v_x_o = T.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + x_0_2)
                                        v_y_o = T.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + y_0_2)
                                        v_k_o = T.axis.reduce(720, k_0_0 * 4 + k_0_1)
                                        T.reads(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ],
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                        )
                                        T.writes(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ]
                                        )
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(
                                            data_im2col_reindex_shared_dyn_wmma_matrix_a[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(A_s0, A_s1),
                                            scope="wmma.matrix_a",
                                            offset_factor=16,
                                        )
                                        B_s0 = T.int32()
                                        B_s1 = T.int32()
                                        B = T.match_buffer(
                                            weight_flatten_reindex_shared_dyn_wmma_matrix_b[
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                                v_k_o * 16 : v_k_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(B_s0, B_s1),
                                            scope="wmma.matrix_b",
                                            offset_factor=16,
                                        )
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C = T.match_buffer(
                                            Conv_reindex_wmma_accumulator[
                                                v_x_o * 16 : v_x_o * 16 + 16,
                                                v_y_o * 16 : v_y_o * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=(C_s0, C_s1),
                                            scope="wmma.accumulator",
                                            offset_factor=16,
                                        )
                                        T.tvm_mma_sync(
                                            C.data,
                                            C.elem_offset // C_s0 // 16 * (C_s0 // 16)
                                            + C.elem_offset % C_s0 // 16,
                                            A_1.data,
                                            A_1.elem_offset // A_s0 // 16 * (A_s0 // 16)
                                            + A_1.elem_offset % A_s0 // 16,
                                            B.data,
                                            B.elem_offset // B_s0 // 16 * (B_s0 // 16)
                                            + B.elem_offset % B_s0 // 16,
                                            C.data,
                                            C.elem_offset // C_s0 // 16 * (C_s0 // 16)
                                            + C.elem_offset % C_s0 // 16,
                                        )
                        for ax0_0, ax1_0 in T.grid(2, 2):
                            with T.block("Conv_reindex_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(32, x_0_0 * 4 + x_0_1 * 2 + ax0_0)
                                v1_o = T.axis.spatial(80, y_0_0 * 4 + y_0_1 * 2 + ax1_0)
                                T.reads(
                                    Conv_reindex_wmma_accumulator[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ]
                                )
                                T.writes(
                                    Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16]
                                )
                                A_s0 = T.int32()
                                A_s1 = T.int32()
                                A_1 = T.match_buffer(
                                    Conv_reindex_wmma_accumulator[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    (16, 16),
                                    "float16",
                                    strides=(A_s0, A_s1),
                                    scope="wmma.accumulator",
                                    offset_factor=16,
                                )
                                C_s0 = T.int32()
                                C_s1 = T.int32()
                                C = T.match_buffer(
                                    Conv[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16],
                                    (16, 16),
                                    "float16",
                                    strides=(C_s0, C_s1),
                                    offset_factor=16,
                                )
                                T.tvm_store_matrix_sync(
                                    A_1.data,
                                    16,
                                    16,
                                    16,
                                    A_1.elem_offset // A_s0 // 16 * (A_s0 // 16)
                                    + A_1.elem_offset % A_s0 // 16,
                                    T.tvm_access_ptr(
                                        T.type_annotation("float16"),
                                        C.data,
                                        C.elem_offset,
                                        C_s0 * 16,
                                        2,
                                    ),
                                    C_s0,
                                    "row_major",
                                )

    mod = tvm.IRModule.from_expr(complex_compute)
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        tvm.build(mod, target="cuda")
    generated_code = postproc_if_missing_async_support()
    # generated_code must contain "  setp.ne.b32 p, %0, 0;"
    assert "setp.ne.b32" in generated_code


class TestMultiplicationNodesAreInligned(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.InjectPTXAsyncCopy()

    def before(A: T.Buffer((32, 128), "float16")):
        tx = T.launch_thread("threadIdx.x", T.int64(32))
        A_flattened = T.Buffer((4096,), "float16", data=A.data)
        A_shared = T.decl_buffer([4096], "float16", scope="shared")

        T.attr("default", "async_scope", 1)
        for i in range(16):
            cse_var_1: T.int64 = T.Cast("int64", i)
            A_shared[
                T.Ramp(tx * T.int64(128) + cse_var_1 * T.int64(8), T.int64(1), 8)
            ] = A_flattened[T.Ramp(tx * T.int64(128) + cse_var_1 * T.int64(8), T.int64(1), 8)]
        T.ptx_commit_group()
        T.ptx_wait_group(0)

    def expected(A: T.Buffer((32, 128), "float16")):
        tx = T.launch_thread("threadIdx.x", T.int64(32))
        A_shared = T.decl_buffer((4096,), "float16", scope="shared")
        for i in range(16):
            cse_var_1: T.int64 = T.Cast("int64", i)
            T.ptx_cp_async(
                "float16",
                A_shared.data,
                tx * T.int64(128) + cse_var_1 * T.int64(8),
                A.data,
                tx * T.int64(128) + cse_var_1 * T.int64(8),
                16,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)


if __name__ == "__main__":
    tvm.testing.main()
