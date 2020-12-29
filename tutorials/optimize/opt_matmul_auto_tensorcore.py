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
"""
.. _opt-matmul-auto-tensorcore:

How to optimize matmul with Auto TensorCore CodeGen
===================================================
**Author**: `Minmin Sun <https://github.com/minminsun>`_, \
            `Lanbo Li <https://github.com/Orion34C>`_, \
            `Chenfan Jia <https://github.com/jcf94>`_, \
            `Jun Yang <https://github.com/yangjunpro>`_

In this tutorial, we will demonstrate how to write a high performance matmul
schedule on Volta/Turing GPUs with TVM Auto TensorCore CodeGen.
This is a transparent solution to generate tensorcore kernel
with most transformations done in ir passes.
Users can also write schedule with tensorization to generate TensorCore code.
Both solutions use the same tensorcore intrinsics.
Please refer to :ref:`opt-conv-tensorcore` tutorial for more details.
"""

################################################################
# Preparation and Algorithm
# -------------------------
# 2 kinds of input data types are supported: float16 and int8.
# For float16, the accumulator is float32.
# For int8, the accumulator is int32.
# For data layouts, 'N' means None-transpose while 'T' means Transpose.

import logging
import sys

import numpy as np
import tvm
from tvm import te

from tvm import autotvm
from tvm.contrib import nvcc


def matmul_nn(A, B, L, dtype="float16", layout="NN"):
    k = te.reduce_axis((0, L), name="k")
    if dtype == "float16":
        out_type = "float"
    elif dtype == "int8":
        out_type = "int"
    elif dtype == "int4" or dtype == "int1":
        out_type = "int"
    if layout == "NN":
        return te.compute(
            (N, M), lambda i, j: te.sum(A[i, k].astype(out_type) * B[k, j].astype(out_type), axis=k)
        )
    if layout == "NT":
        return te.compute(
            (N, M), lambda i, j: te.sum(A[k, i].astype(out_type) * B[k, j].astype(out_type), axis=k)
        )
    if layout == "TN":
        return te.compute(
            (N, M), lambda i, j: te.sum(A[i, k].astype(out_type) * B[j, k].astype(out_type), axis=k)
        )
    if layout == "TT":
        return te.compute(
            (N, M), lambda i, j: te.sum(A[k, i].astype(out_type) * B[j, k].astype(out_type), axis=k)
        )


###############################################################################
# Scheduling the Computation
# --------------------------
# This schedule is no different than a non-tensorcore matmul schedule on GPU.
# Please refer to :ref:`opt-gemm` tutorial for basics of optimizing matmul schedule.
# When the "tensor_core" pragma is set, the "rewrite for tensorcore" ir pass
# will automatically transform the schedule for tensorcore codegen,
# otherwise normal CUDA code, with lower performance but equal functionality, will be generated.
#
# .. note::
#
#   *Requirements of TesnsorCore*
#
#   Note that in the following 2 cases, even though the "tensor_core" pragma is set, TVM will still fall back to normal CUDA codegen:
#   (1) The m, n or k of input matrices is not multiple of 16;
#   (2) The warp tile size is not 16x16x16 on CUDA9, or not one of {16x16x16, 32x8x16, 8x32x16} on CUDA version >= 10.0.
#
# In this schedule, storage_align is used to reduce bank conflicts of shared memory. Please refer to this
# `doc <https://tvm.apache.org/docs/api/python/te.html#tvm.te.Stage.storage_align>`_
# for the usage of storage_align primitive. In short, we need to add an offset to some shared memory buffer
# to reduce bank conflicts.
# According to the `wmma doc <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description>`_,
# the stride of load_matrix_sync must be a multiple of 16 bytes,
# so we choose 8 as offset for float16 and 16 as offset for int8.
#
# We use AutoTVM to search for best configurations in this schedule.


@autotvm.template("tutorial/auto_tensorcore/test_gemm")
def test_gemm(N, L, M, dtype, layout):
    if layout == "NN":
        shape_a = (N, L)
        shape_b = (L, M)
    elif layout == "NT":
        shape_a = (L, N)
        shape_b = (L, M)
    elif layout == "TN":
        shape_a = (N, L)
        shape_b = (M, L)
    elif layout == "TT":
        shape_a = (L, N)
        shape_b = (M, L)
    else:
        print("Unsupported layout:", layout)
        sys.exit(1)
    A = te.placeholder(shape_a, name="A", dtype=dtype)
    B = te.placeholder(shape_b, name="B", dtype=dtype)
    C = matmul_nn(A, B, L, dtype, layout)

    s = te.create_schedule(C.op)
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # storage_align params
    factor = 16
    offset = 8
    if dtype == "int8":
        factor = 32
        offset = 16
    elif dtype == "int4":
        factor = 64
        offset = 32
    elif dtype == "int1":
        factor = 256
        offset = 128

    # create cache stages
    AA = s.cache_read(A, "shared", [C])
    if layout == "NN" or layout == "TN":
        s[AA].storage_align(AA.op.axis[0], factor, offset)
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    if layout == "TT" or layout == "NT":
        s[BB].storage_align(BB.op.axis[0], factor, offset)
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    # autotvm search space definition
    cfg = autotvm.get_config()

    cfg.define_knob("bx", [2, 4, 8])
    cfg.define_knob("by", [8, 16, 32, 64])
    cfg.define_knob("step_k", [1, 2, 4, 8, 16, 32])
    cfg.define_knob("v", [4, 8, 16, 32])
    by = cfg["by"].val
    bx = cfg["bx"].val
    step_k = cfg["step_k"].val
    v = cfg["v"].val

    # thread tile
    TX = 8
    TY = 1
    if dtype == "int4" or dtype == "int1":
        TX = 2
    # warp tile
    warp_tile_m = 16  # it could also be 8 or 32 on CUDA version >= 10.0
    warp_tile_k = 16  # it must be 16 for fp16/int8 data type
    if dtype == "int4":
        warp_tile_m = 8
        warp_tile_k = 32
    elif dtype == "int1":
        warp_tile_m = 8
        warp_tile_k = 128
    # block tile
    tile_x = bx * TX
    tile_y = by * TY

    yo, ty = s[C].split(y, tile_y)
    ty, yi = s[C].split(ty, TY)

    # schedule for C stage
    xo, xi = s[C].split(x, tile_x)
    WX = min(warp_tile_m, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    s[C].reorder(yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    # schedule for CL stage
    ko, ki = s[CL].split(k, step_k * warp_tile_k)
    kl, ki = s[CL].split(ki, warp_tile_k)
    s[CL].compute_at(s[C], tx)
    yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, yo, xo)

    # schedule for AA stage
    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[1], factor=bx * v)
    tz, tx = s[AA].split(xi, factor=(WX // TX) * v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[0], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tz, te.thread_axis("threadIdx.z"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    # vectorization is very important for float16/int8 inputs
    s[AA].vectorize(vec)

    # schedule for BB stage
    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[1], factor=bx * v)
    tz, tx = s[BB].split(xi, factor=(WX // TX) * v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[0], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tz, te.thread_axis("threadIdx.z"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    # set the 'tensor_core' pragma for tensorcore codegen
    s[CL].pragma(ko, "tensor_core")

    return s, [A, B, C]


###############################################################################
# AutoTune and Test
# -----------------
# Finally we use a tuner to tune the schedule, generate code with best config
# and run the kernel to compare with numpy to check whether the results are correct.

# check whether the gpu has tensorcore
if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
    raise Exception("skip building this tutorial because cuda is not enabled..")

ctx = tvm.gpu()
if not nvcc.have_tensorcore(ctx.compute_version):
    raise Exception("the gpu has no tensorcore, skipping...")

M, N, L = 512, 32, 512
dtype = "float16"
layout = "NN"
if len(sys.argv) >= 4:
    M, N, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if len(sys.argv) >= 5:
    dtype = sys.argv[4]
if len(sys.argv) >= 6:
    layout = sys.argv[5]

# check whether current gpu arch support support current dtype's wmma codegen
cuda_compute_capability = tvm.runtime._ffi_api.GetDeviceAttr(2, 0, 4)
major, minor = nvcc.parse_compute_version(cuda_compute_capability)
if dtype == "int8":
    assert major == 7 and minor >= 2
elif dtype == "int4" or dtype == "int1":
    # int4/int1 only support layout TN
    assert major == 7 and minor == 5 and layout == "TN"


def tune_and_evaluate(M, N, L, dtype, layout):
    task = autotvm.task.create(
        "tutorial/auto_tensorcore/test_gemm", args=(N, L, M, dtype, layout), target="cuda"
    )
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")],
    )

    dispatch_context = autotvm.apply_history_best("matmul.log")
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    with autotvm.apply_history_best("matmul.log"):
        with tvm.target.Target("cuda"):
            s, arg_bufs = test_gemm(N, L, M, dtype, layout)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
    dev_module = func.imported_modules[0]
    print(dev_module.get_source())

    # check correctness
    if layout == "NN":
        shape_a = (N, L)
        shape_b = (L, M)
    elif layout == "NT":
        shape_a = (L, N)
        shape_b = (L, M)
    elif layout == "TN":
        shape_a = (N, L)
        shape_b = (M, L)
    elif layout == "TT":
        shape_a = (L, N)
        shape_b = (M, L)

    a_np = None
    b_np = None
    c_np = None
    c_np_type = None
    if dtype == "float16":
        c_np_type = np.float32
        a_np = np.random.uniform(size=shape_a).astype(np.float16)
        b_np = np.random.uniform(size=shape_b).astype(np.float16)
        if layout == "NN":
            c_np = np.dot(a_np, b_np)
        elif layout == "NT":
            c_np = np.dot(a_np.T, b_np)
        elif layout == "TN":
            c_np = np.dot(a_np, b_np.T)
        elif layout == "TT":
            c_np = np.dot(a_np.T, b_np.T)
    elif dtype == "int8":
        c_np_type = np.int32
        a_np = np.random.randint(low=-128, high=127, size=shape_a).astype(np.int8)
        b_np = np.random.randint(low=-128, high=127, size=shape_b).astype(np.int8)
        if layout == "NN":
            c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32))
        elif layout == "NT":
            c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32))
        elif layout == "TN":
            c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32).T)
        elif layout == "TT":
            c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32).T)
    elif dtype == "int4":
        c_np_type = np.int32
        a_np_int = np.random.randint(low=-8, high=7, size=shape_a).astype(np.int32)
        b_np_int = np.random.randint(low=-8, high=7, size=shape_b).astype(np.int32)
        # "TN"
        c_np = np.dot(a_np_int.astype(np.int32), b_np_int.astype(np.int32).T)
        a_np = np.zeros(shape=(N, int(L / 8)), dtype=np.int32)
        b_np = np.zeros(shape=(M, int(L / 8)), dtype=np.int32)
        # a_np --> col_major
        for i in range(N):
            for j in range(int(L / 8)):
                for k in range(8):
                    a_np[i, j] = a_np[i, j] | ((a_np_int[i, j * 8 + k] & 0xF) << ((7 - k) * 4))

        # b_np --> row_major
        for i in range(M):
            for j in range(int(L / 8)):
                for k in range(8):
                    b_np[i, j] = b_np[i, j] | ((b_np_int[i, j * 8 + k] & 0xF) << ((7 - k) * 4))
    elif dtype == "int1":
        c_np_type = np.int32
        a_np_int = np.random.randint(low=0, high=1, size=shape_a).astype(np.int32)
        b_np_int = np.random.randint(low=0, high=1, size=shape_b).astype(np.int32)
        # "TN"
        c_np = np.dot(a_np_int.astype(np.int32), b_np_int.astype(np.int32).T)
        a_np = np.zeros(shape=(N, int(L / 32)), dtype=np.int32)
        b_np = np.zeros(shape=(M, int(L / 32)), dtype=np.int32)
        for i in range(N):
            for j in range(int(L / 32)):
                for k in range(32):
                    a_np[i, j] = a_np[i, j] | ((a_np_int[i, j * 32 + k] & 0xF) << (31 - k))

        for i in range(M):
            for j in range(int(L / 32)):
                for k in range(32):
                    b_np[i, j] = b_np[i, j] | ((b_np_int[i, j * 32 + k] & 0xF) << (31 - k))

    c_tvm = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np_type), ctx=ctx)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    b_tvm = tvm.nd.array(b_np, ctx=ctx)
    func(a_tvm, b_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-3)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    print("Time cost of this operator: %f" % evaluator(a_tvm, b_tvm, c_tvm).mean)


# We do not run the tuning in our webpage server since it takes some time.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(M, N, L, dtype, layout)

######################################################################
# Sample Output
# -------------
# .. code-block:: bash
#
#    Best config:
#    [('bx', 4), ('by', 32), ('step_k', 16), ('v', 8)],,None,40
#    Finish loading 162 records
#    produce compute {
#      // attr [iter_var(blockIdx.y, , blockIdx.y)] thread_extent = 1
#      // attr [compute.local] storage_scope = "wmma.accumulator"
#      allocate compute.local[float32 * 256]
#      // attr [A.shared] storage_scope = "shared"
#      allocate A.shared[float16 * 8448]
#      // attr [B.shared] storage_scope = "shared"
#      allocate B.shared[float16 * 8192]
#      // attr [A.shared.local] storage_scope = "wmma.matrix_b"
#      allocate A.shared.local[float16 * 256]
#      // attr [B.shared.local] storage_scope = "wmma.matrix_a"
#      allocate B.shared.local[float16 * 256]
#      // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 16
#      // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 2
#      // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 32
#      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 2
#      produce compute.local {
#        for (j.c.init, 0, 1) {
#          tvm_fill_fragment(compute.local, 16, 16, 16, 0, 0f)
#        }
#        // attr [iter_var(k.outer, )] pragma_tensor_core = 1
#        for (k.outer, 0, 2) {
#          produce A.shared {
#            for (ax0.ax1.outer.fused.outer, 0, 8) {
#              // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 32
#              // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 2
#              // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 2
#              A.shared[ramp((((((ax0.ax1.outer.fused.outer*1056) + (floordiv(threadIdx.y, 8)*264)) + (floormod(threadIdx.y, 8)*32)) + (threadIdx.z*16)) + (threadIdx.x*8)), 1, 8)] = A[ramp(((((((ax0.ax1.outer.fused.outer*2048) + (floordiv(threadIdx.y, 8)*512)) + (k.outer*256)) + (floormod(threadIdx.y, 8)*32)) + (threadIdx.z*16)) + (threadIdx.x*8)), 1, 8)]
#            }
#          }
#          produce B.shared {
#            for (ax0.ax1.outer.fused.outer, 0, 8) {
#              // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 32
#              // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 2
#              // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 2
#              B.shared[ramp(((((ax0.ax1.outer.fused.outer*1024) + (threadIdx.y*32)) + (threadIdx.z*16)) + (threadIdx.x*8)), 1, 8)] = B[ramp(((((((k.outer*131072) + (ax0.ax1.outer.fused.outer*16384)) + (threadIdx.y*512)) + (blockIdx.x*32)) + (threadIdx.z*16)) + (threadIdx.x*8)), 1, 8)]
#            }
#          }
#          for (k.inner.outer, 0, 16) {
#            produce A.shared.local {
#              for (ax1, 0, 1) {
#                tvm_load_matrix_sync(A.shared.local, 16, 16, 16, 0, &(A.shared[(((threadIdx.y/16)*4224) + (k.inner.outer*16))]), 264, "col_major")
#              }
#            }
#            produce B.shared.local {
#              for (ax0, 0, 1) {
#                for (ax1, 0, 1) {
#                  tvm_load_matrix_sync(B.shared.local, 16, 16, 16, 0, &(B.shared[((k.inner.outer*512) + (threadIdx.z*16))]), 32, "col_major")
#                }
#              }
#            }
#            for (k.inner.inner, 0, 1) {
#              for (j.c, 0, 1) {
#                tvm_mma_sync(compute.local, 0, B.shared.local, 0, A.shared.local, 0, compute.local, 0)
#              }
#            }
#          }
#        }
#      }
#      for (j.inner.inner.inner, 0, 1) {
#        tvm_store_matrix_sync(compute.local, 16, 16, 16, 0, &(compute[((((threadIdx.y/16)*8192) + (blockIdx.x*32)) + (threadIdx.z*16))]), 512, "col_major")
#      }
#    }
#
#    #include <cuda_fp16.h>
#    __device__ half max(const half a, const half b)
#    {
#      return __hgt(__half(a), __half(b)) ? a : b;
#    }
#    __device__ half min(const half a, const half b)
#    {
#      return __hlt(__half(a), __half(b)) ? a : b;
#    }
#    __device__ half operator+(const volatile __half &a,  const volatile __half &b)
#    {
#      return __hadd(a, b);
#    }
#    __device__ half operator<=(const volatile __half &a,  const volatile __half &b)
#    {
#      return __hlt(a, b);
#    }
#    __device__ half operator*(const volatile __half &a,  const volatile __half &b)
#    {
#      return __hmul(a, b);
#    }
#    #include <mma.h>
#    extern "C" __global__ void default_function_kernel0( half* __restrict__ A,  half* __restrict__ B,  float* __restrict__ compute) {
#      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> compute_local[1];
#      __shared__ half A_shared[8448];
#      __shared__ half B_shared[8192];
#      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> A_shared_local[1];
#      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> B_shared_local[1];
#      for (int j_c_init = 0; j_c_init < 1; ++j_c_init) {
#        (void)nvcuda::wmma::fill_fragment(compute_local[0], 0.000000e+00f);
#      }
#      for (int k_outer = 0; k_outer < 2; ++k_outer) {
#        __syncthreads();
#        for (int ax0_ax1_outer_fused_outer = 0; ax0_ax1_outer_fused_outer < 8; ++ax0_ax1_outer_fused_outer) {
#          ((__shared__ float4*)(A_shared + (((((ax0_ax1_outer_fused_outer * 1056) + ((((int)threadIdx.y) >> 3) * 264)) + ((((int)threadIdx.y) & 7) * 32)) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 8))))[0] = (( float4*)(A + ((((((ax0_ax1_outer_fused_outer * 2048) + ((((int)threadIdx.y) >> 3) * 512)) + (k_outer * 256)) + ((((int)threadIdx.y) & 7) * 32)) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 8))))[0];
#        }
#        for (int ax0_ax1_outer_fused_outer1 = 0; ax0_ax1_outer_fused_outer1 < 8; ++ax0_ax1_outer_fused_outer1) {
#          ((__shared__ float4*)(B_shared + ((((ax0_ax1_outer_fused_outer1 * 1024) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 8))))[0] = (( float4*)(B + ((((((k_outer * 131072) + (ax0_ax1_outer_fused_outer1 * 16384)) + (((int)threadIdx.y) * 512)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 8))))[0];
#        }
#        __syncthreads();
#        for (int k_inner_outer = 0; k_inner_outer < 16; ++k_inner_outer) {
#          for (int ax1 = 0; ax1 < 1; ++ax1) {
#            (void)nvcuda::wmma::load_matrix_sync(A_shared_local[0], &(A_shared[(((((int)threadIdx.y) / 16) * 4224) + (k_inner_outer * 16))]), 264);
#          }
#          for (int ax0 = 0; ax0 < 1; ++ax0) {
#            for (int ax11 = 0; ax11 < 1; ++ax11) {
#              (void)nvcuda::wmma::load_matrix_sync(B_shared_local[0], &(B_shared[((k_inner_outer * 512) + (((int)threadIdx.z) * 16))]), 32);
#            }
#          }
#          for (int k_inner_inner = 0; k_inner_inner < 1; ++k_inner_inner) {
#            for (int j_c = 0; j_c < 1; ++j_c) {
#              (void)nvcuda::wmma::mma_sync(compute_local[0], B_shared_local[0], A_shared_local[0], compute_local[0]);
#            }
#          }
#        }
#      }
#      for (int j_inner_inner_inner = 0; j_inner_inner_inner < 1; ++j_inner_inner_inner) {
#        (void)nvcuda::wmma::store_matrix_sync(&(compute[((((((int)threadIdx.y) / 16) * 8192) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.z) * 16))]), compute_local[0], 512, nvcuda::wmma::mem_col_major);
#      }
#    }
#
#
#    Time cost of this operator: 0.000008

###############################################################################
# Summary
# -------
# This tutorial demonstrates how to use the AutoTensorCoreCodeGen of TVM
# to generate tensorcore kernels.
