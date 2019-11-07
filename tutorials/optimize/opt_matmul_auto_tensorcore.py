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
==================================
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
# --------------------------
# 2 kinds of input data types are supported: float16 and int8.
# For float16, the accumulator is float32.
# For int8, the accumulator is int32.
# For data layouts, 'N' means None-transpose while 'T' means Transpose.

import logging
import sys

import numpy as np
import tvm

from tvm import autotvm
from tvm.contrib import nvcc

def matmul_nn(A, B, L, dtype='float16', layout='NN'):
    k = tvm.reduce_axis((0, L), name='k')
    if dtype == 'float16':
      out_type = 'float'
    elif dtype == 'int8':
      out_type = 'int'
    if (layout == 'NN'):
      return tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k].astype(out_type) * B[k, j].astype(out_type), axis=k))
    if (layout == 'NT'):
      return tvm.compute((N, M), lambda i, j: tvm.sum(A[k, i].astype(out_type) * B[k, j].astype(out_type), axis=k))
    if (layout == 'TN'):
      return tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k].astype(out_type) * B[j, k].astype(out_type), axis=k))
    if (layout == 'TT'):
      return tvm.compute((N, M), lambda i, j: tvm.sum(A[k, i].astype(out_type) * B[j, k].astype(out_type), axis=k))

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
# In this schedule, storage_align is used to reduce bank conflicts of shared memory.
# We use AutoTVM to search for best configurations in this schedule.

@autotvm.template
def test_gemm(N, L, M, dtype, layout):
    if (layout == "NN"):
      shape_a = (N, L)
      shape_b = (L, M)
    elif (layout == "NT"):
      shape_a = (L, N)
      shape_b = (L, M)
    elif (layout == "TN"):
      shape_a = (N, L)
      shape_b = (M, L)
    elif (layout == "TT"):
      shape_a = (L, N)
      shape_b = (M, L)
    else:
      print ("Unsupported layout:", layout)
      sys.exit(1);
    A = tvm.placeholder(shape_a, name='A', dtype=dtype)
    B = tvm.placeholder(shape_b, name='B', dtype=dtype)
    C = matmul_nn(A, B, L, dtype, layout)

    s = tvm.create_schedule(C.op)
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # storage_align params
    factor = 16
    offset = 8
    if dtype == 'int8':
      factor = 32
      offset = 16

    # create cache stages
    AA = s.cache_read(A, "shared", [C])
    if (layout == "NN" or layout == "TN"):
      s[AA].storage_align(AA.op.axis[0], factor, offset)
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    if (layout == "TT" or layout == "NT"):
      s[BB].storage_align(BB.op.axis[0], factor, offset)
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    #autotvm search space definition
    cfg = autotvm.get_config()

    cfg.define_knob("bx", [2, 4, 8])
    cfg.define_knob("by", [16, 32, 64])
    cfg.define_knob("step_k", [8, 16, 32])
    cfg.define_knob("v", [4, 8])
    by = cfg['by'].val
    bx = cfg['bx'].val
    step_k = cfg['step_k'].val
    v = cfg['v'].val

    # thread tile
    TX = 8
    TY = 1
    # warp tile
    warp_tile_m = 16 # it could also be 8 or 32 on CUDA version >= 10.0
    warp_tile_k = 16 # it must be 16
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
    s[C].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    # schedule for CL stage
    ko, ki = s[CL].split(k, step_k * warp_tile_k)
    kl, ki = s[CL].split(ki, warp_tile_k)
    s[CL].compute_at(s[C], tx)
    yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, yo, xo)

    # schedule for AA stage
    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[1], factor=bx*v)
    tz, tx = s[AA].split(xi, factor=(WX//TX)*v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[0], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))
    # vectorization is very important for float16/int8 inputs
    s[AA].vectorize(vec)

    # schedule for BB stage
    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[1], factor=bx*v)
    tz, tx = s[BB].split(xi, factor=(WX//TX)*v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[0], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[BB].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[BB].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    # set the 'tensor_core' pragma for tensorcore codegen
    s[CL].pragma(ko, 'tensor_core')

    return s, [A, B, C]

###############################################################################
# AutoTune and Test
# --------------------
# Finally we use a tuner to tune the schedule, generate code with best config
# and run the kernel to compare with numpy to check whether the results are correct.

# check whether the gpu has tensorcore
ctx = tvm.gpu()
if not nvcc.have_tensorcore(ctx.compute_version):
  print('the gpu has no tensorcore, skipping...')
  sys.exit(0)

M, N, L = 512, 32, 512
dtype = 'float16'
layout = 'NN'
if len(sys.argv) >= 4:
  M, N, L = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if len(sys.argv) >= 5:
  dtype = sys.argv[4]
if len(sys.argv) >= 6:
  layout = sys.argv[5]

task = autotvm.task.create(test_gemm, args=(N, L, M, dtype, layout), target='cuda')
print(task.config_space)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

tuner = autotvm.tuner.XGBTuner(task)
with tvm.build_config():
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])

dispatch_context = autotvm.apply_history_best("matmul.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("cuda"):
        with tvm.build_config():
            s, arg_bufs = test_gemm(N, L, M, dtype, layout)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
dev_module = func.imported_modules[0]
print(dev_module.get_source())

# check correctness
if (layout == "NN"):
  shape_a = (N, L)
  shape_b = (L, M)
elif (layout == "NT"):
  shape_a = (L, N)
  shape_b = (L, M)
elif (layout == "TN"):
  shape_a = (N, L)
  shape_b = (M, L)
elif (layout == "TT"):
  shape_a = (L, N)
  shape_b = (M, L)

a_np = None
b_np = None
c_np = None
c_np_type = None
if dtype == 'float16':
  c_np_type = np.float32
  a_np = np.random.uniform(size=shape_a).astype(np.float16)
  b_np = np.random.uniform(size=shape_b).astype(np.float16)
  if (layout == "NN"):
    c_np = np.dot(a_np, b_np)
  elif (layout == "NT"):
    c_np = np.dot(a_np.T, b_np)
  elif (layout == "TN"):
    c_np = np.dot(a_np, b_np.T)
  elif (layout == "TT"):
    c_np = np.dot(a_np.T, b_np.T)
elif dtype == 'int8':
  c_np_type = np.int32
  a_np = np.random.randint(low=-128, high=127, size=shape_a).astype(np.int8)
  b_np = np.random.randint(low=-128, high=127, size=shape_b).astype(np.int8)
  if (layout == "NN"):
    c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32))
  elif (layout == "NT"):
    c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32))
  elif (layout == "TN"):
    c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32).T)
  elif (layout == "TT"):
    c_np = np.dot(a_np.astype(np.int32).T, b_np.astype(np.int32).T)

c_tvm = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np_type), ctx=ctx)
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)
func(a_tvm, b_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-3)

evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
print('Time cost of this operator: %f' % evaluator(a_tvm, b_tvm, c_tvm).mean)

###############################################################################
# Summary
# --------------------------
# This tutorial demonstrates how to use the AutoTensorCoreCodeGen of TVM
# to generate tensorcore kernels.
