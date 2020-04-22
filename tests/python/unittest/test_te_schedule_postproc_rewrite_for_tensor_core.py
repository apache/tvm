# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import te
import topi
import numpy as np
from tvm.contrib import nvcc

def tensor_core_matmul(warp_tile_m=16, m=64, n=32, l=96):
    A = te.placeholder((n, l), name='A', dtype='float16')
    B = te.placeholder((l, m), name='B', dtype='float16')
    k = te.reduce_axis((0, l), name='k')
    C = te.compute((n, m), lambda i, j: te.sum(A[i, k].astype('float32') * B[k, j].astype('float32'), axis=k))
    s = te.create_schedule(C.op)
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    bx = 4
    by = 32
    step_k = 8
    v = 4
    TX = 8
    TY = 1
    tile_x = bx * TX
    tile_y = by * TY
    WX = min(warp_tile_m, tile_x)
    tile_k = 16
    vthread = 1

    yo, ty = s[C].split(y, tile_y*vthread)
    vy, ty = s[C].split(ty, tile_y)
    ty, yi = s[C].split(ty, TY)

    xo, xi = s[C].split(x, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    ko, ki = s[CL].split(k, step_k * tile_k)
    kl, ki = s[CL].split(ki, tile_k)

    s[C].reorder(yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].bind(vy, te.thread_axis((0, vthread), "vthread", name="vy"))
    s[CL].compute_at(s[C], tx)
    yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, yo, xo)

    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[1], factor=bx*v)
    tz, tx = s[AA].split(xi, factor=(WX//TX)*v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[0], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tz, te.thread_axis("threadIdx.z"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    s[AA].vectorize(vec)

    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[1], factor=bx*v)
    tz, tx = s[BB].split(xi, factor=(WX//TX)*v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[0], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tz, te.thread_axis("threadIdx.z"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    s[CL].pragma(ko, 'tensor_core')

    func = tvm.build(s, [A, B, C], 'cuda')

    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(l, m)).astype(B.dtype)
    c_np = np.zeros((n, m), dtype=np.float32)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=3)
    print('gemm m=%d n=%d k=%d: %f ms' % (m, n, l, evaluator(a, b, c).mean * 1e3))

    c_np = np.dot(a_np, b_np)
    np.testing.assert_allclose(c_np, c.asnumpy(), rtol=1e-3)

def tensor_core_batch_matmul(warp_tile_m=16, m=64, n=32, l=96, batch=2):
    A = te.placeholder((batch, n, l), name='A', dtype='float16')
    B = te.placeholder((batch, l, m), name='B', dtype='float16')
    k = te.reduce_axis((0, l), name='k')
    C = te.compute((batch, n, m), lambda b, i, j: te.sum((A[b, i, k] * B[b, k, j]).astype('float32'), axis=k))
    s = te.create_schedule(C.op)
    z, y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    bx = 2
    by = 32
    step_k = 8
    v = 4
    TX = 8
    TY = 1
    tile_x = bx * TX
    tile_y = by * TY
    WX = min(warp_tile_m, tile_x)
    tile_k = 16
    vthread = 1

    yo, ty = s[C].split(y, tile_y*vthread)
    vy, ty = s[C].split(ty, tile_y)
    ty, yi = s[C].split(ty, TY)

    xo, xi = s[C].split(x, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    ko, ki = s[CL].split(k, step_k * tile_k)
    kl, ki = s[CL].split(ki, tile_k)

    s[C].reorder(z, yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(z, te.thread_axis("blockIdx.z"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].bind(vy, te.thread_axis((0, vthread), "vthread", name="vy"))
    s[CL].compute_at(s[C], tx)
    zo, yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, zo, yo, xo)

    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[2], factor=bx*v)
    tz, tx = s[AA].split(xi, factor=(WX//TX)*v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[1], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tz, te.thread_axis("threadIdx.z"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    s[AA].vectorize(vec)

    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[2], factor=bx*v)
    tz, tx = s[BB].split(xi, factor=(WX//TX)*v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[1], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tz, te.thread_axis("threadIdx.z"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    s[CL].pragma(ko, 'tensor_core')

    func = tvm.build(s, [A, B, C], 'cuda')

    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(batch, n, l)).astype(A.dtype)
    b_np = np.random.uniform(size=(batch, l, m)).astype(B.dtype)
    c_np = np.zeros((batch, n, m), dtype=np.float32)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((batch, n, m), dtype=C.dtype), ctx)
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=3)
    print('batch gemm m=%d n=%d k=%d batch=%d: %f ms' % (m, n, l, batch, evaluator(a, b, c).mean * 1e3))

    for bs in range(batch):
      c_np[bs, :, :] = np.dot(a_np[bs, :, :], b_np[bs, :, :])
    np.testing.assert_allclose(c_np, c.asnumpy(), rtol=1e-3)

def test_tensor_core_matmul():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return
    if not nvcc.have_tensorcore(tvm.gpu(0).compute_version):
        print("skip because gpu does not support tensor core")
        return

    tensor_core_matmul(16) #test with warp_tile 16x16x16
    tensor_core_matmul(8) #test with warp_tile 8x32x16
    tensor_core_matmul(32) #test with warp_tile 32x8x16

def test_tensor_core_batch_matmul():
    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("skip because cuda is not enabled..")
        return
    if not nvcc.have_tensorcore(tvm.gpu(0).compute_version):
        print("skip because gpu does not support tensor core")
        return

    tensor_core_batch_matmul()

if __name__ == '__main__':
    test_tensor_core_matmul()
    test_tensor_core_batch_matmul()
