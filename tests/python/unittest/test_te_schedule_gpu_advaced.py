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
import pytest
import tvm
from tvm import te
import numpy as np

def vcf_check_common(s, args):
    N = 512

    # To check if every vectorize loop transforms to ramp expr successfully
    # TODO(jcf94): Find a better way to process the check in AST
    print(tvm.lower(s, args))

    if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
        print("CUDA device not found, skip the verification.")
    else:
        tgt = tvm.target.cuda()
        mod = tvm.build(s, args, tgt)
        # To check if every vectorize loop transforms to correct instruction
        # print(mod.imported_modules[0].get_source())

        ctx = tvm.context("cuda", 0)
        a = tvm.nd.array(np.random.uniform(size=(512, 512)).astype("float32"), ctx)
        b = tvm.nd.array(np.random.uniform(size=(512, 512)).astype("float32"), ctx)
        c = tvm.nd.array(np.zeros((512, 512), dtype="float32"), ctx)
        mod(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), np.dot(
            a.asnumpy(), b.asnumpy()), rtol=1e-5)

def test_vectorized_cooperative_fetching_x():
    N = 512
    A = te.placeholder((N, N), name='A', dtype='float32')
    B = te.placeholder((N, N), name='B', dtype='float32')
    k = te.reduce_axis((0, N), name='k')
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    s = te.create_schedule(C.op)
    i, j = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])

    i3, i4 = s[C].split(i, factor=4)
    i2, i3 = s[C].split(i3, factor=2)
    i1, i2 = s[C].split(i2, factor=8)
    i0, i1 = s[C].split(i1, factor=1)
    j3, j4 = s[C].split(j, factor=4)
    j2, j3 = s[C].split(j3, factor=2)
    j1, j2 = s[C].split(j2, factor=8)
    j0, j1 = s[C].split(j1, factor=2)
    k1, k2 = s[C].split(k, factor=8)
    k0, k1 = s[C].split(k1, factor=8)
    s[C].reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3, k2, i4, j4)
    block_it = s[C].fuse(i0, j0)
    s[C].bind(block_it, tvm.te.thread_axis("blockIdx.x"))
    vthread_it = s[C].fuse(i1, j1)
    s[C].bind(vthread_it, tvm.te.thread_axis("vthread"))
    thread_it = s[C].fuse(i2, j2)
    s[C].bind(thread_it, tvm.te.thread_axis("threadIdx.x"))
    s[C].vectorize(j4)

    s[AA].compute_at(s[C], k0)
    iaa, jaa = s[AA].op.axis
    s[BB].compute_at(s[C], k0)
    ibb, jbb = s[BB].op.axis
    aa_fused = s[AA].fuse(iaa, jaa)
    bb_fused = s[BB].fuse(ibb, jbb)
    aa1, aa2 = s[AA].split(aa_fused, factor=4)
    aa0, aa1 = s[AA].split(aa1, factor=64)
    bb1, bb2 = s[BB].split(bb_fused, factor=4)
    bb0, bb1 = s[BB].split(bb1, factor=64)
    s[AA].bind(aa1, tvm.te.thread_axis("threadIdx.x"))
    s[AA].vectorize(aa2)
    s[BB].bind(bb1, tvm.te.thread_axis("threadIdx.x"))
    s[BB].vectorize(bb2)

    vcf_check_common(s, [A, B, C])

def test_vectorized_cooperative_fetching_xy():
    N = 512
    A = te.placeholder((N, N), name='A')
    B = te.placeholder((N, N), name='B')
    k = te.reduce_axis((0, N), name='k')
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    s = te.create_schedule(C.op)
    i, j = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])

    i3, i4 = s[C].split(i, factor=4)
    i2, i3 = s[C].split(i3, factor=2)
    i1, i2 = s[C].split(i2, factor=8)
    i0, i1 = s[C].split(i1, factor=1)
    j3, j4 = s[C].split(j, factor=4)
    j2, j3 = s[C].split(j3, factor=2)
    j1, j2 = s[C].split(j2, factor=8)
    j0, j1 = s[C].split(j1, factor=2)
    k1, k2 = s[C].split(k, factor=8)
    k0, k1 = s[C].split(k1, factor=8)
    s[C].reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3, k2, i4, j4)
    block_it = s[C].fuse(i0, j0)
    s[C].bind(block_it, tvm.te.thread_axis("blockIdx.x"))
    vthread_it = s[C].fuse(i1, j1)
    s[C].bind(vthread_it, tvm.te.thread_axis("vthread"))
    s[C].bind(i2, tvm.te.thread_axis("threadIdx.y"))
    s[C].bind(j2, tvm.te.thread_axis("threadIdx.x"))
    s[C].vectorize(j4)

    s[AA].compute_at(s[C], k0)
    iaa, jaa = s[AA].op.axis
    s[BB].compute_at(s[C], k0)
    ibb, jbb = s[BB].op.axis
    aa_fused = s[AA].fuse(iaa, jaa)
    bb_fused = s[BB].fuse(ibb, jbb)
    aa2, aa3 = s[AA].split(aa_fused, factor=4)
    aa1, aa2 = s[AA].split(aa2, factor=8)
    aa0, aa1 = s[AA].split(aa1, factor=8)
    bb2, bb3 = s[BB].split(bb_fused, factor=4)
    bb1, bb2 = s[BB].split(bb2, factor=8)
    bb0, bb1 = s[BB].split(bb1, factor=8)
    s[AA].bind(aa1, tvm.te.thread_axis("threadIdx.y"))
    s[AA].bind(aa2, tvm.te.thread_axis("threadIdx.x"))
    s[AA].vectorize(aa3)
    s[BB].bind(bb1, tvm.te.thread_axis("threadIdx.y"))
    s[BB].bind(bb2, tvm.te.thread_axis("threadIdx.x"))
    s[BB].vectorize(bb3)

    vcf_check_common(s, [A, B, C])

if __name__ == "__main__":
    test_vectorized_cooperative_fetching_x()
    test_vectorized_cooperative_fetching_xy()
