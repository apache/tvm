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

""" Test for vectorized cooperative fetching """

import numpy as np
import tvm
from tvm import ansor, te
import topi

from test_ansor_common import matmul_ansor_test, conv2d_nchw_bn_relu


def init_common():
    dag = ansor.ComputeDAG(matmul_ansor_test(512, 512, 512))
    s0 = dag.get_init_state()
    A, B, C = 0, 1, 2
    B_shared = s0.cache_read(B, "shared", [C], dag)
    C += 1
    B_local = s0.cache_read(B_shared, "local", [C], dag)
    C += 1
    A_shared = s0.cache_read(A, "shared", [C], dag)
    B += 1
    B_shared += 1
    B_local += 1
    C += 1
    A_local = s0.cache_read(A_shared, "local", [C], dag)
    B += 1
    B_shared += 1
    B_local += 1
    C += 1

    return A_shared, A_local, B_shared, B_local, C, dag, s0

def check_common(dag, state):
    s, args = dag.apply_steps_from_state(state)
    # To check if every vectorize loop transforms to ramp expr successfully
    # TODO(jcf94): Find a better way to process the check in AST
    print(tvm.lower(s, args))

    if tvm.context("cuda", 0).exist:
        tgt = tvm.target.cuda()
        mod = tvm.build(s, args, tgt)
        # To check if every vectorize loop transforms to correct instruction
        print(mod.imported_modules[0].get_source())

        ctx = tvm.context("cuda", 0)
        dtype = dag.tensors[0].dtype
        a = tvm.nd.array(np.random.uniform(size=(512, 512)).astype(dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(512, 512)).astype(dtype), ctx)
        c = tvm.nd.array(np.zeros((512, 512), dtype=dtype), ctx)
        mod(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), np.dot(
            a.asnumpy(), b.asnumpy()), rtol=1e-5)
    else:
        print("CUDA device not found, skip this test.")

def test_vectorized_cooperative_fetching_x():
    A_shared, A_local, B_shared, B_local, C, dag, s0 = init_common()

    its0 = s0.split(C, s0.stages[C].iters[0], [1, 8, 2, 4])
    its1 = s0.split(C, s0.stages[C].iters[5], [2, 8, 2, 4])
    its2 = s0.split(C, s0.stages[C].iters[10], [8, 8])
    s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its2[0],
                   its2[1], its0[3], its1[3], its2[2], its0[4], its1[4]])
    s0.fuse(C, [s0.stages[C].iters[0], s0.stages[C].iters[1]])
    s0.bind_thread(C, s0.stages[C].iters[0], "blockIdx.x")
    s0.fuse(C, [s0.stages[C].iters[1], s0.stages[C].iters[2]])
    s0.bind_thread(C, s0.stages[C].iters[1], "vthread")
    s0.fuse(C, [s0.stages[C].iters[2], s0.stages[C].iters[3]])
    s0.bind_thread(C, s0.stages[C].iters[2], "threadIdx.x")
    s0.vectorize(C, its1[4])

    s0.compute_at(B_shared, C, s0.stages[C].iters[3])
    fused_it = s0.fuse(B_shared, s0.stages[B_shared].iters[:])
    its = s0.split(B_shared, fused_it, [64, 4])
    s0.bind_thread(B_shared, its[1], "threadIdx.x")
    s0.vectorize(B_shared, its[2])
    s0.compute_at(B_local, C, s0.stages[C].iters[4])
    fused_it = s0.fuse(B_local, s0.stages[B_local].iters[:])
    its = s0.split(B_local, fused_it, [4])
    s0.vectorize(B_local, its[1])

    s0.compute_at(A_shared, C, s0.stages[C].iters[3])
    fused_it = s0.fuse(A_shared, s0.stages[A_shared].iters[:])
    its = s0.split(A_shared, fused_it, [64, 4])
    s0.bind_thread(A_shared, its[1], "threadIdx.x")
    s0.vectorize(A_shared, its[2])
    s0.compute_at(A_local, C, s0.stages[C].iters[4])
    fused_it = s0.fuse(A_local, s0.stages[A_local].iters[:])
    its = s0.split(A_local, fused_it, [4])
    s0.vectorize(A_local, its[1])

    check_common(dag, s0)

def test_vectorized_cooperative_fetching_xy():
    A_shared, A_local, B_shared, B_local, C, dag, s0 = init_common()

    its0 = s0.split(C, s0.stages[C].iters[0], [1, 8, 2, 4])
    its1 = s0.split(C, s0.stages[C].iters[5], [2, 8, 2, 4])
    its2 = s0.split(C, s0.stages[C].iters[10], [8, 8])
    s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its2[0],
                   its2[1], its0[3], its1[3], its2[2], its0[4], its1[4]])
    s0.fuse(C, [s0.stages[C].iters[0], s0.stages[C].iters[1]])
    s0.bind_thread(C, s0.stages[C].iters[0], "blockIdx.x")
    s0.fuse(C, [s0.stages[C].iters[1], s0.stages[C].iters[2]])
    s0.bind_thread(C, s0.stages[C].iters[1], "vthread")
    s0.bind_thread(C, s0.stages[C].iters[2], "threadIdx.x")
    s0.bind_thread(C, s0.stages[C].iters[3], "threadIdx.y")
    s0.vectorize(C, its1[4])

    s0.compute_at(B_shared, C, s0.stages[C].iters[4])
    fused_it = s0.fuse(B_shared, s0.stages[B_shared].iters[:])
    its = s0.split(B_shared, fused_it, [8, 8, 4])
    s0.bind_thread(B_shared, its[1], "threadIdx.x")
    s0.bind_thread(B_shared, its[2], "threadIdx.y")
    s0.vectorize(B_shared, its[3])
    s0.compute_at(B_local, C, s0.stages[C].iters[5])
    fused_it = s0.fuse(B_local, s0.stages[B_local].iters[:])
    its = s0.split(B_local, fused_it, [4])
    s0.vectorize(B_local, its[1])

    s0.compute_at(A_shared, C, s0.stages[C].iters[4])
    fused_it = s0.fuse(A_shared, s0.stages[A_shared].iters[:])
    its = s0.split(A_shared, fused_it, [8, 8, 4])
    s0.bind_thread(A_shared, its[1], "threadIdx.x")
    s0.bind_thread(A_shared, its[2], "threadIdx.y")
    s0.vectorize(A_shared, its[3])
    s0.compute_at(A_local, C, s0.stages[C].iters[5])
    fused_it = s0.fuse(A_local, s0.stages[A_local].iters[:])
    its = s0.split(A_local, fused_it, [4])
    s0.vectorize(A_local, its[1])

    check_common(dag, s0)

if __name__ == "__main__":
    test_vectorized_cooperative_fetching_x()
    test_vectorized_cooperative_fetching_xy()
