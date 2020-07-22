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

"""Test loop state and schedule primitives"""

import numpy as np

import tvm
from tvm import auto_scheduler, te
import topi

from test_auto_scheduler_common import matmul_auto_scheduler_test, conv2d_nchw_bn_relu


def test_split_fuse_reorder_annotation():
    A, B, C = matmul_auto_scheduler_test(N=512, M=512, K=512)
    dag = auto_scheduler.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()
    i, j, k = s0[C].iters

    assert i.range.extent == 512

    io, ii = s0.split(C, i, [16])
    assert s0[C].iters[0] == io
    assert s0[C].iters[1] == ii
    assert io.range.extent == 32
    assert ii.range.extent == 16

    jo, ji = s0.split(C, j, [8])
    assert jo.range.extent == 64
    assert ji.range.extent == 8

    s0.reorder(C, [io, jo, k, ji, ii])
    assert s0[C].iters[2].range.extent == 512

    fused_it = s0.fuse(C, [io, jo])
    assert fused_it.range.extent == 2048

    s1 = dag.get_init_state()
    i, j, _ = s1[C].iters
    i1, i2, i3 = s1.split(C, i, [8, 2])
    j1, j2, j3 = s1.split(C, j, [32, 8], False)
    assert s1[C].iters[0].range.extent == 32
    assert s1[C].iters[1].range.extent == 8
    assert s1[C].iters[2].range.extent == 2
    assert s1[C].iters[3].range.extent == 32
    assert s1[C].iters[4].range.extent == 8
    assert s1[C].iters[5].range.extent == 2

    res = s1.bind(C, i1, "blockIdx.x")
    assert res == s1[C].iters[0]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["blockIdx.x"]

    res = s1.bind(C, i2, "vthread")
    assert res == s1[C].iters[1]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["vthread"]

    res = s1.bind(C, i3, "threadIdx.y")
    assert res == s1[C].iters[2]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["threadIdx.y"]

    res = s1.parallel(C, j1)
    assert res == s1[C].iters[3]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["parallel"]

    res = s1.unroll(C, j2)
    assert res == s1[C].iters[4]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["unroll"]

    res = s1.vectorize(C, j3)
    assert res == s1[C].iters[5]
    assert res.annotation == auto_scheduler.loop_state.State.ANNOTATION_TRANS_TABLE["vectorize"]


def test_compute_at_root_inline():
    dag = auto_scheduler.ComputeDAG(conv2d_nchw_bn_relu(N=1, H=224, W=224, CI=3, CO=64,
                                                        kernel_size=7, strides=2, padding=3))
    s0 = dag.get_init_state()

    # data, padding, kernel = 0, 1, 2
    conv = s0.stage_ops[3]
    # bias = 4
    bias_add = s0.stage_ops[5]
    # bn_scale = 6
    bn_mul = s0.stage_ops[7]
    # bn_offset = 8
    bn_add = s0.stage_ops[9]
    relu = s0.stage_ops[10]

    s0.compute_inline(bn_add)
    assert s0[bn_add].compute_at == 1

    s0.compute_inline(bn_mul)
    assert s0[bn_mul].compute_at == 1

    s0.compute_inline(bias_add)
    assert s0[bias_add].compute_at == 1

    assert s0[conv].iters[0].range.extent == 1
    assert s0[conv].iters[1].range.extent == 64
    assert s0[conv].iters[2].range.extent == 112
    assert s0[conv].iters[3].range.extent == 112
    assert s0[conv].iters[4].range.extent == 3
    assert s0[conv].iters[5].range.extent == 7
    assert s0[conv].iters[6].range.extent == 7
    s0.compute_at(conv, relu, s0[relu].iters[2])
    assert s0[conv].compute_at == 2
    s0 = dag.infer_bound_from_state(s0)
    assert s0[conv].iters[0].range.extent == 1
    assert s0[conv].iters[1].range.extent == 1
    assert s0[conv].iters[2].range.extent == 1
    assert s0[conv].iters[3].range.extent == 112
    assert s0[conv].iters[4].range.extent == 3
    assert s0[conv].iters[5].range.extent == 7
    assert s0[conv].iters[6].range.extent == 7

    s0.compute_root(bn_mul)
    assert s0[bn_mul].compute_at == 0

    s0.compute_root(conv)
    assert s0[conv].compute_at == 0
    s0 = dag.infer_bound_from_state(s0)
    assert s0[conv].iters[0].range.extent == 1
    assert s0[conv].iters[1].range.extent == 64
    assert s0[conv].iters[2].range.extent == 112
    assert s0[conv].iters[3].range.extent == 112
    assert s0[conv].iters[4].range.extent == 3
    assert s0[conv].iters[5].range.extent == 7
    assert s0[conv].iters[6].range.extent == 7


if __name__ == "__main__":
    test_split_fuse_reorder_annotation()
    test_compute_at_root_inline()
