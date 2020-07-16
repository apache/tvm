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
    A, B, C = matmul_auto_scheduler_test(512, 512, 512)
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

    s1.parallel(C, j1)
    s1.unroll(C, j2)
    s1.vectorize(C, j3)
    s1.bind(C, i1, "blockIdx.x")
    s1.bind(C, i2, "vthread")
    s1.bind(C, i3, "threadIdx.y")


def test_compute_at_root_inline():
    dag = auto_scheduler.ComputeDAG(conv2d_nchw_bn_relu(1, 224, 224, 3, 64, 7, 2, 3))
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
    s0.compute_inline(bn_mul)
    s0.compute_inline(bias_add)
    s0.compute_at(conv, relu, s0[relu].iters[2])
    print(s0)
    assert str(s0) == \
        "Placeholder: Data, Kernel, Bias, Bn_scale, Bn_offset\n" + \
        "for i1 (0,3)\n" + \
        "  for i2 (0,230)\n" + \
        "    for i3 (0,230)\n" + \
        "      pad_temp = ...\n" + \
        "for i1 (0,64)\n" + \
        "  for i2 (0,112)\n" + \
        "    for nn (None)\n" + \
        "      for ff (None)\n" + \
        "        for yy (None)\n" + \
        "          for xx (None)\n" + \
        "            for rc (None)\n" + \
        "              for ry (None)\n" + \
        "                for rx (None)\n" + \
        "                  compute = ...\n" + \
        "    for i3 (0,112)\n" + \
        "      compute = ...\n"

    s0.compute_root(conv)
    s0.compute_root(bn_mul)
    assert str(s0) == \
        "Placeholder: Data, Kernel, Bias, Bn_scale, Bn_offset\n" + \
        "for i1 (0,3)\n" + \
        "  for i2 (0,230)\n" + \
        "    for i3 (0,230)\n" + \
        "      pad_temp = ...\n" + \
        "for nn (None)\n" + \
        "  for ff (None)\n" + \
        "    for yy (None)\n" + \
        "      for xx (None)\n" + \
        "        for rc (None)\n" + \
        "          for ry (None)\n" + \
        "            for rx (None)\n" + \
        "              compute = ...\n" + \
        "for i (None)\n" + \
        "  for j (None)\n" + \
        "    for k (None)\n" + \
        "      for l (None)\n" + \
        "        Bn_mul = ...\n" + \
        "for i1 (0,64)\n" + \
        "  for i2 (0,112)\n" + \
        "    for i3 (0,112)\n" + \
        "      compute = ...\n"


if __name__ == "__main__":
    test_split_fuse_reorder_annotation()
    test_compute_at_root_inline()
