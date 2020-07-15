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


def test_split_fuse_reorder():
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

if __name__ == "__main__":
    test_split_fuse_reorder()
