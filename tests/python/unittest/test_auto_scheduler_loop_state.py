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
from tvm import topi

from tvm.testing.auto_scheduler import (
    matmul_auto_scheduler_test,
    conv2d_nchw_bn_relu_auto_scheduler_test,
)


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
    dag = auto_scheduler.ComputeDAG(
        conv2d_nchw_bn_relu_auto_scheduler_test(
            N=1, H=224, W=224, CI=3, CO=64, kernel_size=7, strides=2, padding=3
        )
    )
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


def test_cache_read_write():
    N, H, W, CO, CI, KH, KW, strides, padding = 4, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)

    data = te.placeholder((N, CI, H, W), name="Data")
    kernel_data = te.placeholder((CO, CI, KH, KW), name="Kernel_data")
    k0, k1 = te.compute(
        kernel_data.shape,
        lambda *i: (kernel_data(*i) + 1, kernel_data(*i) / 2),
        name="Kernel_split",
    )
    kernel = te.compute(kernel_data.shape, lambda *i: k0(*i) + k1(*i), name="Kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1)
    relu = topi.nn.relu(conv)
    add = topi.add(data, relu)

    dag = auto_scheduler.ComputeDAG([data, kernel_data, add])
    s0 = dag.get_init_state()

    pad_temp = s0.stage_ops[1]
    kernel_split = s0.stage_ops[3]

    # 0: init state
    ori_its = s0[add].iters
    its = s0.split(add, s0[add].iters[0], [2])
    s0.reorder(add, [its[0], ori_its[1], its[1], ori_its[2], ori_its[3]])
    s0.compute_inline(relu)

    # 1: simple cache_write with compute_at
    conv_global = s0.cache_write(conv, "global")
    s0.compute_at(conv_global, conv, s0[conv].iters[3])

    # 2: simple cache_read with compute_at
    kernel_global = s0.cache_read(kernel, "global", [conv_global])
    s0.compute_at(kernel_global, conv_global, s0[conv_global].iters[4])
    """
        Placeholder: Data, Kernel_data
        for i0 (0,4)
          for i1 (0,512)
            for i2 (0,9)
              for i3 (0,9)
                pad_temp = ...
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel_split = ...
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel = ...
        for nn (0,4)
          for ff (0,512)
            for yy (0,7)
              for xx (0,7)
                for nn_c (None)
                  for ff_c (None)
                    for yy_c (None)
                      for xx_c (None)
                        for rc (None)
                          for ax0 (None)
                            for ax1 (None)
                              for ax2 (None)
                                for ax3 (None)
                                  Kernel.global = ...
                          for ry (None)
                            for rx (None)
                              compute.global = ...
                compute = ...
        for ax0.0 (0,2)
          for ax1 (0,512)
            for ax0.1 (0,2)
              for ax2 (0,7)
                for ax3 (0,7)
                  T_add = ...
    """
    s1 = dag.infer_bound_from_state(s0)
    assert s1[conv].iters[0].range.extent == 4
    assert s1[conv].iters[1].range.extent == 512
    assert s1[conv].iters[2].range.extent == 7
    assert s1[conv].iters[3].range.extent == 7
    assert s1[kernel_global].iters[0].range.extent == 1
    assert s1[kernel_global].iters[1].range.extent == 1
    assert s1[kernel_global].iters[2].range.extent == 3
    assert s1[kernel_global].iters[3].range.extent == 3
    assert s1[conv_global].iters[0].range.extent == 1
    assert s1[conv_global].iters[1].range.extent == 1
    assert s1[conv_global].iters[2].range.extent == 1
    assert s1[conv_global].iters[3].range.extent == 1
    assert s1[conv_global].iters[4].range.extent == 512
    assert s1[conv_global].iters[5].range.extent == 3
    assert s1[conv_global].iters[6].range.extent == 3

    # 3: two level cache_read with compute_at
    #    preparing for GPU's shared memory & local memory
    pad_temp_global = s0.cache_read(pad_temp, "global", [conv_global])
    pad_temp_shared = s0.cache_read(pad_temp_global, "shared", [conv_global])
    s0.compute_at(pad_temp_global, conv_global, s0[conv_global].iters[2])
    s0.compute_at(pad_temp_shared, conv_global, s0[conv_global].iters[4])

    # 4: cache_read with multi readers
    #    This stage cannot be compute at to its consumer
    s0.cache_read(data, "global", [pad_temp, add])
    """
        Placeholder: Data, Kernel_data
        for ax0 (0,4)
          for ax1 (0,512)
            for ax2 (0,7)
              for ax3 (0,7)
                Data.global = ...
        for i0 (0,4)
          for i1 (0,512)
            for i2 (0,9)
              for i3 (0,9)
                pad_temp = ...
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel_split = ...
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel = ...
        for nn (0,4)
          for ff (0,512)
            for yy (0,7)
              for xx (0,7)
                for nn_c (None)
                  for ff_c (None)
                    for yy_c (None)
                      for ax0 (None)
                        for ax1 (None)
                          for ax2 (None)
                            for ax3 (None)
                              pad_temp.global = ...
                      for xx_c (None)
                        for rc (None)
                          for ax0 (None)
                            for ax1 (None)
                              for ax2 (None)
                                for ax3 (None)
                                  Kernel.global = ...
                          for ax0 (None)
                            for ax1 (None)
                              for ax2 (None)
                                for ax3 (None)
                                  pad_temp.global.shared = ...
                          for ry (None)
                            for rx (None)
                              compute.global = ...
                compute = ...
        for ax0.0 (0,2)
          for ax1 (0,512)
            for ax0.1 (0,2)
              for ax2 (0,7)
                for ax3 (0,7)
                  T_add = ...
    """
    s1 = dag.infer_bound_from_state(s0)
    assert s1[conv].iters[0].range.extent == 4
    assert s1[conv].iters[1].range.extent == 512
    assert s1[conv].iters[2].range.extent == 7
    assert s1[conv].iters[3].range.extent == 7
    assert s1[kernel_global].iters[0].range.extent == 1
    assert s1[kernel_global].iters[1].range.extent == 1
    assert s1[kernel_global].iters[2].range.extent == 3
    assert s1[kernel_global].iters[3].range.extent == 3
    assert s1[conv_global].iters[0].range.extent == 1
    assert s1[conv_global].iters[1].range.extent == 1
    assert s1[conv_global].iters[2].range.extent == 1
    assert s1[conv_global].iters[3].range.extent == 1
    assert s1[conv_global].iters[4].range.extent == 512
    assert s1[conv_global].iters[5].range.extent == 3
    assert s1[conv_global].iters[6].range.extent == 3
    assert s1[pad_temp_global].iters[0].range.extent == 1
    assert s1[pad_temp_global].iters[1].range.extent == 512
    assert s1[pad_temp_global].iters[2].range.extent == 3
    assert s1[pad_temp_global].iters[3].range.extent == 3
    assert s1[pad_temp_shared].iters[0].range.extent == 1
    assert s1[pad_temp_shared].iters[1].range.extent == 1
    assert s1[pad_temp_shared].iters[2].range.extent == 3
    assert s1[pad_temp_shared].iters[3].range.extent == 3

    # 5: cache_write with multi outputs
    # TVM's cache_write actually has a bug with this case:
    #
    # After schedule.cache_write, TVM generate one new stage:
    #   From: kernel_data -> kernel_split -> kernel
    #   To:   kernel_data -> kernel_split_global -> kernel_split -> kernel
    #
    # But with topo sort analyse, we get:
    #  //   kernel_data -> kernel_split_global -> kernel_split -> kernel
    #         \                                                /
    #          ----------------> kernel_split ---------------->
    #
    # TODO(jcf94): Seems there's bug with the input/output tensor. Such multi outputs case
    # should be unusual, so we make some hack on DoCacheWrite. This should be fixed later.
    kernel_split_global = s0.cache_write(kernel_split, "global")
    """
        Placeholder: Data, Kernel_data
        for ax0 (0,4)
          for ax1 (0,512)
            for ax2 (0,7)
              for ax3 (0,7)
                Data.global = ...
        for i0 (0,4)
          for i1 (0,512)
            for i2 (0,9)
              for i3 (0,9)
                pad_temp = ...
        for i0_c (0,512)
          for i1_c (0,512)
            for i2_c (0,3)
              for i3_c (0,3)
                Kernel_split.global = ...
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel_split = ...
        (******* Bug here, there should not be two kernel_split stage *******)
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel_split = ...
        (******* Bug here, there should not be two kernel_split stage *******)
        for i0 (0,512)
          for i1 (0,512)
            for i2 (0,3)
              for i3 (0,3)
                Kernel = ...
        for nn (0,4)
          for ff (0,512)
            for yy (0,7)
              for xx (0,7)
                for nn_c (None)
                  for ff_c (None)
                    for yy_c (None)
                      for ax0 (None)
                        for ax1 (None)
                          for ax2 (None)
                            for ax3 (None)
                              pad_temp.global = ...
                      for xx_c (None)
                        for rc (None)
                          for ax0 (None)
                            for ax1 (None)
                              for ax2 (None)
                                for ax3 (None)
                                  Kernel.global = ...
                          for ax0 (None)
                            for ax1 (None)
                              for ax2 (None)
                                for ax3 (None)
                                  pad_temp.global.shared = ...
                          for ry (None)
                            for rx (None)
                              compute.global = ...
                compute = ...
        for ax0.0 (0,2)
          for ax1 (0,512)
            for ax0.1 (0,2)
              for ax2 (0,7)
                for ax3 (0,7)
                  T_add = ...
    """
    assert len(s0[kernel_split].iters) == len(s0[kernel_split_global].iters)
    for it0, it1 in zip(s0[kernel_split].iters, s0[kernel_split_global].iters):
        assert it0.range == it1.range


def test_follow_split_follow_fused_split():
    A, B, C = matmul_auto_scheduler_test(512, 512, 512)
    dag = auto_scheduler.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()

    C_global = s0.cache_write(C, "global")
    its0 = s0.split(C, s0[C].iters[0], [4, 2, 8, 4], True)
    split_step0 = len(s0.transform_steps) - 1
    for level in range(1, 6):
        tmp = s0.copy()
        tmp.follow_split(C_global, tmp[C_global].iters[0], split_step0, level)
        for i in range(0, level):
            assert tmp[C].iters[i].range.extent == tmp[C_global].iters[i].range.extent

    its1 = s0.split(C, s0[C].iters[5], [2, 2, 4, 8])
    split_step1 = len(s0.transform_steps) - 1
    its = []
    for i0, i1 in zip(its0, its1):
        its.append(i0)
        its.append(i1)
    s0.reorder(C, its)
    for i in range(0, 5):
        s0.fuse(C, [s0[C].iters[i], s0[C].iters[i + 1]])

    for level in range(0, 4):
        tmp = s0.copy()
        tmp.follow_fused_split(
            C_global, tmp[C_global].iters[0], [split_step0, split_step1], level, False
        )
        assert tmp[C].iters[level + 1].range.extent == tmp[C_global].iters[0].range.extent

    for level in range(0, 4):
        tmp = s0.copy()
        tmp.follow_fused_split(
            C_global, tmp[C_global].iters[0], [split_step0, split_step1], level, True
        )
        assert tmp[C].iters[level + 1].range.extent == tmp[C_global].iters[1].range.extent


def test_rfactor():
    A, B, C = matmul_auto_scheduler_test(8, 8, 512)
    dag = auto_scheduler.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()

    ko, ki = s0.split(C, s0[C].iters[2], [16])

    s1 = s0.copy()
    C_r = s1.rfactor(C, ko, 2)
    """
        Placeholder: A, B
        for i (0,8)
          for j (0,8)
            for k_o (0,32)
              for k_i (0,16)
                C.rf = ...
        for ax0 (0,8)
          for ax1 (0,8)
            for k_o_v (0,32)
              C.repl = ...
    """
    assert s1[C_r].iters[0].range.extent == 8
    assert s1[C_r].iters[1].range.extent == 8
    assert s1[C_r].iters[2].range.extent == 32
    assert s1[C_r].iters[3].range.extent == 16
    assert s1[C].iters[0].range.extent == 8
    assert s1[C].iters[1].range.extent == 8
    assert s1[C].iters[2].range.extent == 32

    s2 = s0.copy()
    C_r = s2.rfactor(C, ki, 2)
    """
        Placeholder: A, B
        for i (0,8)
          for j (0,8)
            for k_i (0,16)
              for k_o (0,32)
                C.rf = ...
        for ax0 (0,8)
          for ax1 (0,8)
            for k_i_v (0,16)
              C.repl = ...
    """
    assert s2[C_r].iters[0].range.extent == 8
    assert s2[C_r].iters[1].range.extent == 8
    assert s2[C_r].iters[2].range.extent == 16
    assert s2[C_r].iters[3].range.extent == 32
    assert s2[C].iters[0].range.extent == 8
    assert s2[C].iters[1].range.extent == 8
    assert s2[C].iters[2].range.extent == 16


if __name__ == "__main__":
    test_split_fuse_reorder_annotation()
    test_compute_at_root_inline()
    test_cache_read_write()
    test_follow_split_follow_fused_split()
    test_rfactor()
