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
from tvm import ansor, te
import topi

from test_ansor_common import matmul_ansor_test, conv2d_nchw_bn_relu


def test_split_fuse_reorder_annotation():
    A, B, C = matmul_ansor_test(512, 512, 512)
    dag = ansor.ComputeDAG([A, B, C])
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
    s1.bind_thread(C, i1, "blockIdx.x")
    s1.bind_thread(C, i2, "vthread")
    s1.bind_thread(C, i3, "threadIdx.y")


def test_follow_split_follow_fused_split():
    A, B, C = matmul_ansor_test(512, 512, 512)
    dag = ansor.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()

    C_global = s0.cache_write(C, "global")

    its0 = s0.split(C, s0[C].iters[0], [4, 2, 8, 4], True)
    split_step0 = s0.transform_steps_size() - 1
    for level in range(1, 6):
        tmp = s0.copy()
        tmp.follow_split(C_global, tmp[C_global].iters[0], split_step0, level)
        for i in range(0, level):
            assert tmp[C].iters[i].range.extent == \
                   tmp[C_global].iters[i].range.extent

    its1 = s0.split(C, s0[C].iters[5], [2, 2, 4, 8])
    split_step1 = s0.transform_steps_size() - 1
    its = []
    for i0, i1 in zip(its0, its1):
        its.append(i0)
        its.append(i1)
    s0.reorder(C, its)
    for i in range(0, 5):
        s0.fuse(C, [s0[C].iters[i], s0[C].iters[i + 1]])

    for level in range(0, 4):
        tmp = s0.copy()
        tmp.follow_fused_split(C_global, tmp[C_global].iters[0],
                               [split_step0, split_step1], level, False)
        assert tmp[C].iters[level + 1].range.extent == \
               tmp[C_global].iters[0].range.extent

    for level in range(0, 4):
        tmp = s0.copy()
        tmp.follow_fused_split(C_global, tmp[C_global].iters[0],
                               [split_step0, split_step1], level, True)
        assert tmp[C].iters[level + 1].range.extent == \
               tmp[C_global].iters[1].range.extent


def test_compute_at_root_inline():
    dag = ansor.ComputeDAG(conv2d_nchw_bn_relu(1, 224, 224, 3, 64, 7, 2, 3))
    s0 = dag.get_init_state()

    # data, padding, kernel = 0, 1, 2
    conv = s0.stage_tensors[3]
    # bias = 4
    bias_add = s0.stage_tensors[5]
    # bn_scale = 6
    bn_mul = s0.stage_tensors[7]
    # bn_offset = 8
    bn_add = s0.stage_tensors[9]
    relu = s0.stage_tensors[10]

    s0.compute_inline(bn_add)
    s0.compute_inline(bn_mul)
    s0.compute_inline(bias_add)
    s0.compute_at(conv, relu, s0[relu].iters[2])
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


def test_cache_read_write():
    N, H, W, CO, CI, KH, KW, strides, padding = 4, 7, 7, 512, 512, 3, 3, (
        1, 1), (1, 1)

    data = te.placeholder((N, CI, H, W), name='Data')
    kernel_data = te.placeholder((CO, CI, KH, KW), name='Kernel_data')
    k0, k1 = te.compute(kernel_data.shape,
                        lambda *i: (kernel_data(*i)+1, kernel_data(*i)/2),
                        name='Kernel_split')
    kernel = te.compute(kernel_data.shape,
                        lambda *i: k0(*i) + k1(*i),
                        name='Kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1)
    relu = topi.nn.relu(conv)
    add = topi.add(data, relu)

    dag = ansor.ComputeDAG([data, kernel_data, add])
    s0 = dag.get_init_state()

    pad_temp = s0.stage_tensors[1]
    kernel_split = s0.stage_tensors[3]

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
    assert str(s0) == \
        "Placeholder: Data, Kernel_data\n" + \
        "for i0 (0,4)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,9)\n" + \
        "      for i3 (0,9)\n" + \
        "        pad_temp = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel_split = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel = ...\n" + \
        "for nn (0,4)\n" + \
        "  for ff (0,512)\n" + \
        "    for yy (0,7)\n" + \
        "      for xx (0,7)\n" + \
        "        for nn_c (None)\n" + \
        "          for ff_c (None)\n" + \
        "            for yy_c (None)\n" + \
        "              for xx_c (None)\n" + \
        "                for rc (None)\n" + \
        "                  for ax0 (None)\n" + \
        "                    for ax1 (None)\n" + \
        "                      for ax2 (None)\n" + \
        "                        for ax3 (None)\n" + \
        "                          Kernel.global = ...\n" + \
        "                  for ry (None)\n" + \
        "                    for rx (None)\n" + \
        "                      compute.global = ...\n" + \
        "        compute = ...\n" + \
        "for ax0.0 (0,2)\n" + \
        "  for ax1 (0,512)\n" + \
        "    for ax0.1 (0,2)\n" + \
        "      for ax2 (0,7)\n" + \
        "        for ax3 (0,7)\n" + \
        "          T_add = ...\n"

    # 3: two level cache_read with compute_at
    #    preparing for GPU's shared memory & local memory
    pad_temp_global = s0.cache_read(pad_temp, "global", [conv_global])
    pad_temp_shared = s0.cache_read(pad_temp_global, "shared", [conv_global])
    s0.compute_at(pad_temp_global, conv_global, s0[conv_global].iters[2])
    s0.compute_at(pad_temp_shared, conv_global, s0[conv_global].iters[4])

    # 4: cache_read with multi readers
    #    This stage cannot be compute at to its consumer
    s0.cache_read(data, "global", [pad_temp, add])
    assert str(s0) == \
        "Placeholder: Data, Kernel_data\n" + \
        "for ax0 (0,4)\n" + \
        "  for ax1 (0,512)\n" + \
        "    for ax2 (0,7)\n" + \
        "      for ax3 (0,7)\n" + \
        "        Data.global = ...\n" + \
        "for i0 (0,4)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,9)\n" + \
        "      for i3 (0,9)\n" + \
        "        pad_temp = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel_split = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel = ...\n" + \
        "for nn (0,4)\n" + \
        "  for ff (0,512)\n" + \
        "    for yy (0,7)\n" + \
        "      for xx (0,7)\n" + \
        "        for nn_c (None)\n" + \
        "          for ff_c (None)\n" + \
        "            for yy_c (None)\n" + \
        "              for ax0 (None)\n" + \
        "                for ax1 (None)\n" + \
        "                  for ax2 (None)\n" + \
        "                    for ax3 (None)\n" + \
        "                      pad_temp.global = ...\n" + \
        "              for xx_c (None)\n" + \
        "                for rc (None)\n" + \
        "                  for ax0 (None)\n" + \
        "                    for ax1 (None)\n" + \
        "                      for ax2 (None)\n" + \
        "                        for ax3 (None)\n" + \
        "                          Kernel.global = ...\n" + \
        "                  for ax0 (None)\n" + \
        "                    for ax1 (None)\n" + \
        "                      for ax2 (None)\n" + \
        "                        for ax3 (None)\n" + \
        "                          pad_temp.global.shared = ...\n" + \
        "                  for ry (None)\n" + \
        "                    for rx (None)\n" + \
        "                      compute.global = ...\n" + \
        "        compute = ...\n" + \
        "for ax0.0 (0,2)\n" + \
        "  for ax1 (0,512)\n" + \
        "    for ax0.1 (0,2)\n" + \
        "      for ax2 (0,7)\n" + \
        "        for ax3 (0,7)\n" + \
        "          T_add = ...\n"

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
    # Seems there's bug with the input/output tensor. Such multi outputs case
    # should be unusual, so we make some hack on DoCacheWrite
    # To be fixed in the future
    s0.cache_write(kernel_split, "global")
    assert str(s0) == \
        "Placeholder: Data, Kernel_data\n" + \
        "for ax0 (0,4)\n" + \
        "  for ax1 (0,512)\n" + \
        "    for ax2 (0,7)\n" + \
        "      for ax3 (0,7)\n" + \
        "        Data.global = ...\n" + \
        "for i0 (0,4)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,9)\n" + \
        "      for i3 (0,9)\n" + \
        "        pad_temp = ...\n" + \
        "for i0_c (0,512)\n" + \
        "  for i1_c (0,512)\n" + \
        "    for i2_c (0,3)\n" + \
        "      for i3_c (0,3)\n" + \
        "        Kernel_split.global = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel_split = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel_split = ...\n" + \
        "for i0 (0,512)\n" + \
        "  for i1 (0,512)\n" + \
        "    for i2 (0,3)\n" + \
        "      for i3 (0,3)\n" + \
        "        Kernel = ...\n" + \
        "for nn (0,4)\n" + \
        "  for ff (0,512)\n" + \
        "    for yy (0,7)\n" + \
        "      for xx (0,7)\n" + \
        "        for nn_c (None)\n" + \
        "          for ff_c (None)\n" + \
        "            for yy_c (None)\n" + \
        "              for ax0 (None)\n" + \
        "                for ax1 (None)\n" + \
        "                  for ax2 (None)\n" + \
        "                    for ax3 (None)\n" + \
        "                      pad_temp.global = ...\n" + \
        "              for xx_c (None)\n" + \
        "                for rc (None)\n" + \
        "                  for ax0 (None)\n" + \
        "                    for ax1 (None)\n" + \
        "                      for ax2 (None)\n" + \
        "                        for ax3 (None)\n" + \
        "                          Kernel.global = ...\n" + \
        "                  for ax0 (None)\n" + \
        "                    for ax1 (None)\n" + \
        "                      for ax2 (None)\n" + \
        "                        for ax3 (None)\n" + \
        "                          pad_temp.global.shared = ...\n" + \
        "                  for ry (None)\n" + \
        "                    for rx (None)\n" + \
        "                      compute.global = ...\n" + \
        "        compute = ...\n" + \
        "for ax0.0 (0,2)\n" + \
        "  for ax1 (0,512)\n" + \
        "    for ax0.1 (0,2)\n" + \
        "      for ax2 (0,7)\n" + \
        "        for ax3 (0,7)\n" + \
        "          T_add = ...\n"


def test_rfactor():
    A, B, C = matmul_ansor_test(8, 8, 512)
    dag = ansor.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()

    ko, ki = s0.split(C, s0[C].iters[2], [16])

    s1 = s0.copy()
    s1.rfactor(C, ko, 2)
    assert str(s1) == \
        "Placeholder: A, B\n" + \
        "for i (0,8)\n" + \
        "  for j (0,8)\n" + \
        "    for k_o (0,32)\n" + \
        "      for k_i (0,16)\n" + \
        "        C.rf = ...\n" + \
        "for ax0 (0,8)\n" + \
        "  for ax1 (0,8)\n" + \
        "    for k_o_v (0,32)\n" + \
        "      C.repl = ...\n"

    s2 = s0.copy()
    s2.rfactor(C, ki, 2)
    assert str(s2) == \
        "Placeholder: A, B\n" + \
        "for i (0,8)\n" + \
        "  for j (0,8)\n" + \
        "    for k_i (0,16)\n" + \
        "      for k_o (0,32)\n" + \
        "        C.rf = ...\n" + \
        "for ax0 (0,8)\n" + \
        "  for ax1 (0,8)\n" + \
        "    for k_i_v (0,16)\n" + \
        "      C.repl = ...\n"


def vcf_init_common():
    A, B, C = matmul_ansor_test(512, 512, 512)
    dag = ansor.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()
    B_shared = s0.cache_read(B, "shared", [C])
    B_local = s0.cache_read(B_shared, "local", [C])
    A_shared = s0.cache_read(A, "shared", [C])
    A_local = s0.cache_read(A_shared, "local", [C])

    return A_shared, A_local, B_shared, B_local, C, dag, s0


def vcf_check_common(dag, state):
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
    A_shared, A_local, B_shared, B_local, C, dag, s0 = vcf_init_common()

    its0 = s0.split(C, s0[C].iters[0], [1, 8, 2, 4])
    its1 = s0.split(C, s0[C].iters[5], [2, 8, 2, 4])
    its2 = s0.split(C, s0[C].iters[10], [8, 8])
    s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its2[0],
                   its2[1], its0[3], its1[3], its2[2], its0[4], its1[4]])
    s0.fuse(C, [s0[C].iters[0], s0[C].iters[1]])
    s0.bind_thread(C, s0[C].iters[0], "blockIdx.x")
    s0.fuse(C, [s0[C].iters[1], s0[C].iters[2]])
    s0.bind_thread(C, s0[C].iters[1], "vthread")
    s0.fuse(C, [s0[C].iters[2], s0[C].iters[3]])
    s0.bind_thread(C, s0[C].iters[2], "threadIdx.x")
    s0.vectorize(C, its1[4])

    s0.compute_at(B_shared, C, s0[C].iters[3])
    fused_it = s0.fuse(B_shared, s0[B_shared].iters[:])
    its = s0.split(B_shared, fused_it, [64, 4])
    s0.bind_thread(B_shared, its[1], "threadIdx.x")
    s0.vectorize(B_shared, its[2])
    s0.compute_at(B_local, C, s0[C].iters[4])
    fused_it = s0.fuse(B_local, s0[B_local].iters[:])
    its = s0.split(B_local, fused_it, [4])
    s0.vectorize(B_local, its[1])

    s0.compute_at(A_shared, C, s0[C].iters[3])
    fused_it = s0.fuse(A_shared, s0[A_shared].iters[:])
    its = s0.split(A_shared, fused_it, [64, 4])
    s0.bind_thread(A_shared, its[1], "threadIdx.x")
    s0.vectorize(A_shared, its[2])
    s0.compute_at(A_local, C, s0[C].iters[4])
    fused_it = s0.fuse(A_local, s0[A_local].iters[:])
    its = s0.split(A_local, fused_it, [4])
    s0.vectorize(A_local, its[1])

    vcf_check_common(dag, s0)


def test_vectorized_cooperative_fetching_xy():
    A_shared, A_local, B_shared, B_local, C, dag, s0 = vcf_init_common()

    its0 = s0.split(C, s0[C].iters[0], [1, 8, 2, 4])
    its1 = s0.split(C, s0[C].iters[5], [2, 8, 2, 4])
    its2 = s0.split(C, s0[C].iters[10], [8, 8])
    s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its2[0],
                   its2[1], its0[3], its1[3], its2[2], its0[4], its1[4]])
    s0.fuse(C, [s0[C].iters[0], s0[C].iters[1]])
    s0.bind_thread(C, s0[C].iters[0], "blockIdx.x")
    s0.fuse(C, [s0[C].iters[1], s0[C].iters[2]])
    s0.bind_thread(C, s0[C].iters[1], "vthread")
    s0.bind_thread(C, s0[C].iters[2], "threadIdx.x")
    s0.bind_thread(C, s0[C].iters[3], "threadIdx.y")
    s0.vectorize(C, its1[4])

    s0.compute_at(B_shared, C, s0[C].iters[4])
    fused_it = s0.fuse(B_shared, s0[B_shared].iters[:])
    its = s0.split(B_shared, fused_it, [8, 8, 4])
    s0.bind_thread(B_shared, its[1], "threadIdx.x")
    s0.bind_thread(B_shared, its[2], "threadIdx.y")
    s0.vectorize(B_shared, its[3])
    s0.compute_at(B_local, C, s0[C].iters[5])
    fused_it = s0.fuse(B_local, s0[B_local].iters[:])
    its = s0.split(B_local, fused_it, [4])
    s0.vectorize(B_local, its[1])

    s0.compute_at(A_shared, C, s0[C].iters[4])
    fused_it = s0.fuse(A_shared, s0[A_shared].iters[:])
    its = s0.split(A_shared, fused_it, [8, 8, 4])
    s0.bind_thread(A_shared, its[1], "threadIdx.x")
    s0.bind_thread(A_shared, its[2], "threadIdx.y")
    s0.vectorize(A_shared, its[3])
    s0.compute_at(A_local, C, s0[C].iters[5])
    fused_it = s0.fuse(A_local, s0[A_local].iters[:])
    its = s0.split(A_local, fused_it, [4])
    s0.vectorize(A_local, its[1])

    vcf_check_common(dag, s0)


@tvm._ffi.register_func
def test_intrin_gemv():
    m = 16
    l = 64
    a = te.placeholder((l,), name='a')
    b = te.placeholder((l, m), name='b')
    k = te.reduce_axis((0, l), name='k')
    c = te.compute((m,), lambda i: te.sum(a[k] * b[k, i], axis=k), name='c')
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A",
                             offset_factor=1, strides=[1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B",
                             offset_factor=1, strides=[te.var("s0"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C",
                             offset_factor=1, strides=[1])
    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(tvm.tir.call_extern("float32", "gemv_update",
                                    cc.access_ptr("w"),
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r")))
        return ib.get()
    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def test_tensorize():
    A, B, C = matmul_ansor_test(1024, 512, 64)
    dag = ansor.ComputeDAG([A, B, C])
    s0 = dag.get_init_state()

    its = s0.split(C, s0[C].iters[1], [16])
    s0.tensorize(C, its[1], "test_intrin_gemv")

    sch, tensors = dag.apply_steps_from_state(s0)
    tvm.lower(sch, tensors, simple_mode=True)

if __name__ == "__main__":
    test_split_fuse_reorder_annotation()
    test_follow_split_follow_fused_split()
    test_compute_at_root_inline()
    test_cache_read_write()
    test_rfactor()
    test_vectorized_cooperative_fetching_x()
    test_vectorized_cooperative_fetching_xy()
    test_tensorize()
