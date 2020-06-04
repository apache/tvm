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
import tvm
from tvm import te
from tvm import ansor
import topi


def matmul_nkkm(N, M, K):
    A = te.placeholder((N, K), name='A')
    B = te.placeholder((K, M), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(
        A[i][k] * B[k][j], axis=[k]), name='C')

    return [A, B, C]


def conv2d_nchw_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, CI, H, W), name='Data')
    kernel = te.placeholder((CO, CI, kernel_size, kernel_size), name='Kernel')
    bias = te.placeholder((CO, 1, 1), name='Bias')
    bn_scale = te.placeholder((CO, 1, 1), name='Bn_scale')
    bn_offset = te.placeholder((CO, 1, 1), name='Bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0],
                      name='Bias_add')
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0],
                      name='Bn_mul')
    conv = te.compute((N, CO, OH, OW),
                      lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0],
                      name='Bn_add')
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]


def test_compute_dag_basic():
    dag = ansor.ComputeDAG(conv2d_nchw_bn_relu(1, 224, 224, 3, 64, 7, 2, 3))

    print(dag)
    print(dag.access_analyzer)
    print(dag.get_init_state())


def test_state_split_fuse_reorder():
    dag = ansor.ComputeDAG(matmul_nkkm(512, 512, 512))
    s0 = dag.get_init_state()
    s1 = s0
    ti = s0.stage(2).iterator(0)
    tj = s0.stage(2).iterator(1)
    tk = s0.stage(2).iterator(2)

    assert ti.range.extent == 512

    s0, its = s0.split(2, ti, [16])
    tio = its[0]
    tii = its[1]
    assert s0.stage(2).iterator(0).range.extent == 32
    assert s0.stage(2).iterator(1).range.extent == 16

    s0, its = s0.split(2, tj, [8])
    tjo = its[0]
    tji = its[1]
    assert s0.stage(2).iterator(2).range.extent == 64
    assert s0.stage(2).iterator(3).range.extent == 8

    s0 = s0.reorder(2, [tio, tjo, tk, tji, tii])
    assert s0.stage(2).iterator(2).range.extent == 512

    s0, res_it = s0.fuse(2, [tio, tjo])
    assert res_it.range.extent == 2048

    s1, _ = s1.split(2, ti, [8, 2])
    s1, _ = s1.split(2, tj, [32, 8], False)
    assert s1.stage(2).iterator(0).range.extent == 32
    assert s1.stage(2).iterator(1).range.extent == 8
    assert s1.stage(2).iterator(2).range.extent == 2
    assert s1.stage(2).iterator(3).range.extent == 32
    assert s1.stage(2).iterator(4).range.extent == 8
    assert s1.stage(2).iterator(5).range.extent == 2


def test_state_compute_at_root_inline():
    dag = ansor.ComputeDAG(conv2d_nchw_bn_relu(1, 224, 224, 3, 64, 7, 2, 3))

    # data, padding, kernel = 0, 1, 2
    conv = 3
    # bias = 4
    bias_add = 5
    # bn_scale = 6
    bn_mul = 7
    # bn_offset = 8
    bn_add, relu = 9, 10

    s0 = dag.get_init_state()
    s0 = s0.compute_inline(bn_add)
    s0 = s0.compute_inline(bn_mul)
    s0 = s0.compute_inline(bias_add)
    s0 = s0.compute_at(conv, relu, s0.stage(relu).iterator(2))
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

    s0 = s0.compute_root(conv)
    s0 = s0.compute_root(bn_mul)
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


def test_state_cache_read_write():
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
    out = topi.add(data, relu)

    dag = ansor.ComputeDAG([data, kernel_data, out])
    data, pad_temp, kernel_data, kernel_split, kernel, conv, relu, add = 0, 1, 2, 3, 4, 5, 6, 7

    # 0: init state
    s0 = dag.get_init_state()
    ori_its = s0.stage(add).iterators()
    s0, its = s0.split(add, s0.stage(add).iterator(0), [2])
    s0 = s0.reorder(add, [its[0], ori_its[1], its[1], ori_its[2], ori_its[3]])
    s0 = s0.compute_inline(relu)

    # 1: simple cache_write with compute_at
    s0, conv_global = s0.cache_write(conv, "global", dag)
    conv += 1
    relu += 1
    add += 1
    s0 = s0.compute_at(conv_global, conv, s0.stage(conv).iterator(3))

    # 2: simple cache_read with compute_at
    s0, kernel_global = s0.cache_read(kernel, "global", [conv_global], dag)
    conv_global += 1
    conv += 1
    relu += 1
    add += 1
    s0 = s0.compute_at(kernel_global, conv_global,
                       s0.stage(conv_global).iterator(4))
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
    s0, pad_temp_global = s0.cache_read(pad_temp, "global", [conv_global], dag)
    kernel_data += 1
    kernel_split += 1
    kernel += 1
    kernel_global += 1
    conv_global += 1
    conv += 1
    relu += 1
    add += 1
    s0, pad_temp_shared = s0.cache_read(
        pad_temp_global, "shared", [conv_global], dag)
    kernel_data += 1
    kernel_split += 1
    kernel += 1
    kernel_global += 1
    conv_global += 1
    conv += 1
    relu += 1
    add += 1
    s0 = s0.compute_at(pad_temp_global, conv_global,
                       s0.stage(conv_global).iterator(2))
    s0 = s0.compute_at(pad_temp_shared, conv_global,
                       s0.stage(conv_global).iterator(4))

    # 4: cache_read with multi readers
    #    This stage cannot be compute at to its consumer
    s0, data_global = s0.cache_read(data, "global", [pad_temp, add], dag)
    pad_temp += 1
    pad_temp_global += 1
    pad_temp_shared += 1
    kernel_data += 1
    kernel_split += 1
    kernel += 1
    kernel_global += 1
    conv_global += 1
    conv += 1
    relu += 1
    add += 1
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
    # See tests/cpp/ansor_test.cc for more information
    s0, _ = s0.cache_write(kernel_split, "global", dag)
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


def test_follow_split_follow_fused_split():
    dag = ansor.ComputeDAG(matmul_nkkm(512, 512, 512))
    s0 = dag.get_init_state()
    C = 2

    s0, C_global = s0.cache_write(C, "global", dag)
    C += 1

    s0, its0 = s0.split(C, s0.stage(C).iterator(0), [4, 2, 8, 4], True)
    split_step0 = s0.transform_steps_size() - 1
    for level in range(1, 6):
        tmp = s0
        tmp, _ = tmp.follow_split(C_global, tmp.stage(
            C_global).iterator(0), split_step0, level)
        for i in range(0, level):
            assert tmp.stage(C).iterator(i).range.extent == \
                tmp.stage(C_global).iterator(i).range.extent

    s0, its1 = s0.split(C, s0.stage(C).iterator(5), [2, 2, 4, 8])
    split_step1 = s0.transform_steps_size() - 1
    its = []
    for i0, i1 in zip(its0, its1):
        its.append(i0)
        its.append(i1)
    s0 = s0.reorder(C, its)
    for i in range(0, 5):
        s0, _ = s0.fuse(C, [s0.stage(C).iterator(i),
                            s0.stage(C).iterator(i+1)])
    for level in range(0, 4):
        tmp = s0
        tmp, _ = tmp.follow_fused_split(C_global, tmp.stage(C_global).iterator(0),
                                        [split_step0, split_step1], level, False)
        assert tmp.stage(C).iterator(level+1).range.extent == \
            tmp.stage(C_global).iterator(0).range.extent
    for level in range(0, 4):
        tmp = s0
        tmp, _ = tmp.follow_fused_split(C_global, tmp.stage(C_global).iterator(0),
                                        [split_step0, split_step1], level, True)
        assert tmp.stage(C).iterator(level+1).range.extent == \
            tmp.stage(C_global).iterator(1).range.extent


def test_rfactor():
    pass


def test_measure_local_builder_runner():
    dag = ansor.ComputeDAG(matmul_nkkm(512, 512, 512))

    s0 = dag.get_init_state()
    A, B, C = 0, 1, 2
    s0, C_global = s0.cache_write(C, "global", dag)
    C += 1
    s0, its0 = s0.split(C, s0.stage(C).iterator(0), [4, 8, 8])
    s0, its1 = s0.split(C, s0.stage(C).iterator(4), [8, 4, 4])
    s0 = s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2],
                        its0[3], its1[3]])
    s0 = s0.compute_at(C_global, C, s0.stage(C).iterator(3))
    s0, _ = s0.split(C_global, s0.stage(C_global).iterator(2), [16])
    s0, B_global = s0.cache_read(B, "global", [C_global], dag)
    C += 1
    C_global += 1
    s0 = s0.compute_at(B_global, C_global, s0.stage(C_global).iterator(0))
    s0, A_global = s0.cache_read(A, "global", [C_global], dag)
    B += 1
    B_global += 1
    C += 1
    C_global += 1
    s0 = s0.compute_at(A_global, C_global, s0.stage(C_global).iterator(2))

    tgt = tvm.target.create("llvm")
    task = ansor.SearchTask(dag, "test", tgt)

    minp = ansor.MeasureInput(task, s0)
    local_builder = ansor.LocalBuilder()
    local_runner = ansor.LocalRunner()

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = local_runner.run([minp], bress)
    assert mress[0].error_no == 0


def test_search_basic():
    dag = ansor.ComputeDAG(matmul_nkkm(512, 512, 512))
    tgt = tvm.target.create("llvm")
    task = ansor.SearchTask(dag, "test", tgt)


if __name__ == "__main__":
    test_compute_dag_basic()
    test_state_split_fuse_reorder()
    test_state_compute_at_root_inline()
    test_state_cache_read_write()
    test_follow_split_follow_fused_split()
    test_rfactor()
    test_measure_local_builder_runner()
    # test_search_basic()
