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

"""Common functions for ansor test cases"""

from tvm import te, ansor
import topi


@ansor.register_auto_scheduler_workload_func
def matmul_ansor_test(N, M, K):
    A = te.placeholder((N, K), name='A')
    B = te.placeholder((K, M), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name='C')
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


def get_tiled_matmul():
    dag = ansor.ComputeDAG(matmul_ansor_test(512, 512, 512))

    s0 = dag.get_init_state()
    A, B, C = 0, 1, 2
    C_global = s0.cache_write(C, "global", dag)
    C += 1
    its0 = s0.split(C, s0.stages[C].iters[0], [4, 8, 8])
    its1 = s0.split(C, s0.stages[C].iters[4], [8, 4, 4])
    s0.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its0[3], its1[3]])
    s0.compute_at(C_global, C, s0.stages[C].iters[3])
    s0.split(C_global, s0.stages[C_global].iters[2], [16])
    B_global = s0.cache_read(B, "global", [C_global], dag)
    C += 1
    C_global += 1
    s0.compute_at(B_global, C_global, s0.stages[C_global].iters[0])
    A_global = s0.cache_read(A, "global", [C_global], dag)
    B += 1
    B_global += 1
    C += 1
    C_global += 1
    s0.compute_at(A_global, C_global, s0.stages[C_global].iters[2])
    return dag, s0.state_object

