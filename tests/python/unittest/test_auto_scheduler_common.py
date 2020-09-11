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

"""Common functions for auto_scheduler test cases"""

import threading

import tvm
from tvm import te, auto_scheduler
from tvm import topi
from tvm.topi.nn.winograd_util import winograd_transform_matrices
from tvm.topi.util import get_const_tuple


@auto_scheduler.register_workload
def matmul_auto_scheduler_test(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    return [A, B, C]


# Test for register_workload with different name
@auto_scheduler.register_workload("matmul_auto_scheduler_test_rename_1")
def matmul_auto_scheduler_test_rename_0(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")
    return [A, B, C]


@auto_scheduler.register_workload
def conv2d_nchw_bn_relu_auto_scheduler_test(
    N, H, W, CI, CO, kernel_size, strides, padding, dilation=1
):
    data = te.placeholder((N, CI, H, W), name="Data")
    kernel = te.placeholder((CO, CI, kernel_size, kernel_size), name="Kernel")
    bias = te.placeholder((CO, 1, 1), name="Bias")
    bn_scale = te.placeholder((CO, 1, 1), name="Bn_scale")
    bn_offset = te.placeholder((CO, 1, 1), name="Bn_offset")

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    conv = te.compute(
        (N, CO, OH, OW), lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0], name="Bias_add"
    )
    conv = te.compute(
        (N, CO, OH, OW), lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0], name="Bn_mul"
    )
    conv = te.compute(
        (N, CO, OH, OW), lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0], name="Bn_add"
    )
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]


@auto_scheduler.register_workload
def max_pool2d_auto_scheduler_test(N, H, W, CI, padding):
    data = te.placeholder((N, CI, H, W), name="Data")
    out = topi.nn.pool(data, [2, 2], [1, 1], [padding, padding, padding, padding], "max")

    return [data, out]


@auto_scheduler.register_workload
def min_nm_auto_scheduler_test(N, M):
    A = te.placeholder((N, M), name="A")
    B = topi.min(A, axis=-1)

    return [A, B]


@auto_scheduler.register_workload
def softmax_nm_auto_scheduler_test(N, M):
    A = te.placeholder((N, M), name="A")
    B = topi.nn.softmax(A, axis=1)

    return [A, B]


@auto_scheduler.register_workload
def softmax_abcd_auto_scheduler_test(a, b, c, d):
    A = te.placeholder((a, b, c, d), name="A")
    B = topi.nn.softmax(A, axis=-1)

    return [A, B]


@auto_scheduler.register_workload
def conv2d_winograd_nhwc_auto_scheduler_test(
    N, H, W, CI, CO, kernel_size=3, stride=1, padding=0, dilation=1
):
    tile_size = 4
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    N, H, W, CI = get_const_tuple(inputs.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"

    KH = KW = kernel_size
    HPAD, WPAD, _, _ = topi.nn.get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (stride, stride) if isinstance(stride, int) else stride
    assert HSTR == 1 and WSTR == 1 and KH == KW

    data_pad = topi.nn.pad(inputs, (0, HPAD, WPAD, 0), (0, HPAD, WPAD, 0), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, "float32")

    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    r_kh = te.reduce_axis((0, KH), name="r_kh")
    r_kw = te.reduce_axis((0, KW), name="r_kw")
    kshape = (alpha, alpha, CI, CO)
    kernel_pack = te.placeholder(kshape, inputs.dtype, name="weight")

    idxdiv = te.indexdiv
    idxmod = te.indexmod
    # pack input tile
    input_tile = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: data_pad[idxdiv(p, (nH * nW))][idxmod(idxdiv(p, nW), nH) * m + eps][
            idxmod(p, nW) * m + nu
        ][ci],
        name="input_tile",
    )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: te.sum(
            input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
        attrs={"auto_scheduler_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"]},
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, P, CO),
        lambda eps, nu, p, co: te.sum(
            data_pack[eps][nu][p][ci] * kernel_pack[eps][nu][ci][co], axis=[ci]
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    inverse = te.compute(
        (m, m, P, CO),
        lambda vh, vw, p, co: te.sum(
            bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
        attrs={"auto_scheduler_simplify_const_tensor_indices": ["vh", "vw", "r_a", "r_b"]},
    )

    # output
    output = te.compute(
        (N, H, W, CO),
        lambda n, h, w, co: inverse[
            idxmod(h, m), idxmod(w, m), n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), co
        ],
        name="conv2d_winograd",
    )

    return [inputs, kernel_pack, output]


def get_tiled_matmul():
    A, B, C = matmul_auto_scheduler_test(512, 512, 512)
    dag = auto_scheduler.ComputeDAG([A, B, C])

    s0 = dag.get_init_state()
    its0 = s0.split(C, s0[C].iters[0], [4, 8, 8])
    its1 = s0.split(C, s0[C].iters[4], [8, 4, 4])
    s0.reorder(
        C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its0[3], its1[3], s0[C].iters[8]]
    )

    return dag, s0


class PropagatingThread(threading.Thread):
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret
