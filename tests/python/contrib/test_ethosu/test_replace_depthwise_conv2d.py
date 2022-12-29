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

pytest.importorskip("ethosu.vela")

import tvm
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from .infra import make_ethosu_depthwise_conv2d, get_convolutional_args


@pytest.mark.parametrize(
    "trial",
    [
        [(1, 8, 8, 3), 3, (3, 2), (0, 0), (1, 1), (1, 1), "CLIP", "NHWC", "NHWC", "TFL"],
        [(1, 8, 8, 3), 3, (1, 1), (2, 1), (1, 1), (1, 1), "NONE", "NHWC", "NHWC", "NATURAL"],
        [(1, 8, 8, 3), 3, (1, 1), (0, 0), (1, 1), (1, 1), "NONE", "NHWC", "NHWC", "TRUNCATE"],
        [(1, 1, 1, 1), 1, (1, 1), (0, 0), (1, 1), (1, 1), "CLIP", "NHWC", "NHWC", "TFL"],
        [(1, 7, 9, 4), 4, (3, 2), (1, 2), (2, 1), (1, 2), "NONE", "NHWC", "NHWC", "NATURAL"],
        [
            (1, 8, 2, 8, 16),
            18,
            (1, 1),
            (2, 1),
            (1, 1),
            (1, 1),
            "CLIP",
            "NHCWB16",
            "NHWC",
            "TRUNCATE",
        ],
        [(1, 7, 9, 40), 40, (3, 2), (1, 2), (2, 1), (1, 2), "CLIP", "NHWC", "NHCWB16", "TFL"],
        [
            (1, 4, 12, 9, 16),
            182,
            (2, 3),
            (6, 3),
            (2, 2),
            (1, 1),
            "CLIP",
            "NHCWB16",
            "NHCWB16",
            "NATURAL",
        ],
        [(1, 7, 9, 4), 4, (3, 2), (1, 2), (2, 1), (2, 2), "CLIP", "NHWC", "NHWC", "TRUNCATE"],
        [(1, 7, 9, 41), 41, (3, 2), (1, 2), (2, 1), (2, 2), "CLIP", "NHWC", "NHCWB16", "TFL"],
        [
            (1, 13, 12, 19, 16),
            182,
            (1, 3),
            (5, 3),
            (2, 1),
            (2, 1),
            "CLIP",
            "NHCWB16",
            "NHCWB16",
            "NATURAL",
        ],
    ],
)
@tvm.testing.skip_parameterizations(
    "trial3", reason="See https://github.com/apache/tvm/issues/12841"
)
def test_depthwise_conv2d_single(request, trial):
    def _get_func(
        ifm_shape,
        channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        activation,
        ifm_layout,
        ofm_layout,
        rounding_mode,
    ):
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        depthwise = make_ethosu_depthwise_conv2d(
            ifm,
            channels,
            kernel_shape,
            padding,
            strides,
            dilation,
            activation,
            ifm_layout,
            ofm_layout,
            "int8",
            "uint8",
            rounding_mode,
        )
        func = relay.Function(relay.analysis.free_vars(depthwise), depthwise)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func(*trial)
    mod, _ = _lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_convolutional_args(stmt, remove_constants=True))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)
    (
        ifm_shape,
        channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        activation,
        ifm_layout,
        ofm_layout,
        rounding_mode,
    ) = trial
    dilated_kernel_h = (kernel_shape[0] - 1) * dilation[0] + 1
    dilated_kernel_w = (kernel_shape[1] - 1) * dilation[1] + 1
    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3]
        ifm_stride_h = ifm_shape[2] * ifm_shape[3]
        ofm_height = (ifm_shape[1] - dilated_kernel_h + padding[0] + padding[0]) // strides[0] + 1
        ofm_width = (ifm_shape[2] - dilated_kernel_w + padding[1] + padding[1]) // strides[1] + 1
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3]
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]
        ofm_height = (ifm_shape[1] - dilated_kernel_h + padding[0] + padding[0]) // strides[0] + 1
        ofm_width = (ifm_shape[3] - dilated_kernel_w + padding[1] + padding[1]) // strides[1] + 1

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = channels if ofm_width > 1 else 1
        ofm_stride_h = channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width
        ofm_stride_h = 16 * ofm_width * ((channels - 1) // 16 + 1)

    answer = [
        "int8",
        ifm_shape[1],
        ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
        channels,
        ifm_shape[1],
        0,
        ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
        0,
        0,
        0,
        0,
        0.6,
        11,
        ifm_layout,
        ifm_stride_h,
        ifm_stride_w,
        ifm_stride_c,
        "int8",
        ofm_height,
        ofm_width,
        channels,
        ofm_height,
        0,
        ofm_width,
        0,
        0,
        0,
        0,
        0.26,
        15,
        ofm_layout,
        ofm_stride_h,
        ofm_stride_w,
        ofm_stride_c,
        kernel_shape[1],
        kernel_shape[0],
        strides[1],
        strides[0],
        dilation[1],
        dilation[0],
        13,
        padding[0],
        padding[1],
        padding[0],
        padding[1],
        activation,
        15 if activation == "CLIP" else 0,
        105 if activation == "CLIP" else 0,
        rounding_mode,
        "NONE",
        0,
        0,
        0,
    ]
    assert data[0] == answer, data[0]
