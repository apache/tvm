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
from tvm.script import tir as T
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import total_cascader
from .infra import make_ethosu_conv2d, get_convolutional_args


def _create_serial_conv2d_params(
    ifm_shape,
    ifm_channels,
    ofm_channels,
    kernel_shape,
    padding,
    strides,
    dilation,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    rounding_mode="TFL",
    upscale="NONE",
):
    dtype = "int8"
    dilated_kernel_h = (kernel_shape[0] - 1) * dilation[0] + 1
    dilated_kernel_w = (kernel_shape[1] - 1) * dilation[1] + 1
    upscale_factor = 2 if upscale != "NONE" else 1

    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3]
        ifm_stride_h = ifm_shape[2] * ifm_shape[3]
        ofm_height = (
            ifm_shape[1] * upscale_factor - dilated_kernel_h + padding[0] + padding[2]
        ) // strides[0] + 1
        ofm_width = (
            ifm_shape[2] * upscale_factor - dilated_kernel_w + padding[1] + padding[3]
        ) // strides[1] + 1
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3]
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]
        ofm_height = (
            ifm_shape[1] * upscale_factor - dilated_kernel_h + padding[0] + padding[2]
        ) // strides[0] + 1
        ofm_width = (
            ifm_shape[3] * upscale_factor - dilated_kernel_w + padding[1] + padding[3]
        ) // strides[1] + 1

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = ofm_channels if ofm_width > 1 else 1
        ofm_stride_h = ofm_channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width
        ofm_stride_h = 16 * ofm_width * ((ofm_channels - 1) // 16 + 1)

    return [
        dtype,
        ifm_shape[1],
        ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
        ifm_channels,
        ifm_shape[1],
        0,
        ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
        0,
        0,
        0,
        0,
        0.5,
        10,
        ifm_layout,
        ifm_stride_h,
        ifm_stride_w,
        ifm_stride_c,
        dtype,
        ofm_height,
        ofm_width,
        ofm_channels,
        ofm_height,
        0,
        ofm_width,
        0,
        0,
        0,
        0,
        0.25,
        14,
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
        12,
        padding[0],
        padding[1],
        padding[2],
        padding[3],
        activation,
        10 if activation == "CLIP" else 0,
        100 if activation == "CLIP" else 0,
        rounding_mode,
        upscale,
    ]


@pytest.mark.parametrize(
    "trial",
    [
        [
            (1, 8, 8, 3),
            3,
            16,
            (1, 1),
            (2, 1, 2, 1),
            (1, 1),
            (1, 1),
            "CLIP",
            "NHWC",
            "NHWC",
            "TFL",
            "NONE",
        ],
        [
            (1, 8, 8, 3),
            3,
            16,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NONE",
            "NHWC",
            "NHWC",
            "NATURAL",
            "NONE",
        ],
        [
            (1, 1, 1, 1),
            1,
            16,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "CLIP",
            "NHWC",
            "NHWC",
            "TRUNCATE",
            "NONE",
        ],
        [
            (1, 7, 9, 4),
            4,
            13,
            (3, 2),
            (1, 2, 1, 2),
            (2, 1),
            (1, 2),
            "NONE",
            "NHWC",
            "NHWC",
            "TFL",
            "NONE",
        ],
        [
            (1, 8, 2, 8, 16),
            18,
            12,
            (1, 1),
            (2, 1, 2, 1),
            (1, 1),
            (1, 1),
            "CLIP",
            "NHCWB16",
            "NHWC",
            "NATURAL",
            "ZEROS",
        ],
        [
            (1, 7, 9, 4),
            4,
            71,
            (3, 2),
            (1, 2, 0, 2),
            (2, 1),
            (1, 2),
            "CLIP",
            "NHWC",
            "NHCWB16",
            "TRUNCATE",
            "ZEROS",
        ],
        [
            (1, 4, 12, 9, 16),
            182,
            67,
            (2, 3),
            (6, 3, 6, 2),
            (2, 2),
            (1, 1),
            "CLIP",
            "NHCWB16",
            "NHCWB16",
            "TFL",
            "ZEROS",
        ],
        [
            (1, 7, 9, 4),
            4,
            13,
            (3, 2),
            (1, 2, 0, 3),
            (2, 1),
            (2, 2),
            "CLIP",
            "NHWC",
            "NHWC",
            "NATURAL",
            "NEAREST",
        ],
        [
            (1, 7, 9, 4),
            4,
            71,
            (3, 2),
            (1, 2, 0, 2),
            (2, 1),
            (2, 2),
            "CLIP",
            "NHWC",
            "NHCWB16",
            "TRUNCATE",
            "NEAREST",
        ],
        [
            (1, 13, 12, 19, 16),
            182,
            67,
            (1, 3),
            (5, 3, 2, 3),
            (2, 1),
            (2, 1),
            "CLIP",
            "NHCWB16",
            "NHCWB16",
            "TFL",
            "NEAREST",
        ],
    ],
)
def test_conv2d_single(trial):
    def _get_func(
        ifm_shape,
        ifm_channels,
        ofm_channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        activation,
        ifm_layout,
        ofm_layout,
        rounding_mode,
        upscale,
    ):
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        conv = make_ethosu_conv2d(
            ifm,
            ifm_channels,
            ofm_channels,
            kernel_shape,
            padding,
            strides,
            dilation,
            activation=activation,
            ifm_layout=ifm_layout,
            ofm_layout=ofm_layout,
            rounding_mode=rounding_mode,
            upscale=upscale,
        )
        func = relay.Function(relay.analysis.free_vars(conv), conv)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    # TODO(@mbaret) Fix the tests for these known failures
    # These are anticipated to actually be correct, just a testing issue to do with
    # equivalent convolutions.
    known_failures = [
        [(1, 3, 12, 9, 16), 182, 67, (2, 3), (1, 3), (2, 2), (1, 1), "CLIP", "NHCWB16", "NHCWB16"],
        [(1, 2, 12, 9, 16), 182, 67, (1, 3), (6, 3), (2, 2), (1, 1), "CLIP", "NHCWB16", "NHCWB16"],
    ]
    func = _get_func(*trial)
    mod, _ = lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_convolutional_args(stmt, remove_constants=True))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)

    answer = _create_serial_conv2d_params(*trial)
    assert data[0] == answer, data[0]


# fmt: off
@tvm.script.ir_module
class Conv2dDoubleCascade1:
    @T.prim_func
    def main(placeholder_5: T.Buffer[(1, 8, 8, 3), "int8"], ethosu_write_1: T.Buffer[(1, 8, 8, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_2 = T.allocate([1024], "int8", "global", annotations={"disable_lower_builtin": True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 4, 3, 8, 0, 4, T.load("int8", placeholder_5.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 8, 4, 32, 8, 0, 4, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 160, 12, T.load("uint8", buffer_2, 0), 320, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 4, 32, 8, 0, 4, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 128, 32, 1, "int8", 8, 4, 8, 8, 0, 4, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 304, 12, T.load("uint8", buffer_1, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 4, 3, 8, 0, 4, T.load("int8", placeholder_5.data, 12), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 8, 4, 32, 8, 0, 4, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 160, 12, T.load("uint8", buffer_2, 0), 320, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 4, 32, 8, 0, 4, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 128, 32, 1, "int8", 8, 4, 8, 8, 0, 4, T.load("int8", ethosu_write_1.data, 32), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 304, 12, T.load("uint8", buffer_1, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dDoubleCascade2:
    @T.prim_func
    def main(placeholder_5: T.Buffer[(1, 8, 8, 3), "int8"], ethosu_write_1: T.Buffer[(1, 8, 8, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_2 = T.allocate([1536], "int8", "global", annotations={"disable_lower_builtin": True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 6, 8, 3, 6, 0, 8, T.load("int8", placeholder_5.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 5, 8, 32, 5, 0, 8, T.load("int8", ethosu_write_2, 256), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 1312, 12, T.load("uint8", buffer_1, 0), 320, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 8, 32, 5, 0, 8, T.load("int8", ethosu_write_2, 256), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "int8", 4, 8, 8, 4, 0, 8, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 2608, 12, T.load("uint8", buffer, 0), 80, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 6, 8, 3, 6, 0, 8, T.load("int8", placeholder_5.data, 48), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 5, 8, 32, 5, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 1312, 12, T.load("uint8", buffer_1, 0), 320, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 8, 32, 5, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "int8", 4, 8, 8, 4, 0, 8, T.load("int8", ethosu_write_1.data, 256), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 2608, 12, T.load("uint8", buffer, 0), 80, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dDoubleCascade3:
    @T.prim_func
    def main(placeholder_5: T.Buffer[(1, 16, 16, 3), "int8"], ethosu_write_1: T.Buffer[(1, 20, 4, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_2 = T.allocate([2560], "int8", "global", annotations={"disable_lower_builtin": True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 16, 3, 8, 0, 16, T.load("int8", placeholder_5.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 48, 3, 1, "int8", 8, 8, 32, 8, 0, 8, T.load("int8", ethosu_write_2, 512), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer_3, 0), 880, 12, T.load("uint8", buffer_2, 0), 320, 2, 1, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 32, 8, 0, 8, T.load("int8", ethosu_write_2, 512), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "int8", 8, 4, 8, 8, 0, 4, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 8, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer, 0), 1744, 12, T.load("uint8", buffer_1, 0), 80, 2, 1, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 12, 16, 3, 12, 0, 16, T.load("int8", placeholder_5.data, 192), 0, 0, 0, T.float32(0.5), 10, "NHWC", 48, 3, 1, "int8", 10, 8, 32, 10, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer_3, 0), 880, 12, T.load("uint8", buffer_2, 0), 320, 0, 1, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 10, 8, 32, 10, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "int8", 8, 4, 8, 8, 0, 4, T.load("int8", ethosu_write_1.data, 256), 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 8, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer, 0), 1744, 12, T.load("uint8", buffer_1, 0), 80, 0, 1, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 16, 3, 4, 0, 16, T.load("int8", placeholder_5.data, 576), 0, 0, 0, T.float32(0.5), 10, "NHWC", 48, 3, 1, "int8", 4, 8, 32, 4, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer_3, 0), 880, 12, T.load("uint8", buffer_2, 0), 320, 0, 1, 2, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 8, 32, 4, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "int8", 4, 4, 8, 4, 0, 4, T.load("int8", ethosu_write_1.data, 512), 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 8, 1, 2, 3, 2, 1, 2, 1, T.load("uint8", buffer, 0), 1744, 12, T.load("uint8", buffer_1, 0), 80, 0, 1, 2, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dDoubleCascade4:
    @T.prim_func
    def main(placeholder_5: T.Buffer[(1, 8, 1, 8, 16), "int8"], ethosu_write_1: T.Buffer[(1, 8, 2, 8, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_2 = T.allocate([2304], "int8", "global", annotations={"disable_lower_builtin": True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 6, 8, 3, 6, 0, 8, T.load("int8", placeholder_5.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 128, 16, 1, "int8", 5, 8, 35, 5, 0, 8, T.load("int8", ethosu_write_2, 384), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 384, 16, 128, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer, 0), 1456, 12, T.load("uint8", buffer_1, 0), 352, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 8, 35, 5, 0, 8, T.load("int8", ethosu_write_2, 384), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 384, 16, 128, "int8", 4, 8, 26, 4, 0, 8, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 256, 16, 128, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 11040, 12, T.load("uint8", buffer_2, 0), 272, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 6, 8, 3, 6, 0, 8, T.load("int8", placeholder_5.data, 256), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 128, 16, 1, "int8", 5, 8, 35, 5, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 384, 16, 128, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer, 0), 1456, 12, T.load("uint8", buffer_1, 0), 352, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 8, 35, 5, 0, 8, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 384, 16, 128, "int8", 4, 8, 26, 4, 0, 8, T.load("int8", ethosu_write_1.data, 1024), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 256, 16, 128, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_3, 0), 11040, 12, T.load("uint8", buffer_2, 0), 272, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dDoubleCascade5:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 8, 8, 3), "int8"], ethosu_write: T.Buffer[(1, 32, 32, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_1 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 8, 3, 4, 0, 8, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 8, 16, 32, 8, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 512, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 160, 12, T.load("uint8", buffer_1, 0), 320, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "ZEROS", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 16, 32, 8, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 32, 8, 16, 0, 32, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 304, 12, T.load("uint8", buffer_3, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "ZEROS", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 8, 3, 4, 0, 8, T.load("int8", placeholder.data, 96), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "int8", 8, 16, 32, 8, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 512, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 160, 12, T.load("uint8", buffer_1, 0), 320, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "ZEROS", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 16, 32, 8, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 32, 8, 16, 0, 32, T.load("int8", ethosu_write.data, 4096), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 304, 12, T.load("uint8", buffer_3, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "ZEROS", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dDoubleCascade6:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 8, 1, 8, 16), "int8"], ethosu_write: T.Buffer[(1, 32, 2, 32, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_1 = T.allocate([12288], "int8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 3, 8, 0, 8, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 128, 16, 1, "int8", 16, 16, 35, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 768, 16, 256, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer, 0), 1456, 12, T.load("uint8", buffer_1, 0), 352, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NEAREST", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 35, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHCWB16", 768, 16, 256, "int8", 32, 32, 26, 32, 0, 32, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHCWB16", 1024, 16, 512, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 11040, 12, T.load("uint8", buffer_3, 0), 272, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NEAREST", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "trial",
    [
        [
            Conv2dDoubleCascade1,
            (1, 8, 8, 3),
            3,
            32,
            8,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NHWC",
            "NONE",
            (1, 8, 4, 8),
        ],
        [
            Conv2dDoubleCascade2,
            (1, 8, 8, 3),
            3,
            32,
            8,
            (3, 3),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            "NHWC",
            "NONE",
            (1, 4, 8, 8),
        ],
        [
            Conv2dDoubleCascade3,
            (1, 16, 16, 3),
            3,
            32,
            8,
            (3, 2),
            (2, 1, 2, 1),
            (1, 2),
            (1, 2),
            "NHWC",
            "NONE",
            (1, 8, 4, 8),
        ],
        [
            Conv2dDoubleCascade4,
            (1, 8, 1, 8, 16),
            3,
            35,
            26,
            (3, 3),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            "NHCWB16",
            "NONE",
            (1, 4, 2, 8, 16),
        ],
        [
            Conv2dDoubleCascade5,
            (1, 8, 8, 3),
            3,
            32,
            8,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            "NHWC",
            "ZEROS",
            (1, 16, 32, 8),
        ],
        [
            Conv2dDoubleCascade6,
            (1, 8, 1, 8, 16),
            3,
            35,
            26,
            (3, 3),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            "NHCWB16",
            "NEAREST",
            (1, 32, 2, 32, 16),
        ],
    ],
)
def test_conv2d_double_cascade(trial):
    def _get_func(
        ifm_shape,
        ifm_channels,
        mid_channels,
        ofm_channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        layout,
        upscale,
    ):
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        conv1 = make_ethosu_conv2d(
            ifm,
            ifm_channels,
            mid_channels,
            kernel_shape,
            padding,
            strides,
            dilation,
            activation="NONE",
            ifm_layout=layout,
            ofm_layout=layout,
            upscale=upscale,
        )
        conv2 = make_ethosu_conv2d(
            conv1,
            mid_channels,
            ofm_channels,
            kernel_shape,
            padding,
            strides,
            dilation,
            activation="NONE",
            ifm_layout=layout,
            ofm_layout=layout,
            upscale=upscale,
        )
        func = relay.Function(relay.analysis.free_vars(conv2), conv2)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    reference_mod = trial[0]
    params = trial[1:]
    func = _get_func(*params[:-1])
    mod, _ = lower_to_tir(func, cascader=total_cascader(params[-1]))
    script = mod.script(show_meta=True)
    mod = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(mod["main"], reference_mod["main"], True)


# fmt: off
@tvm.script.ir_module
class Conv2dInlineCopy1:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(1, 10, 12, 8), "int8"], ethosu_write_1: T.Buffer[(1, 8, 8, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 4, 8, 0, 8, T.load("int8", placeholder_3.data, 120), 0, 0, 0, T.float32(0.5), 10, "NHWC", 96, 8, 1, "int8", 8, 8, 16, 8, 0, 8, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer, 0), 848, 12, T.load("uint8", buffer_1, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dInlineCopy2:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(1, 7, 9, 5), "int8"], ethosu_write_1: T.Buffer[(1, 3, 5, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 3, 5, 3, 3, 0, 5, T.load("int8", placeholder_3.data, 146), 0, 0, 0, T.float32(0.5), 10, "NHWC", 45, 5, 1, "int8", 3, 5, 16, 3, 0, 5, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 80, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 656, 12, T.load("uint8", buffer, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "trial",
    [
        [Conv2dInlineCopy1, (1, 10, 12, 8), (0, 1, 3, 0), (1, 9, 11, 4)],
        [Conv2dInlineCopy2, (1, 7, 9, 5), (0, 3, 2, 1), (1, 6, 7, 4)],
    ],
)
def test_conv2d_inline_copy(trial):
    def _get_func(ifm_shape, lower, upper, ofm_channels=16):
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        sliced = relay.strided_slice(ifm, lower, upper)
        conv = make_ethosu_conv2d(
            sliced, upper[3] - lower[3], ofm_channels, (3, 3), (1, 1), (1, 1), (1, 1)
        )
        func = relay.Function(relay.analysis.free_vars(conv), conv)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    reference_mod = trial[0]
    params = trial[1:]
    func = _get_func(*params)
    mod, _ = lower_to_tir(func)
    script = mod.script(show_meta=True)
    mod = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(mod["main"], reference_mod["main"], True)


# fmt: off
@tvm.script.ir_module
class Conv2dInlineReshape1:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(4, 6, 8, 1), "int8"], ethosu_write_1: T.Buffer[(1, 8, 6, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 72), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 384), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dInlineReshape2:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(1, 24, 8), "int8"], ethosu_write_1: T.Buffer[(1, 8, 6, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 72), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 384), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dInlineReshape3:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(192, 1), "int8"], ethosu_write_1: T.Buffer[(1, 8, 6, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 72), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 384), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class Conv2dInlineReshape4:
    @T.prim_func
    def main(placeholder_3: T.Buffer[(192,), "int8"], ethosu_write_1: T.Buffer[(1, 8, 6, 16), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 1, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 5, 6, 4, 5, 0, 6, T.load("int8", placeholder_3.data, 72), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 4, 1, "int8", 4, 6, 16, 4, 0, 6, T.load("int8", ethosu_write_1.data, 384), 0, 0, 0, T.float32(0.25), 14, "NHWC", 96, 16, 1, 3, 3, 1, 1, 1, 1, T.load("uint8", buffer_1, 0), 848, 12, T.load("uint8", buffer, 0), 160, 0, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "trial",
    [
        [Conv2dInlineReshape1, (4, 6, 8, 1), (1, 8, 6, 4), "NHWC"],
        [Conv2dInlineReshape2, (1, 4 * 6, 8), (1, 8, 6, 4), "NHWC"],
        [Conv2dInlineReshape3, (4 * 6 * 8, 1), (1, 8, 6, 4), "NHWC"],
        [Conv2dInlineReshape4, (4 * 6 * 8,), (1, 8, 6, 4), "NHWC"],
    ],
)
def test_conv2d_inline_reshape(trial):
    def _get_func(ifm_shape, reshaped, ifm_layout):
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        ifm_reshaped = relay.reshape(ifm, reshaped)
        conv = make_ethosu_conv2d(
            ifm_reshaped,
            reshaped[3],
            16,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            activation="NONE",
            ifm_layout=ifm_layout,
        )
        func = relay.Function(relay.analysis.free_vars(conv), conv)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    reference_mod = trial[0]
    params = trial[1:]
    func = _get_func(*params)
    mod, _ = lower_to_tir(func, cascader=total_cascader((1, 4, 6, 16)))
    script = mod.script(show_meta=True)
    mod = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(mod["main"], reference_mod["main"], True)


# TODO(@mbaret) Fix this case
@pytest.mark.xfail(raises=TypeError, strict=True)
def test_conv2d_big_pad():
    def _get_func():
        ifm_shape = (1, 2, 2, 8)
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        conv = make_ethosu_conv2d(
            ifm, ifm_shape[3], 16, (1, 1), (7, 7), (1, 1), (1, 1), ifm_layout="NHWC"
        )
        func = relay.Function(relay.analysis.free_vars(conv), conv)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func()
    mod, _ = lower_to_tir(func, cascader=total_cascader((1, 4, 4, 16)))


if __name__ == "__main__":
    pytest.main([__file__])
