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
from tvm.relay.backend.contrib.ethosu.tir import spec
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from .infra import make_ethosu_pooling, get_pooling_args


def _create_serial_pooling(
    ifm_shape,
    ofm_channels,
    ifm_layout,
    ofm_layout,
    pool_shape,
    pooling_type,
    strides,
    padding,
    activation="NONE",
    rounding_mode="TFL",
    upscale="NONE",
    ofm_dtype="int8",
):
    upscale_factor = 2 if upscale != "NONE" else 1
    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3]
        ifm_stride_h = ifm_shape[2] * ifm_shape[3]
        ofm_height = (
            ifm_shape[1] * upscale_factor - pool_shape[0] + padding[0] + padding[2]
        ) // strides[0] + 1
        ofm_width = (
            ifm_shape[2] * upscale_factor - pool_shape[1] + padding[1] + padding[3]
        ) // strides[1] + 1
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3] if ofm_channels >= 16 else 1
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]
        ofm_height = (
            ifm_shape[1] * upscale_factor - pool_shape[0] + padding[0] + padding[2]
        ) // strides[0] + 1
        ofm_width = (
            ifm_shape[3] * upscale_factor - pool_shape[1] + padding[1] + padding[3]
        ) // strides[1] + 1

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = ofm_channels if ofm_width > 1 else 1
        ofm_stride_h = ofm_channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width if ofm_channels >= 16 else 1
        ofm_stride_h = 16 * ofm_width * ((ofm_channels - 1) // 16 + 1)

    ifm_channels = ofm_channels if pooling_type != "SUM" else ifm_shape[-1]

    return spec.SerialPooling(
        ifm=spec.SerialFeatureMap(
            data_type="int8",
            height=ifm_shape[1],
            width=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            channels=ifm_channels,
            tile_height_0=ifm_shape[1],
            tile_height_1=0,
            tile_width_0=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ifm_layout,
            stride_h=ifm_stride_h,
            stride_w=ifm_stride_w,
            stride_c=ifm_stride_c,
        ),
        ofm=spec.SerialFeatureMap(
            data_type=ofm_dtype,
            height=ofm_height,
            width=ofm_width,
            channels=ofm_channels,
            tile_height_0=ofm_height,
            tile_height_1=0,
            tile_width_0=ofm_width,
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ofm_layout,
            stride_h=ofm_stride_h,
            stride_w=ofm_stride_w,
            stride_c=ofm_stride_c,
        ),
        pooling_type=pooling_type,
        pool_shape=spec.SerialKernel(
            width=pool_shape[1],
            height=pool_shape[0],
            stride_w=strides[1],
            stride_h=strides[0],
            dilation_w=1,
            dilation_h=1,
        ),
        padding=spec.SerialPadding(
            top=padding[0], left=padding[1], bottom=padding[2], right=padding[3]
        ),
        activation=spec.SerialActivation(
            op=activation,
            clip_min=10 if activation == "CLIP" else 0,
            clip_max=100 if activation == "CLIP" else 0,
        ),
        rounding_mode=rounding_mode,
        upscale=upscale,
        block_config=spec.SerialBlockConfig(0, 0, 0),
    )


@pytest.mark.parametrize(
    "ifm_shape, ofm_channels, ifm_layout, ofm_layout, rounding_mode, upscale",
    [
        ((1, 5, 9, 3), 3, "NHWC", "NHWC", "TFL", "NONE"),
        ((1, 8, 3, 9, 16), 40, "NHCWB16", "NHCWB16", "NATURAL", "NONE"),
        ((1, 8, 3, 9, 16), 40, "NHCWB16", "NHWC", "TRUNCATE", "ZEROS"),
        ((1, 8, 9, 40), 40, "NHWC", "NHCWB16", "TFL", "ZEROS"),
        ((1, 8, 9, 8), 8, "NHWC", "NHCWB16", "TFL", "NEAREST"),
        ((1, 5, 9, 3), 3, "NHWC", "NHWC", "TFL", "NEAREST"),
    ],
)
@pytest.mark.parametrize("pooling_type", ["AVG", "MAX"])
@pytest.mark.parametrize("activation", ["NONE", "CLIP"])
def test_avg_max_pooling_single(
    ifm_shape,
    ofm_channels,
    ifm_layout,
    ofm_layout,
    pooling_type,
    activation,
    rounding_mode,
    upscale,
):
    pool_shape = (3, 2)
    strides = (1, 2)

    # When strides are not (1, 1) it is possible to create invalid
    # padding configurations. It is possible to construct a pooling
    # operation with invalid padding, but the compiler will account
    # for this and adjust the padding accordingly, leading to a
    # mismatch between the expected and actual result. Therefore,
    # hardcoded padding values are used for each case.
    padding = (1, 1, 1, 0) if upscale == "NONE" else (0, 0, 0, 0)

    dtype = "int8"

    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    pooling = make_ethosu_pooling(
        ifm,
        pooling_type,
        pool_shape,
        ofm_channels,
        dtype,
        strides,
        padding,
        activation,
        ifm_layout,
        ofm_layout,
        rounding_mode,
        upscale,
    )
    func = relay.Function(relay.analysis.free_vars(pooling), pooling)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_pooling_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)

    serial_pooling = _create_serial_pooling(
        ifm_shape,
        ofm_channels,
        ifm_layout,
        ofm_layout,
        pool_shape,
        pooling_type,
        strides,
        padding,
        activation,
        rounding_mode,
        upscale,
    )
    assert data[0] == ["ethosu_pooling"] + list(serial_pooling)


@pytest.mark.parametrize(
    "ifm_shape, ofm_layout, rounding_mode",
    [
        ((1, 5, 9, 3), "NHWC", "TFL"),
        ((1, 8, 9, 40), "NHCWB16", "TFL"),
        ((1, 8, 9, 8), "NHCWB16", "TRUNCATE"),
        ((1, 5, 9, 3), "NHWC", "NATURAL"),
    ],
)
@pytest.mark.parametrize("activation", ["NONE", "CLIP"])
def test_sum_pooling_single(
    ifm_shape,
    ofm_layout,
    activation,
    rounding_mode,
):
    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    pooling = make_ethosu_pooling(
        ifm=ifm,
        pooling_type="SUM",
        pool_shape=(1, 1),
        ofm_channels=1,
        ofm_dtype="int32",
        strides=(1, 1),
        padding=(0, 0, 0, 0),
        activation=activation,
        ofm_layout=ofm_layout,
        rounding_mode=rounding_mode,
    )
    func = relay.Function(relay.analysis.free_vars(pooling), pooling)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_pooling_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)

    serial_pooling = _create_serial_pooling(
        ifm_shape=ifm_shape,
        ofm_channels=1,
        ifm_layout="NHWC",
        ofm_layout=ofm_layout,
        pool_shape=(1, 1),
        pooling_type="SUM",
        strides=(1, 1),
        padding=(0, 0, 0, 0),
        activation=activation,
        rounding_mode=rounding_mode,
        ofm_dtype="int32",
    )
    assert data[0] == ["ethosu_pooling"] + list(serial_pooling)


def test_correct_stride_with_multiple_pooling():
    """Testing a specific case of two pooling operations with NHWC inputs/outputs
    but a NHCWB16 intermediate tensor. This lead to elements being accessed in the
    wrong order by the NPU, due to incorrect stride values being calculated."""

    ifm_shape = (1, 4, 4, 8)
    ofm_channels = 8
    pooling_type = "MAX"
    pool_shape = (1, 1)
    strides = (1, 1)
    padding = (0, 0, 0, 0)
    dtype = "int8"

    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    op = make_ethosu_pooling(
        ifm,
        pooling_type,
        pool_shape,
        ofm_channels,
        dtype,
        strides,
        padding,
        ifm_layout="NHWC",
        ofm_layout="NHCWB16",
    )
    op = make_ethosu_pooling(
        op,
        pooling_type,
        pool_shape,
        ofm_channels,
        dtype,
        strides,
        padding,
        ifm_layout="NHCWB16",
        ofm_layout="NHWC",
    )
    func = relay.Function(relay.analysis.free_vars(op), op)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)

    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_pooling_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)

    serial_pooling_1 = _create_serial_pooling(
        [1, 4, 4, 8],
        8,
        "NHWC",
        "NHCWB16",
        pool_shape,
        pooling_type,
        strides,
        padding,
    )
    serial_pooling_2 = _create_serial_pooling(
        [1, 4, 1, 4, 16],
        8,
        "NHCWB16",
        "NHWC",
        pool_shape,
        pooling_type,
        strides,
        padding,
    )

    assert data[0] == ["ethosu_pooling"] + list(serial_pooling_1)
    assert data[1] == ["ethosu_pooling"] + list(serial_pooling_2)


if __name__ == "__main__":
    tvm.testing.main()
