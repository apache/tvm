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
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from .infra import make_ethosu_pooling, get_pooling_args


@pytest.mark.parametrize(
    "ifm_shape, ofm_channels, ifm_layout, ofm_layout",
    [
        ((1, 5, 9, 3), 3, "NHWC", "NHWC"),
        ((1, 8, 3, 9, 16), 40, "NHCWB16", "NHCWB16"),
        ((1, 8, 3, 9, 16), 40, "NHCWB16", "NHWC"),
        ((1, 8, 9, 40), 40, "NHWC", "NHCWB16"),
    ],
)
@pytest.mark.parametrize("pooling_type", ["AVG", "MAX"])
@pytest.mark.parametrize("activation", ["NONE", "CLIP"])
def test_pooling_single(
    ifm_shape,
    ofm_channels,
    ifm_layout,
    ofm_layout,
    pooling_type,
    activation,
):
    pool_shape = (3, 2)
    strides = (1, 2)
    padding = (1, 1, 1, 0)
    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    pooling = make_ethosu_pooling(
        ifm,
        pooling_type,
        pool_shape,
        ofm_channels,
        strides,
        padding,
        activation,
        ifm_layout,
        ofm_layout,
    )
    func = relay.Function(relay.analysis.free_vars(pooling), pooling)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_pooling_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)
    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3]
        ifm_stride_h = ifm_shape[2] * ifm_shape[3]
        ofm_height = (ifm_shape[1] - pool_shape[0] + padding[0] + padding[0]) // strides[0] + 1
        ofm_width = (ifm_shape[2] - pool_shape[1] + padding[1] + padding[1]) // strides[1] + 1
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3]
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]
        ofm_height = (ifm_shape[1] - pool_shape[0] + padding[0] + padding[0]) // strides[0] + 1
        ofm_width = (ifm_shape[3] - pool_shape[1] + padding[1] + padding[1]) // strides[1] + 1

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = ofm_channels if ofm_width > 1 else 1
        ofm_stride_h = ofm_channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width
        ofm_stride_h = 16 * ofm_width * ((ofm_channels - 1) // 16 + 1)

    serial_pooling = spec.SerialPooling(
        ifm=spec.SerialFeatureMap(
            data_type="int8",
            height=ifm_shape[1],
            width=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            channels=ofm_channels,
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
            data_type="int8",
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
        upscale="NONE",
    )

    assert data[0] == ["ethosu_pooling"] + list(serial_pooling)


if __name__ == "__main__":
    pytest.main([__file__])
