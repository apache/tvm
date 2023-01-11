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
from .infra import make_ethosu_identity, get_pooling_args


@pytest.mark.parametrize("ifm_shape", [[1, 5, 9, 3], [20, 14, 7], [31, 40], [101]])
def test_identity(ifm_shape):
    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    identity = make_ethosu_identity(ifm)

    func = relay.Function(relay.analysis.free_vars(identity), identity)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_pooling_args(stmt))

    # Construct the ifm shape that the initial ifm shape gets legalized into
    ref_ifm_shape = ifm_shape

    if len(ref_ifm_shape) < 4:
        ref_ifm_shape = [1] + ref_ifm_shape

    while len(ref_ifm_shape) < 4:
        ref_ifm_shape.append(1)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)
    ifm_stride_c = 1
    ifm_stride_w = ref_ifm_shape[3]
    ifm_stride_h = ref_ifm_shape[2] * ref_ifm_shape[3]
    ofm_height = ref_ifm_shape[1]
    ofm_width = ref_ifm_shape[2]
    ofm_channels = ref_ifm_shape[3]
    ofm_stride_c = 1
    ofm_stride_w = ofm_channels if ofm_width > 1 else 1
    ofm_stride_h = ofm_channels * ofm_width if ofm_height > 1 else 1

    # The identity operator TIR gets converted into serial pooling
    serial_pooling = spec.SerialPooling(
        ifm=spec.SerialFeatureMap(
            data_type="int8",
            height=ref_ifm_shape[1],
            width=ref_ifm_shape[2],
            channels=ofm_channels,
            tile_height_0=ref_ifm_shape[1],
            tile_height_1=0,
            tile_width_0=ref_ifm_shape[2],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout="NHWC",
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
            layout="NHWC",
            stride_h=ofm_stride_h,
            stride_w=ofm_stride_w,
            stride_c=ofm_stride_c,
        ),
        pooling_type="AVG",
        pool_shape=spec.SerialKernel(1, 1, 1, 1, 1, 1),
        padding=spec.SerialPadding(0, 0, 0, 0),
        activation=spec.SerialActivation(op="NONE", clip_min=0, clip_max=0),
        upscale="NONE",
        rounding_mode="TFL",
        block_config=spec.SerialBlockConfig(0, 0, 0),
    )

    assert data[0] == ["ethosu_identity"] + list(serial_pooling)


if __name__ == "__main__":
    tvm.testing.main()
