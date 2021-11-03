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

from tvm import relay, TVMError
from tvm import relay
from tvm.relay.testing import run_opt_pass
from .infra import make_ethosu_conv2d
from .infra import make_ethosu_depthwise_conv2d
from .infra import make_ethosu_pooling


@pytest.mark.parametrize(
    ["ifm_shape", "ifm_layout"], [((1, 56, 72, 55), "NHWC"), ((1, 56, 4, 72, 16), "NHCWB16")]
)
@pytest.mark.parametrize(
    "ofm_shape,ofm_layout", [((1, 54, 38, 122), "NHWC"), ((1, 54, 8, 38, 16), "NHCWB16")]
)
def test_ethosu_conv2d_type_inference(
    ifm_shape,
    ifm_layout,
    ofm_shape,
    ofm_layout,
):
    ifm_channels = 55
    ofm_channels = 122
    kernel_shape = (3, 2)
    padding = (0, 1, 2, 3)
    strides = (1, 2)
    dilation = (2, 1)
    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    conv2d = make_ethosu_conv2d(
        ifm,
        ifm_channels,
        ofm_channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    func = relay.Function([ifm], conv2d)
    func = run_opt_pass(func, relay.transform.InferType())
    assert tuple(func.body.checked_type.shape) == ofm_shape


@pytest.mark.parametrize(
    "ifm_shape, ifm_layout", [((1, 46, 71, 55), "NHWC"), ((1, 46, 4, 71, 16), "NHCWB16")]
)
@pytest.mark.parametrize(
    "ofm_shape, ofm_layout", [((1, 44, 37, 55), "NHWC"), ((1, 44, 4, 37, 16), "NHCWB16")]
)
def test_ethosu_depthwise_conv2d_type_inference(
    ifm_shape,
    ifm_layout,
    ofm_shape,
    ofm_layout,
):
    channels = 55
    kernel_shape = (3, 2)
    padding = (0, 1, 2, 3)
    strides = (1, 2)
    dilation = (2, 1)
    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    depthwise_conv2d = make_ethosu_depthwise_conv2d(
        ifm,
        channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    func = relay.Function([ifm], depthwise_conv2d)
    func = run_opt_pass(func, relay.transform.InferType())
    assert tuple(func.body.checked_type.shape) == ofm_shape


@pytest.mark.parametrize(
    "ifm_shape, ifm_layout", [((1, 56, 72, 55), "NHWC"), ((1, 56, 4, 72, 16), "NHCWB16")]
)
@pytest.mark.parametrize(
    "ofm_shape, ofm_layout", [((1, 56, 38, 55), "NHWC"), ((1, 56, 4, 38, 16), "NHCWB16")]
)
def test_ethosu_pooling_type_inference(
    ifm_shape,
    ifm_layout,
    ofm_shape,
    ofm_layout,
):
    dtype = "int8"
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    pooling_type = "AVG"
    pool_shape = (3, 2)
    ofm_channels = 55
    strides = (1, 2)
    padding = (0, 1, 2, 3)
    pooling = make_ethosu_pooling(
        ifm,
        pooling_type,
        pool_shape,
        ofm_channels,
        strides,
        padding,
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    func = relay.Function([ifm], pooling)
    func = run_opt_pass(func, relay.transform.InferType())
    assert tuple(func.body.checked_type.shape) == ofm_shape
    assert func.body.checked_type.dtype == dtype


def test_ethosu_pooling_invalid_pooling_type():
    invalid_pooling_type = "A"
    dtype = "int8"
    ifm = relay.var("ifm", shape=[1, 56, 72, 55], dtype=dtype)
    pool_shape = (3, 2)
    ofm_channels = 55
    strides = (1, 2)
    padding = (0, 1, 2, 3)
    pooling = make_ethosu_pooling(
        ifm,
        invalid_pooling_type,
        pool_shape,
        ofm_channels,
        strides,
        padding,
    )
    func = relay.Function([ifm], pooling)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_pooling_invalid_dtype():
    invalid_dtype = "int32"
    ifm = relay.var("ifm", shape=[1, 56, 72, 55], dtype=invalid_dtype)
    pooling_type = "MAX"
    pool_shape = (3, 2)
    ofm_channels = 55
    strides = (1, 2)
    padding = (0, 1, 2, 3)
    pooling = make_ethosu_pooling(
        ifm,
        pooling_type,
        pool_shape,
        ofm_channels,
        strides,
        padding,
    )
    func = relay.Function([ifm], pooling)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


if __name__ == "__main__":
    pytest.main([__file__])
