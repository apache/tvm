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
from tvm.relay.testing import run_opt_pass
from .infra import make_ethosu_conv2d
from .infra import make_ethosu_depthwise_conv2d
from .infra import make_ethosu_pooling
from .infra import make_ethosu_binary_elementwise
from .infra import make_ethosu_identity
from .infra import make_ethosu_unary_elementwise


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
    "ifm_dtype,weight_dtype,scale_bias_dtype",
    [("float32", "int8", "uint8"), ("int8", "float32", "uint8"), ("int8", "int8", "float32")],
)
def test_ethosu_conv2d_invalid_dtypes(ifm_dtype, weight_dtype, scale_bias_dtype):
    ifm_channels = 55
    ofm_channels = 122
    kernel_shape = (3, 2)
    padding = (0, 1, 2, 3)
    strides = (1, 2)
    dilation = (2, 1)
    ifm = relay.var("ifm", shape=(1, 56, 72, 55), dtype=ifm_dtype)
    conv2d = make_ethosu_conv2d(
        ifm,
        ifm_channels,
        ofm_channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        weight_dtype=weight_dtype,
        scale_bias_dtype=scale_bias_dtype,
    )
    func = relay.Function([ifm], conv2d)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_conv2d_invalid_upscale_method():
    invalid_upscale_method = "FOO"
    ifm_channels = 55
    ofm_channels = 122
    kernel_shape = (3, 2)
    padding = (0, 1, 2, 3)
    strides = (1, 2)
    dilation = (2, 1)
    ifm = relay.var("ifm", shape=(1, 56, 72, 55), dtype="int8")
    conv2d = make_ethosu_conv2d(
        ifm,
        ifm_channels,
        ofm_channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        weight_dtype="int8",
        scale_bias_dtype="uint8",
        upscale=invalid_upscale_method,
    )
    func = relay.Function([ifm], conv2d)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


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
    "ifm_dtype,weight_dtype,scale_bias_dtype",
    [("float32", "int8", "uint8"), ("int8", "float32", "uint8"), ("int8", "int8", "float32")],
)
def test_ethosu_depthwise_conv2d_invalid_dtypes(ifm_dtype, weight_dtype, scale_bias_dtype):
    channels = 55
    kernel_shape = (3, 2)
    padding = (0, 1, 2, 3)
    strides = (1, 2)
    dilation = (2, 1)
    dilation = (2, 1)
    ifm = relay.var("ifm", shape=(1, 56, 72, 55), dtype=ifm_dtype)
    depthwise_conv2d = make_ethosu_depthwise_conv2d(
        ifm,
        channels,
        kernel_shape,
        padding,
        strides,
        dilation,
        weight_dtype=weight_dtype,
        scale_bias_dtype=scale_bias_dtype,
    )
    func = relay.Function([ifm], depthwise_conv2d)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


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
        dtype,
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
        dtype,
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
        "int8",
        strides,
        padding,
    )
    func = relay.Function([ifm], pooling)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_pooling_invalid_upscale_method():
    invalid_upscale_method = "FOO"
    dtype = "int8"

    ifm = relay.var("ifm", shape=[1, 56, 72, 55], dtype=dtype)
    pooling = make_ethosu_pooling(
        ifm,
        "MAX",
        (3, 2),
        55,
        dtype,
        (1, 2),
        (0, 1, 2, 3),
        upscale=invalid_upscale_method,
    )
    func = relay.Function([ifm], pooling)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


@pytest.mark.parametrize(
    "ifm_shape, ifm_layout", [((1, 4, 5, 33), "NHWC"), ((1, 4, 3, 5, 16), "NHCWB16")]
)
@pytest.mark.parametrize(
    "ofm_shape, ofm_layout", [((1, 4, 5, 33), "NHWC"), ((1, 4, 3, 5, 16), "NHCWB16")]
)
def test_ethosu_binary_elementwise_type_inference(
    ifm_shape,
    ifm_layout,
    ofm_shape,
    ofm_layout,
):
    dtype = "int8"
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm2 = relay.var("ifm2", shape=ifm_shape, dtype=dtype)
    operator_type = "ADD"
    ifm_channels, ifm2_channels = 33, 33
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        dtype,
        ifm_layout=ifm_layout,
        ifm2_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    func = relay.Function([ifm, ifm2], binary_elementwise)
    func = run_opt_pass(func, relay.transform.InferType())
    assert tuple(func.body.checked_type.shape) == ofm_shape
    assert func.body.checked_type.dtype == dtype


def test_ethosu_binary_elementwise_invalid_operator_type():
    invalid_operator_type = "A"
    ifm_shape = [1, 4, 5, 33]
    dtype = "int8"
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm2 = relay.var("ifm2", shape=ifm_shape, dtype=dtype)
    ifm_channels, ifm2_channels = 33, 33
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        invalid_operator_type,
        dtype,
    )
    func = relay.Function([ifm, ifm2], binary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_binary_elementwise_invalid_data_types():
    dtype = "int8"
    dtype2 = "int32"
    operator_type = "ADD"
    ifm_shape = [1, 4, 5, 33]
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm2 = relay.var("ifm2", shape=ifm_shape, dtype=dtype2)
    ifm_channels, ifm2_channels = 33, 33
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        dtype,
    )
    func = relay.Function([ifm, ifm2], binary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


@pytest.mark.parametrize("operator_type", ["MIN", "MAX"])
def test_ethosu_binary_elementwise_min_max_invalid_data_type(operator_type):
    invalid_dtype = "int32"
    ifm_shape = [1, 4, 5, 33]
    ifm = relay.var("ifm", shape=ifm_shape, dtype=invalid_dtype)
    ifm2 = relay.var("ifm2", shape=ifm_shape, dtype=invalid_dtype)
    ifm_channels, ifm2_channels = 33, 33
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        invalid_dtype,
    )
    func = relay.Function([ifm, ifm2], binary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


@pytest.mark.parametrize("invalid_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("operator_type", ["RHS", "SHR"])
def test_ethosu_binary_elementwise_shift_invalid_data_type(invalid_dtype, operator_type):
    ifm_shape = [1, 4, 5, 33]
    ifm = relay.var("ifm", shape=ifm_shape, dtype=invalid_dtype)
    ifm2 = relay.var("ifm2", shape=ifm_shape, dtype=invalid_dtype)
    ifm_channels, ifm2_channels = 33, 33
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        invalid_dtype,
    )
    func = relay.Function([ifm, ifm2], binary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


@pytest.mark.parametrize("shape", [(1, 56, 72, 55), (241, 7, 755), (28, 44), (5003,)])
def test_ethosu_identity_type_inference(shape):
    dtype = "int8"
    ifm = relay.var("ifm", shape=shape, dtype=dtype)
    identity = make_ethosu_identity(ifm)
    func = relay.Function([ifm], identity)
    func = run_opt_pass(func, relay.transform.InferType())
    assert tuple(func.body.checked_type.shape) == shape
    assert func.body.checked_type.dtype == dtype


def test_ethosu_identity_invalid_shape():
    invalid_shape = [1, 2, 3, 4, 5]
    dtype = "int8"
    ifm = relay.var("ifm", shape=invalid_shape, dtype=dtype)

    identity = make_ethosu_identity(ifm)
    func = relay.Function([ifm], identity)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_identity_invalid_dtype():
    invalid_dtype = "int32"
    ifm = relay.var("ifm", shape=[6000], dtype=invalid_dtype)

    identity = make_ethosu_identity(ifm)
    func = relay.Function([ifm], identity)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


@pytest.mark.parametrize(
    "ifm_shape, ifm_layout", [((1, 4, 5, 33), "NHWC"), ((1, 4, 3, 5, 16), "NHCWB16")]
)
@pytest.mark.parametrize(
    "ofm_shape, ofm_layout", [((1, 4, 5, 33), "NHWC"), ((1, 4, 3, 5, 16), "NHCWB16")]
)
@pytest.mark.parametrize("operator_type, data_type", [("ABS", "int8"), ("CLZ", "int32")])
def test_ethosu_unary_elementwise_type_inference(
    ifm_shape,
    ifm_layout,
    ofm_shape,
    ofm_layout,
    operator_type,
    data_type,
):
    ifm = relay.var("ifm", shape=ifm_shape, dtype=data_type)
    ofm_channels = 33
    unary_elementwise = make_ethosu_unary_elementwise(
        ifm,
        ofm_channels,
        operator_type,
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    f = relay.Function([ifm], unary_elementwise)
    f = run_opt_pass(f, relay.transform.InferType())
    assert tuple(f.body.checked_type.shape) == ofm_shape


def test_ethosu_unary_elementwise_invalid_operator_type():
    ifm = relay.var("ifm", shape=(1, 3, 7, 12), dtype="int8")
    invalid_op_type = "ABBBS"
    unary_elementwise = make_ethosu_unary_elementwise(
        ifm,
        12,
        invalid_op_type,
    )
    func = relay.Function([ifm], unary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


def test_ethosu_unary_elementwise_invalid_dtype():
    invalid_dtype = "int32"
    ifm = relay.var("ifm", shape=(1, 5, 15, 25), dtype=invalid_dtype)

    unary_elementwise = make_ethosu_unary_elementwise(
        ifm,
        25,
        "ABS",
    )
    func = relay.Function([ifm], unary_elementwise)
    with pytest.raises(TVMError):
        run_opt_pass(func, relay.transform.InferType())


if __name__ == "__main__":
    tvm.testing.main()
