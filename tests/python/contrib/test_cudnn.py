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

import sys

import pytest

import tvm
from tvm import te
from tvm.contrib import cudnn
from tvm.contrib.nvcc import have_fp16
import numpy as np
import tvm.topi.testing
import tvm.testing


requires_cudnn = pytest.mark.skipif(
    tvm.get_global_func("tvm.contrib.cudnn.conv.output_shape_from_cudnn", True) is None,
    reason="CuDNN is not enabled",
)


def verify_conv2d(data_dtype, conv_dtype, tensor_format=0, groups=1):
    in_channel = 4
    out_channel = 16
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    height = 32
    width = 32

    if data_dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
        print("Skip because gpu does not have fp16 support")
        return

    # schedule
    if tensor_format == 0:
        xshape = [batch, in_channel, height, width]
        wshape = [out_channel, in_channel // groups, filter_h, filter_w]
    else:
        xshape = [batch, height, width, in_channel]
        wshape = [out_channel, filter_h, filter_w, in_channel // groups]

    X = te.placeholder(xshape, name="X", dtype=data_dtype)
    W = te.placeholder(wshape, name="W", dtype=data_dtype)
    Y = cudnn.conv_forward(
        X,
        W,
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        conv_dtype=conv_dtype,
        algo=-1,
        groups=groups,
    )
    yshape = [x.value for x in Y.shape]
    s = te.create_schedule(Y.op)

    # validation
    dev = tvm.cuda(0)
    f = tvm.build(s, [X, W, Y], "cuda --host=llvm", name="conv2d")
    x_np = np.random.uniform(-1, 1, xshape).astype(data_dtype)
    w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)
    y_np = np.zeros(yshape).astype(data_dtype)
    x = tvm.nd.array(x_np, dev)
    w = tvm.nd.array(w_np, dev)
    y = tvm.nd.array(y_np, dev)
    if tensor_format == 0:
        c_np = tvm.topi.testing.conv2d_nchw_python(x_np, w_np, 1, 1, groups=groups)
    elif tensor_format == 1:
        wt = w_np.transpose((1, 2, 3, 0))  # OHWI => HWIO
        c_np = tvm.topi.testing.conv2d_nhwc_python(x_np, wt, 1, 1, groups=groups)

    f(x, w, y)
    tvm.testing.assert_allclose(y.numpy(), c_np, atol=1e-2, rtol=1e-2)


@tvm.testing.requires_gpu
@requires_cudnn
def test_conv2d():
    verify_conv2d("float32", "float32", tensor_format=0)
    verify_conv2d("float16", "float32", tensor_format=1)
    # This test is flaky, disable for now
    # verify_conv2d("float16", "float16", tensor_format=0)
    verify_conv2d("int8", "int32", tensor_format=1)

    verify_conv2d("float32", "float32", tensor_format=0, groups=2)
    verify_conv2d("float16", "float32", tensor_format=1, groups=2)
    verify_conv2d("float16", "float16", tensor_format=0, groups=2)
    verify_conv2d("int8", "int32", tensor_format=1, groups=2)


def verify_conv3d(data_dtype, conv_dtype, tensor_format=0, groups=1):
    in_channel = 4
    out_channel = 16
    filter_d = 3
    filter_h = 3
    filter_w = 3
    pad_d = 1
    pad_h = 1
    pad_w = 1
    stride_d = 1
    stride_h = 1
    stride_w = 1
    dilation_d = 1
    dilation_h = 1
    dilation_w = 1
    batch = 3
    depth = 32
    height = 32
    width = 32

    # schedule
    xshape = [batch, in_channel, depth, height, width]
    wshape = [out_channel, in_channel // groups, filter_d, filter_h, filter_w]

    X = te.placeholder(xshape, name="X", dtype=data_dtype)
    W = te.placeholder(wshape, name="W", dtype=data_dtype)
    Y = cudnn.conv_forward(
        X,
        W,
        [pad_d, pad_h, pad_w],
        [stride_d, stride_h, stride_w],
        [dilation_d, dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        algo=-1,
        conv_dtype=conv_dtype,
        groups=groups,
    )
    yshape = [x.value for x in Y.shape]
    s = te.create_schedule(Y.op)

    # validation
    dev = tvm.cuda(0)
    f = tvm.build(s, [X, W, Y], target="cuda --host=llvm", name="conv3d")
    x_np = np.random.uniform(-1, 1, xshape).astype(data_dtype)
    w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)
    y_np = np.zeros(yshape).astype(data_dtype)
    x = tvm.nd.array(x_np, dev)
    w = tvm.nd.array(w_np, dev)
    y = tvm.nd.array(y_np, dev)
    if tensor_format == 0:
        c_np = tvm.topi.testing.conv3d_ncdhw_python(x_np, w_np, 1, 1, groups)
    else:
        raise AssertionError("For now, conv3d tensor format only support: 0(NCHW)")

    f(x, w, y)
    tvm.testing.assert_allclose(y.numpy(), c_np, atol=3e-5, rtol=1e-4)


@tvm.testing.requires_gpu
@requires_cudnn
def test_conv3d():
    verify_conv3d("float32", "float32", tensor_format=0)
    verify_conv3d("float32", "float32", tensor_format=0, groups=2)


def verify_softmax(shape, axis, dtype="float32", log_softmax=False):
    cudnn_op = cudnn.log_softmax if log_softmax else cudnn.softmax
    testing_op = (
        tvm.topi.testing.log_softmax_python if log_softmax else tvm.topi.testing.softmax_python
    )

    A = te.placeholder(shape, dtype=dtype, name="A")
    B = cudnn_op(A, axis)
    s = te.create_schedule([B.op])

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = testing_op(a_np)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    f = tvm.build(s, [A, B], target="cuda --host=llvm", name="softmax")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3)


def verify_softmax_4d(shape, dtype="float32", log_softmax=False):
    cudnn_op = cudnn.log_softmax if log_softmax else cudnn.softmax
    testing_op = (
        tvm.topi.testing.log_softmax_python if log_softmax else tvm.topi.testing.softmax_python
    )

    A = te.placeholder(shape, dtype=dtype, name="A")
    B = cudnn_op(A, axis=1)
    s = te.create_schedule([B.op])

    dev = tvm.cuda(0)
    n, c, h, w = shape
    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = testing_op(a_np.transpose(0, 2, 3, 1).reshape(h * w, c))
    b_np = b_np.reshape(n, h, w, c).transpose(0, 3, 1, 2)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    f = tvm.build(s, [A, B], target="cuda --host=llvm", name="softmax")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@requires_cudnn
def test_softmax():
    verify_softmax((32, 10), -1)
    verify_softmax((3, 4), -1)
    verify_softmax((1, 5), -1, "float64")
    verify_softmax_4d((1, 16, 256, 256))
    verify_softmax_4d((1, 16, 256, 256), "float64")

    verify_softmax((32, 10), -1, log_softmax=True)
    verify_softmax((3, 4), -1, log_softmax=True)
    verify_softmax((1, 5), -1, "float64", log_softmax=True)
    verify_softmax_4d((1, 16, 256, 256), log_softmax=True)
    verify_softmax_4d((1, 16, 256, 256), "float64", log_softmax=True)


test_kwargs_default_2d = {
    "tensor_format": 0,
    "pad": [1, 1],
    "stride": [1, 1],
    "dilation": [1, 1],
    "x_shape": [16, 4, 32, 32],
    "w_shape": [8, 4, 3, 3],
    "groups": 1,
    "conv_dtype": "float32",
    "data_dtype": "float32",
}
test_kwargs_default_3d = {
    "tensor_format": 0,
    "pad": [1, 1, 1],
    "stride": [1, 1, 1],
    "dilation": [1, 1, 1],
    "x_shape": [16, 4, 32, 32, 32],
    "w_shape": [8, 4, 3, 3, 3],
    "groups": 1,
    "conv_dtype": "float32",
    "data_dtype": "float32",
}
conv_output_shape_conditions = {
    "2d_small": test_kwargs_default_2d,
    "2d_large": {
        **test_kwargs_default_2d,
        "x_shape": [16, 32, 512, 1024],
        "w_shape": [8, 32, 5, 5],
    },
    "2d_pad": {**test_kwargs_default_2d, "pad": [2, 3]},
    "2d_stride": {**test_kwargs_default_2d, "stride": [2, 3]},
    "2d_dilation": {**test_kwargs_default_2d, "dilation": [2, 3]},
    "2d_groups": {**test_kwargs_default_2d, "groups": 4, "w_shape": [8, 1, 3, 3]},
    "2d_NHWC": {
        **test_kwargs_default_2d,
        "tensor_format": 1,
        "x_shape": [16, 32, 32, 4],
        "w_shape": [8, 3, 3, 4],
    },
    "2d_NCHW_VECT_C": {
        **test_kwargs_default_2d,
        "tensor_format": 2,
        "w_shape": [8, 16, 3, 3],
        "data_dtype": "int8x4",
    },
    "3d_small": test_kwargs_default_3d,
    "3d_large": {
        **test_kwargs_default_3d,
        "x_shape": [16, 32, 64, 128, 256],
        "w_shape": [8, 32, 5, 5, 5],
    },
    "3d_pad": {**test_kwargs_default_3d, "pad": [2, 3, 4]},
    "3d_stride": {**test_kwargs_default_3d, "stride": [2, 3, 4]},
    "3d_dilation": {**test_kwargs_default_3d, "dilation": [2, 3, 4]},
    "3d_groups": {**test_kwargs_default_3d, "groups": 4, "w_shape": [8, 1, 3, 3, 3]},
    "3d_NCHW_VECT_C": {
        **test_kwargs_default_3d,
        "tensor_format": 2,
        "w_shape": [8, 16, 3, 3, 3],
        "data_dtype": "int8x4",
    },
}


@pytest.fixture(
    params=[pytest.param(kwargs, id=name) for name, kwargs in conv_output_shape_conditions.items()]
)
def conv_output_shape_kwargs(request):
    return request.param


@tvm.testing.requires_gpu
@requires_cudnn
def test_conv_output_shape(conv_output_shape_kwargs):
    shape_from_cudnn = cudnn._conv_output_shape_from_cudnn(**conv_output_shape_kwargs)
    shape_from_python = cudnn.conv_output_shape(**conv_output_shape_kwargs)
    assert shape_from_cudnn == shape_from_python


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
