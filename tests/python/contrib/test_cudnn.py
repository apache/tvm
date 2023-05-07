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
import tvm.testing
from tvm import te
from tvm import relay
from tvm.contrib import cudnn
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import graph_executor
import numpy as np
import tvm.topi.testing
import tvm.testing
from tvm.relay.op.contrib.cudnn import partition_for_cudnn


requires_cudnn = pytest.mark.skipif(
    tvm.get_global_func("tvm.contrib.cudnn.conv2d.forward", True) is None,
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
    verify_conv2d("float16", "float16", tensor_format=0)
    verify_conv2d("float16", "float16", tensor_format=1)
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


def verify_conv2d_backward_data(data_dtype, conv_dtype, tensor_format=0, tol=1e-5):
    batch = 3
    in_channel = 4
    out_channel = 16
    filter_h, filter_w = 3, 3
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1
    height, width = 32, 32

    if tensor_format == 0:
        xshape = [batch, in_channel, height, width]
        wshape = [out_channel, in_channel, filter_h, filter_w]
        oshape = xshape
        oshape[1] = out_channel
        ref_func = tvm.topi.testing.conv2d_transpose_nchw_python
    else:
        xshape = [batch, height, width, in_channel]
        wshape = [out_channel, filter_h, filter_w, in_channel]
        oshape = xshape
        oshape[3] = out_channel
        ref_func = lambda dy_np, w_np, strides, padding, out_pad: tvm.topi.testing.conv2d_transpose_nhwc_python(
            dy_np, np.transpose(w_np, [1, 2, 3, 0]), "HWOI", strides, padding, out_pad
        )

    dy_np = np.random.uniform(-1, 1, oshape).astype(data_dtype)
    w_np = np.random.uniform(-1, 1, wshape).astype(data_dtype)

    if data_dtype == "float16":
        dx_np = ref_func(
            dy_np.astype("float32"),
            w_np.astype("float32"),
            (stride_h, stride_w),
            (pad_h, pad_w),
            (0, 0),
        )
        dx_np = dx_np.astype("float16")
    else:
        dx_np = ref_func(dy_np, w_np, (stride_h, stride_w), (pad_h, pad_w), (0, 0))

    dy = te.placeholder(oshape, name="dy", dtype=data_dtype)
    w = te.placeholder(wshape, name="dw", dtype=data_dtype)
    dx = cudnn.conv_backward_data(
        dy,
        w,
        [pad_h, pad_w],
        [stride_h, stride_w],
        [1, 1],
        conv_mode=1,
        tensor_format=tensor_format,
        conv_dtype=conv_dtype,
        groups=1,
    )

    s = te.create_schedule(dx.op)

    dev = tvm.cuda(0)
    f = tvm.build(s, [dy, w, dx], "cuda --host=llvm", name="conv2d_backward_data")

    dy = tvm.nd.array(dy_np, dev)
    w = tvm.nd.array(w_np, dev)
    dx = tvm.nd.array(dx_np, dev)

    f(dy, w, dx)
    tvm.testing.assert_allclose(dx.numpy(), dx_np, atol=tol, rtol=tol)


@tvm.testing.requires_gpu
@requires_cudnn
def test_conv2d_backward_data():
    verify_conv2d_backward_data("float32", "float32", tensor_format=0, tol=1e-5)
    verify_conv2d_backward_data("float32", "float32", tensor_format=1, tol=1e-2)
    # The scipy convolve function does not support fp16, so the reference will be computed with
    # fp32. Use larger tolerance to be on the safe side (1e-2 also seems mostly ok).
    verify_conv2d_backward_data("float16", "float16", tensor_format=1, tol=1e-1)


def verify_conv2d_backward_filter(data_dtype, conv_dtype, tensor_format=0, tol=1e-5):
    batch = 3
    in_channel = 4
    out_channel = 16
    filter_h, filter_w = 3, 3
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1
    height, width = 32, 32

    if tensor_format == 0:
        x_shape = [batch, in_channel, height, width]
        dy_shape = [batch, out_channel, height, width]
    else:
        x_shape = [batch, height, width, in_channel]
        dy_shape = [batch, height, width, out_channel]

    x_np = np.random.uniform(-1, 1, x_shape).astype(data_dtype)
    dy_np = np.random.uniform(-1, 1, dy_shape).astype(data_dtype)

    dw_np = tvm.topi.testing.conv2d_backward_weight_python(
        dy_np,
        x_np,
        (filter_h, filter_w),
        (stride_h, stride_w),
        (pad_h, pad_w),
        "NCHW" if tensor_format == 0 else "NHWC",
    )

    x = te.placeholder(x_shape, name="x", dtype=data_dtype)
    dy = te.placeholder(dy_shape, name="dy", dtype=data_dtype)
    dw = cudnn.conv_backward_filter(
        dy,
        x,
        (filter_h, filter_w),
        [pad_h, pad_w],
        [stride_h, stride_w],
        [1, 1],
        conv_mode=1,
        tensor_format=tensor_format,
        conv_dtype=conv_dtype,
    )

    s = te.create_schedule(dw.op)

    dev = tvm.cuda(0)
    f = tvm.build(s, [dy, x, dw], "cuda --host=llvm", name="conv2d_backward_filter")

    x = tvm.nd.array(x_np, dev)
    dy = tvm.nd.array(dy_np, dev)
    dw = tvm.nd.array(dw_np, dev)

    f(dy, x, dw)
    tvm.testing.assert_allclose(dw.numpy(), dw_np, atol=tol, rtol=tol)


@tvm.testing.requires_gpu
@requires_cudnn
def test_conv2d_backward_filter():
    verify_conv2d_backward_filter("float32", "float32", tensor_format=0, tol=1e-2)
    verify_conv2d_backward_filter("float32", "float32", tensor_format=1, tol=1e-2)


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


def _verify_cudnn_relay(expr):
    np.random.seed(42)

    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    func = mod["main"]
    cudnn_mod = partition_for_cudnn(mod)
    assert len(cudnn_mod.get_global_vars()) == 2

    input_data = []
    for param in func.params:
        shape = [int(x) for x in param.checked_type.shape]
        input_data.append(
            (
                param.name_hint,
                np.random.uniform(-32, 32, size=shape).astype(param.checked_type.dtype),
            )
        )

    cuda_config = (tvm.target.cuda(), tvm.cuda(), cudnn_mod)
    cpu_config = (tvm.target.Target("llvm"), tvm.cpu(), mod)
    outputs = []
    for target, dev, test_mod in [cuda_config, cpu_config]:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(test_mod, target=target, target_host=cpu_config[0])
            module = graph_executor.GraphModule(lib["default"](dev))
            for name, data in input_data:
                module.set_input(name, tvm.nd.array(data, dev))

            module.run()
            out_type = func.body.checked_type
            outputs.append(
                module.get_output(0, tvm.nd.empty(out_type.shape, dtype=out_type.dtype)).numpy()
            )

    tvm.testing.assert_allclose(
        outputs[0],
        outputs[1],
        rtol=1e-2,
        atol=30,
    )


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((200,), 0),
        ((13, 27), 0),
        ((44, 12, 67), 1),
        ((1, 16, 16, 8), 2),
        ((2, 4, 6, 8, 10), 3),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float16",
        "float64",
    ],
)
def test_relay_cudnn_softmax(shape, axis, dtype):
    x = tvm.relay.var("x", tvm.relay.TensorType(shape, dtype))
    softmax = relay.op.nn.softmax(x, axis=axis)
    _verify_cudnn_relay(softmax)


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((32, 16), -1),
        ((13, 27), 1),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float16",
        "float64",
    ],
)
def test_relay_cudnn_log_softmax(shape, axis, dtype):
    x = tvm.relay.var("x", tvm.relay.TensorType(shape, dtype))
    log_softmax = relay.op.nn.log_softmax(x, axis=axis)
    _verify_cudnn_relay(log_softmax)


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "n,h,w,ci,co,groups",
    [
        (1, 16, 20, 8, 16, 1),
        (10, 17, 19, 16, 8, 4),
    ],
)
@pytest.mark.parametrize(
    "kh,kw,padding",
    [
        (1, 1, (3, 1, 3, 1)),
        (3, 3, (1, 2)),
        (7, 2, (0, 0)),
    ],
)
@pytest.mark.parametrize(
    "strides,dilation,dtype",
    [
        ((1, 1), (1, 1), "float32"),
        ((2, 1), (2, 2), "float16"),
        ((3, 3), (1, 2), "float64"),
    ],
)
def test_relay_cudnn_conv2d(n, h, w, ci, co, kh, kw, strides, dilation, padding, groups, dtype):
    data = tvm.relay.var("data", tvm.relay.TensorType((n, ci, h, w), dtype))
    weight = tvm.relay.var("weight", tvm.relay.TensorType((co, ci // groups, kh, kw), dtype))
    conv2d = relay.op.nn.conv2d(
        data,
        weight,
        groups=groups,
        channels=co,
        kernel_size=(kh, kw),
        strides=strides,
        dilation=dilation,
        padding=padding,
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    _verify_cudnn_relay(conv2d)


@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "n,h,w,ci,co,groups",
    [
        (1, 16, 20, 8, 16, 1),
        (10, 17, 19, 16, 8, 4),
    ],
)
@pytest.mark.parametrize(
    "kh,kw,padding,strides,dilation,dtype",
    [
        (1, 1, (3, 1, 3, 1), (1, 1), (1, 1), "float32"),
        (3, 3, (1, 2), (2, 1), (2, 2), "float16"),
        (7, 2, (0, 0), (3, 3), (1, 2), "float64"),
    ],
)
@pytest.mark.parametrize("activation", [True, False])
def test_relay_cudnn_conv2d_bias_act(
    n, h, w, ci, co, kh, kw, strides, dilation, padding, groups, dtype, activation
):
    data = tvm.relay.var("data", tvm.relay.TensorType((n, ci, h, w), dtype))
    weight = tvm.relay.var("weight", tvm.relay.TensorType((co, ci // groups, kh, kw), dtype))
    bias = relay.var("bias", relay.TensorType((co,), dtype))
    conv2d = relay.op.nn.conv2d(
        data,
        weight,
        groups=groups,
        channels=co,
        kernel_size=(kh, kw),
        strides=strides,
        dilation=dilation,
        padding=padding,
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    out = relay.op.nn.bias_add(conv2d, bias)
    if activation:
        out = relay.op.nn.relu(out)

    _verify_cudnn_relay(out)


if __name__ == "__main__":
    tvm.testing.main()
