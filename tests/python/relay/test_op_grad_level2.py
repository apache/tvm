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
import numpy as np

from tvm import topi
import tvm.topi.testing
import tvm
from tvm import te
from tvm import relay
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
import tvm.testing


def verify_max_pool2d_grad(x_shape, pool_size, strides, padding, ceil_mode):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.max_pool2d(
        x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode
    )

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    ref_grad = tvm.topi.testing.pool_grad_nchw(
        data,
        out_grad,
        pool_size=pool_size,
        strides=strides,
        padding=[ph, pw, ph, pw],
        pool_type="max",
        ceil_mode=ceil_mode,
    )

    for target, ctx in tvm.testing.enabled_targets():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad,) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_max_pool2d_grad():
    verify_max_pool2d_grad(
        (1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0), ceil_mode=False
    )
    verify_max_pool2d_grad(
        (1, 4, 16, 16), pool_size=(1, 1), strides=(1, 1), padding=(1, 1), ceil_mode=False
    )


def verify_avg_pool2d_grad(x_shape, pool_size, strides, padding, ceil_mode, count_include_pad):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.avg_pool2d(
        x,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    ref_grad = tvm.topi.testing.pool_grad_nchw(
        data,
        out_grad,
        pool_size=pool_size,
        strides=strides,
        padding=[ph, pw, ph, pw],
        pool_type="avg",
        ceil_mode=ceil_mode,
    )

    for target, ctx in tvm.testing.enabled_targets():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad,) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_avg_pool2d_grad():
    verify_avg_pool2d_grad(
        (1, 4, 16, 16),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=True,
    )
    verify_avg_pool2d_grad(
        (1, 4, 16, 16),
        pool_size=(1, 1),
        strides=(1, 1),
        padding=(1, 1),
        ceil_mode=False,
        count_include_pad=False,
    )


def verify_global_avg_pool2d_grad(x_shape):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.global_avg_pool2d(x)

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    ref_grad = tvm.topi.testing.pool_grad_nchw(
        data,
        out_grad,
        pool_size=(x_shape[2], x_shape[3]),
        strides=(1, 1),
        padding=[0, 0, 0, 0],
        pool_type="avg",
        ceil_mode=False,
    )

    for target, ctx in tvm.testing.enabled_targets():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad,) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_global_avg_pool2d_grad():
    verify_global_avg_pool2d_grad((1, 4, 16, 16))
    verify_global_avg_pool2d_grad((1, 8, 8, 24))


def verify_conv2d_grad(dshape, wshape, strides, padding, dilation, groups=1, mode="higher_order"):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Skip because pytorch is not installed")
        return

    dtype = "float32"
    data = relay.var("data", shape=dshape, dtype=dtype)
    weight = relay.var("weight", shape=wshape, dtype=dtype)
    conv = relay.nn.conv2d(
        data, weight, strides=strides, padding=padding, dilation=dilation, groups=groups
    )
    fwd_func = relay.Function([data, weight], conv)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func, mode=mode))

    data_pt = torch.randn(*dshape, dtype=torch.float32, requires_grad=True)
    weight_pt = torch.randn(*wshape, dtype=torch.float32, requires_grad=True)
    out_pt = F.conv2d(
        data_pt, weight_pt, stride=strides, padding=padding, dilation=dilation, groups=groups
    )
    grad_output_pt = torch.ones(out_pt.shape)
    grad_input_pt = (
        F.grad.conv2d_input(
            dshape,
            weight_pt,
            grad_output_pt,
            stride=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        .detach()
        .numpy()
    )
    grad_weight_pt = (
        F.grad.conv2d_weight(
            data_pt,
            wshape,
            grad_output_pt,
            stride=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        .detach()
        .numpy()
    )

    for target, ctx in tvm.testing.enabled_targets():
        data = tvm.nd.array(data_pt.detach().numpy(), ctx)
        weight = tvm.nd.array(weight_pt.detach().numpy(), ctx)
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (grad_input, grad_weight) = intrp.evaluate(bwd_func)(data, weight)
        np.testing.assert_allclose(grad_input.asnumpy(), grad_input_pt, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(grad_weight.asnumpy(), grad_weight_pt, rtol=1e-4, atol=1e-4)


@tvm.testing.uses_gpu
def test_conv2d_grad():
    verify_conv2d_grad((1, 4, 16, 16), (16, 4, 3, 3), [1, 1], [1, 1], [1, 1])
    verify_conv2d_grad((1, 4, 16, 16), (16, 4, 1, 1), [1, 1], [0, 0], [1, 1])
    verify_conv2d_grad((1, 4, 16, 16), (16, 4, 1, 1), [2, 2], [0, 0], [1, 1])
    verify_conv2d_grad((1, 4, 16, 16), (16, 4, 3, 3), [1, 1], [1, 1], [1, 1], mode="first_order")


def verify_dense_grad(d_shape, w_shape):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(w_shape, "float32"))
    fwd_func = relay.Function([data, weight], relay.nn.dense(data, weight))
    check_grad(fwd_func)


def test_dense_grad():
    verify_dense_grad((1, 8), (16, 8))
    verify_dense_grad((1, 4), (3, 4))
    verify_dense_grad((5, 4), (3, 4))


def verify_batch_flatten_grad(d_shape):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], relay.nn.batch_flatten(data))
    check_grad(fwd_func)


def test_batch_flatten_grad():
    verify_batch_flatten_grad((1, 2, 3, 4))
    verify_batch_flatten_grad((1, 8))


if __name__ == "__main__":
    test_max_pool2d_grad()
    test_avg_pool2d_grad()
    test_global_avg_pool2d_grad()
    test_conv2d_grad()
    test_dense_grad()
    test_batch_flatten_grad()
