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
import pytest
from tvm import topi
import tvm.topi.testing
import tvm
from tvm import te
from tvm import relay
from tvm.relay.testing import check_grad, run_infer_type, run_opt_pass
from tvm.relay.transform import gradient
import tvm.testing

executor_kind = tvm.testing.parameter("debug")


def verify_max_pool2d_grad(executor_kind, x_shape, pool_size, strides, padding, ceil_mode):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.max_pool2d(
        x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode
    )

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.utils.get_const_tuple(fwd_func.ret_type.shape)
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

    for target, dev in tvm.testing.enabled_targets():
        op_res, (op_grad,) = relay.create_executor(
            executor_kind, device=dev, target=target
        ).evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.numpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_max_pool2d_grad(executor_kind):
    verify_max_pool2d_grad(
        executor_kind,
        (1, 4, 16, 16),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
    )
    verify_max_pool2d_grad(
        executor_kind,
        (1, 4, 16, 16),
        pool_size=(1, 1),
        strides=(1, 1),
        padding=(1, 1),
        ceil_mode=False,
    )


def verify_avg_pool2d_grad(
    x_shape,
    pool_size,
    strides,
    padding,
    ceil_mode,
    count_include_pad,
    executor_kind,
    dtype="float32",
):

    for shape_dtype in ["int32", "int64"]:
        x = relay.var("x", shape=[tvm.tir.IntImm(shape_dtype, x) for x in x_shape], dtype=dtype)
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

        data = np.random.rand(*x_shape).astype(dtype)
        ph, pw = padding
        y_shape = topi.utils.get_const_tuple(fwd_func.ret_type.shape)
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

        for target, dev in tvm.testing.enabled_targets():
            op_res, (op_grad,) = relay.create_executor(
                executor_kind, device=dev, target=target
            ).evaluate(bwd_func)(data)
            np.testing.assert_allclose(op_grad.numpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_avg_pool2d_grad(executor_kind):
    verify_avg_pool2d_grad(
        (1, 4, 16, 16),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=True,
        executor_kind=executor_kind,
    )
    verify_avg_pool2d_grad(
        (1, 4, 16, 16),
        pool_size=(1, 1),
        strides=(1, 1),
        padding=(1, 1),
        ceil_mode=False,
        count_include_pad=False,
        executor_kind=executor_kind,
    )
    verify_avg_pool2d_grad(
        (1, 4, 16, 16),
        pool_size=(1, 1),
        strides=(1, 1),
        padding=(1, 1),
        ceil_mode=False,
        count_include_pad=False,
        executor_kind=executor_kind,
        dtype="float16",
    )


def verify_global_avg_pool2d_grad(executor_kind, x_shape):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.global_avg_pool2d(x)

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    y_shape = topi.utils.get_const_tuple(fwd_func.ret_type.shape)
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

    for target, dev in tvm.testing.enabled_targets():
        op_res, (op_grad,) = relay.create_executor(
            executor_kind, device=dev, target=target
        ).evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.numpy(), ref_grad, rtol=0.01)


@tvm.testing.uses_gpu
def test_global_avg_pool2d_grad(executor_kind):
    verify_global_avg_pool2d_grad(executor_kind, (1, 4, 16, 16))
    verify_global_avg_pool2d_grad(executor_kind, (1, 8, 8, 24))


def verify_conv2d_grad(
    dshape, wshape, strides, padding, dilation, groups=1, mode="higher_order", executor_kind="vm"
):
    dtype = "float32"
    data = relay.var("data", shape=dshape, dtype=dtype)
    weight = relay.var("weight", shape=wshape, dtype=dtype)
    conv = relay.nn.conv2d(
        data,
        weight,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        out_dtype=dtype,
    )
    fwd_func = relay.Function([data, weight], conv)
    check_grad(fwd_func, mode=mode, executor_kind=executor_kind)


@tvm.testing.uses_gpu
def test_conv2d_grad(executor_kind):
    verify_conv2d_grad(
        (1, 4, 16, 16), (16, 4, 3, 3), [1, 1], [1, 1], [1, 1], executor_kind=executor_kind
    )
    verify_conv2d_grad(
        (1, 4, 16, 16), (16, 4, 1, 1), [1, 1], [0, 0], [1, 1], executor_kind=executor_kind
    )
    verify_conv2d_grad(
        (1, 4, 16, 16), (16, 4, 1, 1), [2, 2], [0, 0], [1, 1], executor_kind=executor_kind
    )
    verify_conv2d_grad(
        (1, 4, 16, 16),
        (16, 4, 3, 3),
        [1, 1],
        [1, 1],
        [1, 1],
        mode="first_order",
        executor_kind=executor_kind,
    )


def verify_dense_grad(d_shape, w_shape, executor_kind):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(w_shape, "float32"))
    fwd_func = relay.Function([data, weight], relay.nn.dense(data, weight))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_dense_grad(executor_kind):
    verify_dense_grad((1, 8), (16, 8), executor_kind)
    verify_dense_grad((1, 4), (3, 4), executor_kind)
    verify_dense_grad((5, 4), (3, 4), executor_kind)


def verify_matmul_grad(a_shape, b_shape, transpose_a, transpose_b, executor_kind):
    tensor_a = relay.var("tensor_a", relay.TensorType(a_shape, "float32"))
    tensor_b = relay.var("tensor_b", relay.TensorType(b_shape, "float32"))
    fwd_func = relay.Function(
        [tensor_a, tensor_b],
        relay.nn.matmul(tensor_a, tensor_b, transpose_a=transpose_a, transpose_b=transpose_b),
    )
    check_grad(fwd_func, executor_kind=executor_kind)


def test_matmul_grad(executor_kind):
    verify_matmul_grad((1, 8), (8, 16), False, False, executor_kind)
    verify_matmul_grad((4, 1), (4, 3), True, False, executor_kind)
    verify_matmul_grad((4, 5), (3, 4), True, True, executor_kind)


def verify_batch_flatten_grad(d_shape, executor_kind):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], relay.nn.batch_flatten(data))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_batch_flatten_grad(executor_kind):
    verify_batch_flatten_grad((1, 2, 3, 4), executor_kind)
    verify_batch_flatten_grad((1, 8), executor_kind)


def verify_conv2d_backward_weight(
    executor_kind, dy_shape, x_shape, kernel_size, stride, padding, groups=1, out_channels=None
):
    dtype = "float32"
    dy = relay.var("dy", shape=dy_shape, dtype=dtype)
    x = relay.var("x", shape=x_shape, dtype=dtype)
    dw_func = relay.Function(
        [dy, x],
        relay.nn.conv2d_backward_weight(
            dy,
            x,
            strides=stride,
            padding=padding,
            kernel_size=kernel_size,
            groups=groups,
            channels=out_channels,
            out_dtype=dtype,
        ),
    )

    dw_func_legalized = run_opt_pass(dw_func, relay.transform.Legalize())

    for dw, target in [(dw_func_legalized, "llvm"), (dw_func, "cuda -libs=cudnn")]:
        if "cudnn" in target and not tvm.contrib.cudnn.exists():
            continue

        dev = tvm.device(target, 0)
        dy_np = np.random.randn(*dy_shape).astype(dtype)
        x_np = np.random.randn(*x_shape).astype(dtype)

        dw_np = (
            relay.create_executor(executor_kind, device=dev, target=target)
            .evaluate(dw)(dy_np, x_np)
            .numpy()
        )
        ref_dw_np = tvm.topi.testing.conv2d_backward_weight_python(
            dy_np, x_np, kernel_size, stride, padding, groups=groups, channels=out_channels
        )

        np.testing.assert_allclose(dw_np, ref_dw_np, rtol=1e-4, atol=1e-4)


def test_conv2d_backward_weight(executor_kind):
    verify_conv2d_backward_weight(
        executor_kind, (2, 8, 32, 32), (2, 4, 32, 32), (3, 3), (1, 1), (1, 1)
    )
    verify_conv2d_backward_weight(
        executor_kind, (2, 16, 15, 15), (2, 3, 32, 32), (3, 3), (2, 2), (0, 0)
    )
    verify_conv2d_backward_weight(
        executor_kind,
        (1, 16, 32, 32),
        (1, 16, 32, 32),
        (3, 3),
        (1, 1),
        (1, 1),
        groups=16,
        out_channels=16,
    )


def test_conv2d_backward_weight_infer_type():
    # From https://github.com/apache/tvm/pull/10439
    depthwise_conv_code = """
    fn (%input0: Tensor[(1, 3, 32, 32), float32], %v0_weight: Tensor[(3, 1, 3, 3), float32], %v0_bias: Tensor[(3), float32]) {
      %0 = nn.conv2d(%input0, %v0_weight, padding=[1, 1, 1, 1], groups=3, channels=3, kernel_size=[3, 3]);
      nn.bias_add(%0, %v0_bias)
    }
    """

    normal_conv_code = """
    fn (%input0: Tensor[(1, 3, 32, 32), float32], %v0_weight: Tensor[(3, 3, 3, 3), float32], %v0_bias: Tensor[(3), float32]) {
      %0 = nn.conv2d(%input0, %v0_weight, padding=[1, 1, 1, 1], groups=1, channels=3, kernel_size=[3, 3]);
      nn.bias_add(%0, %v0_bias)
    }
    """

    SEMVER = '#[version = "0.0.5"]\n'

    for code in [normal_conv_code, depthwise_conv_code]:
        expr = tvm.relay.parse_expr(SEMVER + code)
        fmod = tvm.IRModule.from_expr(expr)

        mod = relay.transform.InferType()(fmod)
        bwd_expr = relay.transform.gradient(mod["main"], mode="first_order")

        bwd_mod = tvm.IRModule.from_expr(bwd_expr)
        bwd_mod = relay.transform.InferType()(bwd_mod)


if __name__ == "__main__":
    tvm.testing.main()
