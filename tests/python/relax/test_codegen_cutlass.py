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
import math
from typing import List, Tuple

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax, relay
from tvm.relax.backend import get_patterns_with_prefix
from tvm.script import relax as R


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


def get_relay_conv2d_bias_relu(
    d_shape, w_shape, data_dtype="float16", weight_dtype="float16", out_dtype="float16"
):
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    bias = relay.var("bias", shape=(1, 1, 1, w_shape[0]), dtype=out_dtype)
    return relay.nn.relu(
        relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_dtype=out_dtype,
        )
        + bias
    )


def get_relay_matmul(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    out_dtype="float16",
):
    x = relay.var("x", shape=x_shape, dtype=x_dtype)
    y = relay.var("y", shape=y_shape, dtype=y_dtype)
    return relay.nn.dense(x, y, out_dtype=out_dtype)


def get_relay_matmul_bias(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    bias = relay.var("bias", shape=(y_shape[0],), dtype=bias_dtype)
    return relay.nn.bias_add(
        get_relay_matmul(
            x_shape,
            y_shape,
            x_dtype,
            y_dtype,
            out_dtype,
        ),
        bias,
    )


def get_relay_matmul_bias_relu(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    return relay.nn.relu(
        get_relay_matmul_bias(
            x_shape,
            y_shape,
            x_dtype,
            y_dtype,
            bias_dtype,
            out_dtype,
        )
    )


def get_relay_matmul_bias_gelu(
    x_shape,
    y_shape,
    x_dtype="float16",
    y_dtype="float16",
    bias_dtype="float16",
    out_dtype="float16",
):
    bias_add = get_relay_matmul_bias(x_shape, y_shape, x_dtype, y_dtype, bias_dtype, out_dtype)
    mul = bias_add * relay.const((1.0 / math.sqrt(2.0)), dtype=out_dtype)
    if out_dtype == "float16":
        erf = relay.cast(relay.op.erf(relay.cast(mul, "float32")), "float16")
    else:
        erf = relay.op.erf(mul)
    mul_half = erf * relay.const(0.5, dtype=out_dtype)
    add = mul_half + relay.const(0.5, dtype=out_dtype)
    return add * bias_add


def get_relay_conv2d_relu_x2(
    d_shape, w_shape, data_dtype="float16", weight_dtype="float16", out_dtype="float16"
):
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight1 = relay.var("weight1", shape=w_shape, dtype=weight_dtype)
    weight2 = relay.var("weight2", shape=w_shape, dtype=weight_dtype)

    conv1 = relay.nn.conv2d(
        data=data,
        weight=weight1,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="OHWI",
        out_dtype=out_dtype,
    )
    return relay.nn.conv2d(
        data=conv1,
        weight=weight2,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="OHWI",
        out_dtype=out_dtype,
    )


def get_relay_ref(relay_expr, *args):
    relay_mod = tvm.IRModule.from_expr(relay_expr)

    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "HWIO"]})]
        )
        relay_mod = seq(relay_mod)

    return (
        relay.create_executor("graph", mod=relay_mod, device=tvm.gpu(0), target="cuda")
        .evaluate()(*args)
        .numpy()
    )


@tvm.script.ir_module
class Conv2dBiasReLU:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight: R.Tensor((32, 3, 3, 16), "float16"),
        bias: R.Tensor((1, 1, 1, 32), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.relu(
                relax.op.add(
                    relax.op.nn.conv2d(
                        data, weight, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    ),
                    bias,
                )
            )
            R.output(conv1)

        return conv1


@tvm.script.ir_module
class Conv2dx2:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight1: R.Tensor((16, 3, 3, 16), "float16"),
        weight2: R.Tensor((16, 3, 3, 16), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.conv2d(
                data, weight1, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            conv2 = relax.op.nn.conv2d(
                conv1, weight2, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            R.output(conv2)

        return conv2


has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

cutlass_enabled = pytest.mark.skipif(
    not has_cutlass,
    reason="CUTLASS not enabled.",
)

pytestmark = [cutlass_enabled]


def get_result_with_relax_cutlass_offload(mod, *args):
    patterns = [(entry.name, entry.pattern) for entry in get_patterns_with_prefix("cutlass")]

    assert len(patterns) != 0, "Cannot find cutlass patterns"

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True),
            relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}}),
        ]
    )

    mod = seq(mod)

    target = tvm.target.Target("cuda")
    ex = relax.build(mod, target)

    dev = tvm.gpu(0)
    vm = relax.VirtualMachine(ex, dev)

    return vm["main"](*(tvm.nd.array(arg, dev) for arg in args)).numpy()


def test_conv2d_offload():
    data = np.random.randn(16, 32, 32, 16).astype("float16")
    weight = np.random.randn(32, 3, 3, 16).astype("float16")
    bias = np.random.randn(1, 1, 1, 32).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dBiasReLU, data, weight, bias)

    ref_relay_expr = get_relay_conv2d_bias_relu(data.shape, weight.shape)
    ref = get_relay_ref(ref_relay_expr, data, weight, bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def get_relax_matmul_module(x, y, transposed_y=False, with_bias=False, activation=None):
    m, k = x.shape
    if transposed_y:
        n = y.shape[-2]
    else:
        n = y.shape[-1]
    dtype = str(x.dtype)

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            x = R.arg("x", R.Tensor(x.shape, dtype))
            y = R.arg("y", R.Tensor(y.shape, dtype))
            if with_bias:
                bias = R.arg("bias", R.Tensor((n,), dtype))

            with R.dataflow() as frame:
                if transposed_y:
                    y = R.emit(R.permute_dims(y))
                result = R.emit(R.matmul(x, y, out_dtype=dtype))
                if with_bias:
                    result = R.emit(result + bias)
                if activation is not None:
                    result = R.emit(activation(result))
                R.output(result)

            R.ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


@pytest.fixture(params=["float16"])
def target_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        # M, K, N
        (32, 6, 16),
        (29, 17, 19),
        (64, 512, 1024),
    ]
)
def matmul_size(request):
    return request.param


@pytest.fixture
def matmul_x(matmul_size, target_dtype):
    m, k, _ = matmul_size
    return np.random.randn(m, k).astype(target_dtype)


@pytest.fixture
def matmul_y(matmul_size, target_dtype):
    _, k, n = matmul_size
    return np.random.randn(k, n).astype(target_dtype)


@pytest.fixture
def matmul_bias(matmul_size, target_dtype):
    _, _, n = matmul_size
    return np.random.randn(n).astype(target_dtype)


def test_matmul_offload(matmul_x, matmul_y):
    x, y = matmul_x, matmul_y

    mod = get_relax_matmul_module(x, y)
    out = get_result_with_relax_cutlass_offload(mod, x, y)
    ref_relay_expr = get_relay_matmul(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose())

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_bias_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(x, y, with_bias=True)
    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_bias_relu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(x, y, with_bias=True, activation=R.nn.relu)
    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias_relu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_bias_gelu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(x, y, with_bias=True, activation=R.nn.gelu)
    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)

    ref_relay_expr = get_relay_matmul_bias_gelu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3)


def test_kernel_sharing():
    data_np = np.random.randn(16, 32, 32, 16).astype("float16")
    weight1_np = np.random.randn(16, 3, 3, 16).astype("float16")
    weight2_np = np.random.randn(16, 3, 3, 16).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dx2, data_np, weight1_np, weight2_np)

    relay_expr = get_relay_conv2d_relu_x2(data_np.shape, weight1_np.shape)
    ref = get_relay_ref(relay_expr, data_np, weight1_np, weight2_np)

    tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_matmul_transposed_offload(matmul_x, matmul_y):
    x, y = matmul_x, matmul_y

    mod = get_relax_matmul_module(x, y.transpose(), transposed_y=True)
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose())
    ref_relay_expr = get_relay_matmul(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose())

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_transposed_bias_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=None
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)

    ref_relay_expr = get_relay_matmul_bias(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_transposed_bias_relu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=R.nn.relu
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)

    ref_relay_expr = get_relay_matmul_bias_relu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_matmul_transposed_bias_gelu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=R.nn.gelu
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)

    ref_relay_expr = get_relay_matmul_bias_gelu(x.shape, y.shape[::-1])
    ref = get_relay_ref(ref_relay_expr, x, y.transpose(), bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
