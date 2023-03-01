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

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend import get_patterns_with_prefix
from tvm.script import relax as R


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


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
        data: R.Tensor((16, 32, 32, 8), "float16"),
        weight1: R.Tensor((8, 3, 3, 8), "float16"),
        weight2: R.Tensor((8, 3, 3, 8), "float16"),
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


def build_and_run(mod, inputs_np, target, legalize=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    dev = tvm.device(target, 0)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def get_result_with_relax_cutlass_offload(mod, *args):
    patterns = [(entry.name, entry.pattern) for entry in get_patterns_with_prefix("cutlass")]

    assert len(patterns) != 0, "Cannot find cutlass patterns"

    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True),
            relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}}),
        ]
    )

    return build_and_run(seq(mod), args, "cuda")


def test_conv2d_offload():
    data = np.random.randint(low, high, size=(16, 32, 32, 16)).astype("float16")
    weight = np.random.randint(low, high, size=(32, 3, 3, 16)).astype("float16")
    bias = np.random.randint(low, high, size=(1, 1, 1, 32)).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dBiasReLU, data, weight, bias)

    ref = build_and_run(Conv2dBiasReLU, [data, weight, bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def get_relax_matmul_module(x, y, transposed_y=False, with_bias=False, activation=None):
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

            R.func_ret_value(frame.output_vars[0])

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
        (64, 128, 1024),
    ]
)
def matmul_size(request):
    return request.param


low, high = -10, 10


@pytest.fixture
def matmul_x(matmul_size, target_dtype):
    m, k, _ = matmul_size
    return np.random.randint(low, high, size=(m, k)).astype(target_dtype)


@pytest.fixture
def matmul_y(matmul_size, target_dtype):
    _, k, n = matmul_size
    return np.random.randint(low, high, size=(k, n)).astype(target_dtype)


@pytest.fixture
def matmul_bias(matmul_size, target_dtype):
    _, _, n = matmul_size
    return np.random.randint(low, high, size=(n,)).astype(target_dtype)


def test_matmul_offload(matmul_x, matmul_y):
    x, y = matmul_x, matmul_y

    mod = get_relax_matmul_module(x, y)
    out = get_result_with_relax_cutlass_offload(mod, x, y)
    ref = build_and_run(mod, [x, y], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_bias_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(x, y, with_bias=True)
    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)
    ref = build_and_run(mod, [x, y, bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_bias_relu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(x, y, with_bias=True, activation=R.nn.relu)
    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)
    ref = build_and_run(mod, [x, y, bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_bias_gelu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias
    mod = get_relax_matmul_module(x, y, with_bias=True, activation=R.nn.gelu)

    out = get_result_with_relax_cutlass_offload(mod, x, y, bias)
    ref = build_and_run(mod, [x, y, bias], "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3)


def test_kernel_sharing():
    low, high = -1, 1
    data_np = np.random.randint(low, high, size=(16, 32, 32, 8)).astype("float16")
    weight1_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")
    weight2_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dx2, data_np, weight1_np, weight2_np)
    ref = build_and_run(Conv2dx2, [data_np, weight1_np, weight2_np], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_transposed_offload(matmul_x, matmul_y):
    x, y = matmul_x, matmul_y

    mod = get_relax_matmul_module(x, y.transpose(), transposed_y=True)
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose())
    ref = build_and_run(mod, [x, y.transpose()], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_transposed_bias_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=None
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)
    ref = build_and_run(mod, [x, y.transpose(), bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_transposed_bias_relu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=R.nn.relu
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)
    ref = build_and_run(mod, [x, y.transpose(), bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_matmul_transposed_bias_gelu_offload(matmul_x, matmul_y, matmul_bias):
    x, y, bias = matmul_x, matmul_y, matmul_bias

    mod = get_relax_matmul_module(
        x, y.transpose(), transposed_y=True, with_bias=True, activation=R.nn.gelu
    )
    out = get_result_with_relax_cutlass_offload(mod, x, y.transpose(), bias)
    ref = build_and_run(mod, [x, y.transpose(), bias], "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
