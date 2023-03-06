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
import tvm.topi.testing
from tvm import relax
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import relax as R
from tvm.script import tir as T


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

    mod = partition_for_cutlass(mod)
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    return build_and_run(mod, args, "cuda")


def test_conv2d_offload():
    low, high = -1, 1
    data = np.random.randint(low, high, size=(16, 32, 32, 16)).astype("float16")
    weight = np.random.randint(low, high, size=(32, 3, 3, 16)).astype("float16")
    bias = np.random.randint(low, high, size=(1, 1, 1, 32)).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dBiasReLU, data, weight, bias)

    ref = build_and_run(Conv2dBiasReLU, [data, weight, bias], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def test_kernel_sharing():
    low, high = -1, 1
    data_np = np.random.randint(low, high, size=(16, 32, 32, 8)).astype("float16")
    weight1_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")
    weight2_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")

    out = get_result_with_relax_cutlass_offload(Conv2dx2, data_np, weight1_np, weight2_np)
    ref = build_and_run(Conv2dx2, [data_np, weight1_np, weight2_np], "llvm", legalize=True)

    np.testing.assert_equal(out, ref)


def get_relax_matmul_module(
    x_shape, y_shape, dtype, transposed_y=False, with_bias=False, activation=None
):
    if transposed_y:
        n = y_shape[-2]
    else:
        n = y_shape[-1]

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            x = R.arg("x", R.Tensor(x_shape, dtype))
            y = R.arg("y", R.Tensor(y_shape, dtype))
            if with_bias:
                bias = R.arg("bias", R.Tensor((n,), dtype))

            with R.dataflow() as frame:
                if transposed_y:
                    axes = list(range(len(y_shape) - 2)) + [-1, -2]
                    y = R.emit(R.permute_dims(y, axes=axes))
                result = R.emit(R.matmul(x, y, out_dtype=dtype))
                if with_bias:
                    result = R.emit(result + bias)
                if activation is not None:
                    result = R.emit(activation(result))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def _to_concrete_shape(symbolic_shape, var_table):
    result = []
    for dim in symbolic_shape:
        if not isinstance(dim, tvm.tir.expr.Var):
            result.append(dim)
            continue

        if dim not in var_table:
            var_table[dim] = np.random.randint(10, 50)
        result.append(var_table[dim])

    return tuple(result)


_vars = {
    "a": tvm.tir.expr.Var("a", "int64"),
    "b": tvm.tir.expr.Var("b", "int64"),
}


_epilogue_table = {
    "none": (False, None),
    "bias": (True, None),
    "relu": (True, R.nn.relu),
    "gelu": (True, R.nn.gelu),
}


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue",
    [
        # Regular
        ((32, 6), (6, 16), False, "none"),
        ((_vars["a"], 6), (6, 16), False, "bias"),
        # Transposed
        ((4, 16), (16, 128), True, "relu"),
        ((35, 8), (8, 8), True, "gelu"),
        # 3D x 3D
        ((6, 32, 8), (6, 8, 10), False, "bias"),
        ((6, 32, 8), (6, 8, 10), True, "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "gelu"),
        # 3D x 2D
        ((6, 32, 8), (8, 10), False, "none"),
        ((_vars["a"], 32, 8), (8, 10), False, "bias"),
        ((10, 16, 8), (8, 10), True, "relu"),
        # 2D x 3D
        ((32, 8), (10, 8, 10), False, "relu"),
        ((32, 8), (_vars["a"], 8, 10), True, "gelu"),
        # ND x 2D
        ((3, 6, 32, 8), (8, 10), False, "bias"),
        ((_vars["a"], _vars["b"], 6, 32, 8), (8, 10), False, "none"),
        # 2D x ND
        ((32, 8), (5, 3, 8, 10), False, "gelu"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 10), True, "relu"),
        ((3, 2, 4, 16, 15), (1, 1, 15, 2), True, "gelu"),
        ((1, 1, 16, 15), (3, 2, _vars["a"], 15, 2), False, "none"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
    ],
)
def test_matmul_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    dtype,
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(dtype)
    y = np.random.randn(*concrete_y_shape).astype(dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
        with_bias=with_bias,
        transposed_y=transpose_y,
        activation=activation,
    )
    out = get_result_with_relax_cutlass_offload(mod, *args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "x_shape, y_shape, expected",
    [
        # Regular matmul
        ((3, 4), (4, 5), True),
        # Batch matmul without stretching
        ((3, 16, 15), (3, 15, 2), True),
        ((_vars["a"], 16, 15), (_vars["a"], 15, 2), True),
        # Broadcast 2D to 3D
        ((3, 16, 15), (15, 2), True),
        ((_vars["a"], 16, 15), (15, 2), True),
        ((16, 15), (3, 15, 2), True),
        # Broadcast one-length dimension
        ((1, 16, 15), (3, 15, 2), True),
        ((3, 16, 15), (1, 15, 2), True),
        ((1, 1, 16, 15), (3, 2, 4, 15, 2), True),
        ((1, 1, 16, 15), (3, _vars["a"], 4, 15, 2), True),
        # ND x ND
        ((3, 2, 4, 16, 15), (3, 2, 4, 15, 2), True),
        ((_vars["a"], 2, 4, 16, 15), (_vars["a"], 2, 4, 15, 2), True),
        (
            (_vars["a"], _vars["b"], 4, 16, 15),
            (_vars["a"], _vars["b"], 4, 15, 2),
            True,
        ),
        # ND x ND with one-length dimension
        ((1, 2, 4, 16, 15), (1, 2, 4, 15, 2), True),
        ((3, 2, 1, 16, 15), (3, 2, 1, 15, 2), True),
        # Extra one-length dimension doesn't block broadcasting
        ((3, 2, 1, 16, 15), (1, 1, 3, 2, 1, 15, 2), True),
        # Not broadcasting all dims. Cannot be computed by stride-based batch gemm
        ((3, 1, 1, 16, 15), (3, 2, 4, 15, 2), False),
        ((3, 2, 4, 16, 15), (2, 4, 15, 2), False),
        # Different shape
        ((3, 4, 16, 15), (3, 2, 15, 2), False),
        ((3, _vars["a"], 16, 15), (3, _vars["b"], 15, 2), False),
        # Cannot prove that broadcast dimensions are equal
        ((_vars["a"], 16, 15), (3, 15, 2), False),
        ((3, _vars["a"], 1, 16, 15), (1, 1, 3, 2, 1, 15, 2), False),
        # Reduction axis must be constant
        ((3, _vars["a"]), (_vars["a"], 5), False),
    ],
)
def test_is_shape_valid_for_cutlass_matmul(x_shape, y_shape, expected):
    assert is_shape_valid_for_cutlass_matmul(x_shape, y_shape) == expected


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, dtype",
    [
        # Not broadcasting all dims. Cannot be computed by stride-based batch gemm
        ((3, 1, 1, 16, 15), (3, 2, 4, 15, 2), False, "float16"),
        ((3, 2, _vars["a"], 16, 15), (3, 2, 4, 15, 2), False, "float16"),
        ((1, 2, 1, 16, 15), (2, 1, 4, 15, 2), False, "float16"),
        ((3, 2, 4, 16, 15), (2, 4, 15, 2), True, "float16"),
        ((3, 16, 15), (2, 1, 3, 15, 2), True, "float16"),
        ((3, 16, 15), (_vars["a"], 1, 3, 15, 2), True, "float16"),
        ((_vars["a"], 1, 3, 16, 15), (_vars["b"], 1, 3, 15, 2), True, "float16"),
        ((_vars["a"], _vars["b"], 3, 16, 15), (_vars["a"], 1, 3, 15, 2), True, "float16"),
    ],
)
def test_cutlass_partition_matmul_blocked(x_shape, y_shape, transpose_y, dtype):
    if transpose_y:
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    mod = get_relax_matmul_module(
        x_shape, y_shape, dtype, with_bias=False, transposed_y=transpose_y
    )

    assert len(mod.functions) == 1


@pytest.fixture(params=["float16", "float32"])
def attention_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        # B, S, N, H
        (32, (8, 8), 16, (8, 8)),
        (4, (16, 8), 32, (8, 8)),  # s != s_kv
        (4, (16, 8), 32, (8, 16)),  # h != h_v
        (32, (8, 8), 16, (4, 4)),  # h is not aligned
        (2, (8, 8), 8, (256, 256)),  # needs output accumulator buffer
    ]
)
def attention_size(request):
    return request.param


def get_relax_attention_module(q, k, v, bias=None):
    dtype = str(q.dtype)

    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import relax as relax_builder

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            q = R.arg("q", R.Tensor(q.shape, dtype))
            k = R.arg("k", R.Tensor(k.shape, dtype))
            v = R.arg("v", R.Tensor(v.shape, dtype))
            if bias is not None:
                bias = R.arg("bias", R.Tensor(bias.shape, dtype))
            with R.dataflow() as frame:
                result = R.emit(R.nn.attention(q, k, v, bias))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


@memoize("topi.tests.test_codegen_cutlass.test_attention_offload")
def get_numpy_attention_ref(b, s, s_kv, n, h, h_v, dtype):
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    attn = tvm.topi.testing.softmax_python(score, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return q, k, v, ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_attention_offload(attention_size, attention_dtype):
    b, (s, s_kv), n, (h, h_v) = attention_size
    q, k, v, ref = get_numpy_attention_ref(b, s, s_kv, n, h, h_v, attention_dtype)

    mod = get_relax_attention_module(q, k, v)
    out = get_result_with_relax_cutlass_offload(mod, q, k, v)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@memoize("topi.tests.test_codegen_cutlass.test_attention_bias_4d_offload")
def get_numpy_attention_bias_4d_ref(b, s, s_kv, n, h, h_v, dtype):
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    bias = np.random.randn(b, n, s, s_kv).astype(dtype)
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    score_bias = score + bias  # b, n, s, s_kv
    attn = tvm.topi.testing.softmax_python(score_bias, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return q, k, v, bias, ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_attention_bias_4d_offload(attention_size, attention_dtype):
    b, (s, s_kv), n, (h, h_v) = attention_size
    q, k, v, bias, ref = get_numpy_attention_bias_4d_ref(b, s, s_kv, n, h, h_v, attention_dtype)

    mod = get_relax_attention_module(q, k, v, bias)
    out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@memoize("topi.tests.test_codegen_cutlass.test_attention_bias_3d_offload")
def get_numpy_attention_bias_3d_ref(b, s, s_kv, n, h, h_v, dtype):
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    bias = np.random.randn(b, s, s_kv).astype(dtype)
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    score_bias = score + bias.reshape(b, 1, s, s_kv)  # b, n, s, s_kv
    attn = tvm.topi.testing.softmax_python(score_bias, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return q, k, v, bias, ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_attention_bias_3d_offload(attention_size, attention_dtype):
    b, (s, s_kv), n, (h, h_v) = attention_size
    q, k, v, bias, ref = get_numpy_attention_bias_3d_ref(b, s, s_kv, n, h, h_v, attention_dtype)

    mod = get_relax_attention_module(q, k, v, bias)
    out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@memoize("topi.tests.test_codegen_cutlass.test_attention_bias_2d_offload")
def get_numpy_attention_bias_2d_ref(b, s, s_kv, n, h, h_v, dtype):
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, n, h).astype(dtype)
    v = np.random.randn(b, s_kv, n, h_v).astype(dtype)
    bias = np.random.randn(b, s_kv).astype(dtype)
    qt = q.transpose(0, 2, 1, 3)  # b, n, s, h
    kt = k.transpose(0, 2, 3, 1)  # b, n, h, s_kv
    score = qt @ kt / np.sqrt(q.shape[-1])  # b, n, s, s_kv
    score_bias = score + bias.reshape(b, 1, 1, s_kv)  # b, n, s, s_kv
    attn = tvm.topi.testing.softmax_python(score_bias, -1)
    vt = v.transpose(0, 2, 1, 3)  # b, n, s_kv, h_v
    ref = attn @ vt  # b, n, s, h_v
    return q, k, v, bias, ref.transpose(0, 2, 1, 3)  # b, s, n, h_v


def test_attention_bias_2d_offload(attention_size, attention_dtype):
    b, (s, s_kv), n, (h, h_v) = attention_size
    q, k, v, bias, ref = get_numpy_attention_bias_2d_ref(b, s, s_kv, n, h, h_v, attention_dtype)

    mod = get_relax_attention_module(q, k, v, bias)
    out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
