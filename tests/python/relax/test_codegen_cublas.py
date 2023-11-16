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
from tvm.relax.backend.contrib.cublas import partition_for_cublas
from tvm.relax.testing import get_relax_matmul_module
from tvm.script import relax as R


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


has_cublas = tvm.get_global_func("relax.ext.cublas", True)

cublas_enabled = pytest.mark.skipif(
    not has_cublas,
    reason="CUBLAS not enabled.",
)

pytestmark = [cublas_enabled]


def build_and_run(mod, inputs_np, target, legalize=False, cuda_graph=False):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(
        config={
            "relax.backend.use_cuda_graph": cuda_graph,
            "relax.transform.apply_legalize_ops": legalize,
        }
    ):
        ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

    # For cuda graph, run the compiled function twice to make sure that we can launch the cached
    # graph on the second run.
    if cuda_graph:
        f(*inputs)

    return f(*inputs).numpy()


def get_result_with_relax_cublas_offload(mod, np_inputs, cuda_graph=False):
    mod = partition_for_cublas(mod)
    mod = relax.transform.RunCodegen()(mod)

    return build_and_run(mod, np_inputs, "cuda", cuda_graph)


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
        ((8, 8), (8, 8), False, "none"),
        ((_vars["a"], 6), (6, 16), False, "bias"),
        # Transposed
        ((4, 16), (16, 128), True, "relu"),
        ((35, 8), (8, 8), True, "gelu"),
        # # 3D x 3D
        ((6, 32, 8), (6, 8, 10), False, "bias"),
        ((6, 32, 8), (6, 8, 10), True, "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "gelu"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 10), True, "relu"),
        # ND x 2D
        ((5, 3, 32, 8), (8, 10), False, "none"),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_matmul_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    in_dtype,
    out_dtype,
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_cublas_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue",
    [
        # Regular
        ((8, 8), (8, 8), False, "none"),
        ((_vars["a"], 8), (8, 16), False, "none"),
        # Transposed
        ((4, 16), (16, 128), True, "none"),
        ((35, 16), (16, 128), False, "none"),
        # # 3D x 3D
        ((6, 32, 8), (6, 8, 12), False, "none"),
        ((6, 32, 8), (6, 8, 10), True, "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "none"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 12), False, "none"),
        # ND x 2D
        ((5, 3, 32, 8), (8, 12), False, "none"),
    ],
)
def test_matmul_igemm_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
):
    in_dtype = "int8"
    out_dtype = "int32"
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_cublas_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_cublas_partition_matmul_without_bias():
    # cuBLAS does not handle 2D bias (residual input)
    mod = get_relax_matmul_module((16, 32), (32, 32), "float16", "float16", bias_shape=(16, 32))
    mod = partition_for_cublas(mod)

    # R.add is still in the main function
    assert len(mod["main"].body.blocks[0].bindings) == 2


@pytest.mark.parametrize(
    "M, N, K, was_partitioned", [(16, 8, 32, True), (16, 8, 33, False), (16, 9, 32, False)]
)
def test_cublas_partition_igemm(M, N, K, was_partitioned):
    mod = get_relax_matmul_module((M, K), (K, N), "int8", "int32")
    mod = partition_for_cublas(mod)
    func_name = "fused_relax_matmul_cublas" if was_partitioned else "R.matmul"
    assert func_name in mod["main"].script()


def test_cublas_partition_igemm_with_bias():
    mod = get_relax_matmul_module((16, 32), (32, 8), "int8", "int32", bias_shape=(8,))
    mod = partition_for_cublas(mod)
    func = mod["main"].script()
    assert "fused_relax_matmul_cublas" in func and "R.add" in func


def test_cublas_matmul_cuda_graph():
    @tvm.script.ir.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float16"),
            w0: R.Tensor((16, 16), "float16"),
            w1: R.Tensor((16, 16), "float16"),
            w2: R.Tensor((16, 16), "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(lv0, w1)
                lv2 = R.matmul(lv1, w2)
                R.output(lv2)
            return lv2

    mod = Mod
    shape = [16, 16]
    data = np.random.rand(*shape).astype(np.float16)
    w0 = np.random.rand(*shape).astype(np.float16)
    w1 = np.random.rand(*shape).astype(np.float16)
    w2 = np.random.rand(*shape).astype(np.float16)
    inputs = (data, w0, w1, w2)

    out = get_result_with_relax_cublas_offload(Mod, inputs, cuda_graph=True)

    with tvm.target.Target("cuda"):
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    ref = build_and_run(mod, inputs, "llvm", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
