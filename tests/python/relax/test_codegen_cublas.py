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


def build_and_run(mod, inputs_np, target, legalize=False):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

    dev = tvm.device(target, 0)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def get_result_with_relax_cublas_offload(mod, *args):
    mod = partition_for_cublas(mod)
    mod = relax.transform.RunCodegen()(mod)

    return build_and_run(mod, args, "cuda")


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
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "float32",
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

    out = get_result_with_relax_cublas_offload(mod, *args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
