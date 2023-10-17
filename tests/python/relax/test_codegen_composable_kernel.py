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
from tvm.relax.backend.contrib.composable_kernel import partition_for_composable_kernel
from tvm.script import relax as R
from tvm.relax.testing import get_relax_matmul_module


has_composable_kernel = tvm.get_global_func("relax.ext.composable_kernel", True)

composable_kernel_enabled = pytest.mark.skipif(
    not has_composable_kernel,
    reason="ComposableKernel not enabled.",
)

pytestmark = [composable_kernel_enabled]


def build_and_run(mod, inputs_np, target, legalize=True):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)  # For cpu reference, nop for composable kernel.

    with tvm.transform.PassContext():
        ex = relax.build(mod, target)

    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.nd.array(inp, dev) for inp in inputs_np]

    return f(*inputs).numpy()


def get_result_with_relax_composable_kernel_offload(
    mod, *args, assert_all_bindings_fused=True, num_final_bindings=1
):
    mod = partition_for_composable_kernel(mod)

    if assert_all_bindings_fused:
        assert len(mod["main"].body.blocks[0].bindings) == num_final_bindings

    codegen_pass = relax.transform.RunCodegen()
    mod = codegen_pass(mod)

    return build_and_run(mod, args, "rocm")


def _to_concrete_shape(symbolic_shape, var_table=None):
    if var_table is None:
        var_table = {}

    result = []
    for dim in symbolic_shape:
        if isinstance(dim, tuple):
            result.append(_to_concrete_shape(dim, var_table))
            continue

        if not isinstance(dim, tvm.tir.expr.Var):
            result.append(dim)
            continue

        if dim not in var_table:
            var_table[dim] = np.random.randint(10, 50)
        result.append(var_table[dim])

    return tuple(result)


_epilogue_table = {
    "none": (False, None),
    "bias": (True, None),
    "relu": (True, R.nn.relu),
    "gelu": (True, R.nn.gelu),
    "silu": (True, R.nn.silu),
}


_residual_block_table = {
    "none": (None, None),
    "add_relu": (R.add, R.nn.relu),
    "mul_relu": (R.multiply, R.nn.relu),
    "add": (R.add, None),
    "mul": (R.multiply, None),
}


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue, residual_block",
    [
        # Regular
        # ((64, 128), (128, 256), False, "none", "none"),
        # Transposed
        ((64, 128), (128, 256), False, "none", "none"),
        # 3D x 3D
        # ((6, 32, 64), (6, 64, 128), True, "none", "none"),
        # 3D x 2D
        # TODO(@tiandi): test 3D x 2D
        # ((6, 32, 64), (64, 128), True, "none", "none"),
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
    residual_block,
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

    residual_bin_op, residual_activation = _residual_block_table[residual_block]

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
        residual_bin_op=residual_bin_op,
        residual_activation=residual_activation,
    )
    out = get_result_with_relax_composable_kernel_offload(mod, *args)
    ref = build_and_run(mod, args, "llvm")

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
