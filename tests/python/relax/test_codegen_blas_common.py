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
"""Shared test utilities for cuBLAS and hipBLAS codegen tests."""
import numpy as np

import tvm
from tvm import relax
from tvm.relax.testing import get_relax_matmul_module


def build_and_run(mod, inputs_np, target, legalize=False, cuda_graph=False):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(
        config={
            "relax.backend.use_cuda_graph": cuda_graph,
            "relax.transform.apply_legalize_ops": legalize,
        }
    ):
        ex = tvm.compile(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs_np]

    # For cuda graph, run the compiled function twice to make sure that we can launch the cached
    # graph on the second run.
    if cuda_graph:
        f(*inputs)

    return f(*inputs).numpy()


def to_concrete_shape(symbolic_shape, var_table):
    result = []
    for dim in symbolic_shape:
        if not isinstance(dim, tvm.tir.expr.Var):
            result.append(dim)
            continue

        if dim not in var_table:
            var_table[dim] = np.random.randint(10, 50)
        result.append(var_table[dim])

    return tuple(result)


def run_matmul_offload_test(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    in_dtype,
    out_dtype,
    epilogue_table,
    partition_fn,
    target,
):
    """Shared test logic for matmul offload tests across different BLAS backends.

    Parameters
    ----------
    x_shape : tuple
        Shape of the first input tensor.
    y_shape : tuple
        Shape of the second input tensor.
    transpose_y : bool
        Whether to transpose the second input.
    epilogue : str
        Type of epilogue operation.
    in_dtype : str
        Input data type.
    out_dtype : str
        Output data type.
    epilogue_table : dict
        Mapping of epilogue names to (with_bias, activation) tuples.
    partition_fn : callable
        Function to partition the module for the specific BLAS backend.
    target : str
        Target device (e.g., "cuda" or "rocm").
    """
    with_bias, activation = epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = to_concrete_shape(x_shape, var_table)
    concrete_y_shape = to_concrete_shape(y_shape, var_table)
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

    mod = partition_fn(mod)
    mod = relax.transform.RunCodegen()(mod)
    out = build_and_run(mod, args, target)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)
