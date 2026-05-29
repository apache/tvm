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

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.tirx as tirx
from tvm.ir.module import IRModule
from tvm.runtime.executable import Executable
from tvm.script import tirx as T
from tvm.support.nvcc import have_fp16

VECTOR_N_INPUTS = 8


def make_prim_func(
    name: str,
    dtype: str,
    num_inputs: int,
    op: Callable[[tirx.PrimExpr, ...], tirx.PrimExpr],
) -> tirx.PrimFunc:
    """Make a primitive function that applies the given operation to the input buffer."""
    if num_inputs == 1:

        @T.prim_func
        def kernel(
            A: T.Buffer((VECTOR_N_INPUTS,), dtype),
            B: T.Buffer((VECTOR_N_INPUTS,), dtype),
        ):
            T.func_attr({"global_symbol": name + "_kernel", "tirx.noalias": True})
            for i in T.thread_binding(VECTOR_N_INPUTS, thread="threadIdx.x"):
                B[i] = op(A[i])

        return kernel
    elif num_inputs == 2:

        @T.prim_func
        def kernel(
            A: T.Buffer((VECTOR_N_INPUTS,), dtype),
            E: T.Buffer((VECTOR_N_INPUTS,), dtype),
            B: T.Buffer((VECTOR_N_INPUTS,), dtype),
        ):
            T.func_attr({"global_symbol": name + "_kernel", "tirx.noalias": True})
            for i in T.thread_binding(VECTOR_N_INPUTS, thread="threadIdx.x"):
                B[i] = op(A[i], E[i])

        return kernel
    else:
        raise ValueError(f"Unsupported number of inputs: {num_inputs}")


@dataclass(frozen=True)
class MathCase:
    name: str
    op: Callable[[tirx.PrimExpr, ...], tirx.PrimExpr]
    num_inputs: int
    default_intrinsic_f16: str
    default_intrinsic_bf16: str
    default_intrinsic_f32: str
    default_intrinsic_f64: str
    fast_math_intrinsic_f32: str
    np_ref: object
    rtol: float = 1e-5
    atol: float = 1e-6


MATH_CASES = [
    MathCase(
        "exp_case",
        T.exp,
        1,
        "hexp",
        "hexp",
        "expf",
        "exp",
        "__expf",
        lambda x: np.exp(x),
    ),
    MathCase(
        "exp10_case",
        T.exp10,
        1,
        "hexp10",
        "hexp10",
        "exp10f",
        "exp10",
        "__exp10f",
        lambda x: np.power(10.0, x),
    ),
    MathCase(
        "log_case",
        T.log,
        1,
        "hlog",
        "hlog",
        "logf",
        "log",
        "__logf",
        lambda x: np.log(x),
    ),
    MathCase(
        "log2_case",
        T.log2,
        1,
        "hlog2",
        "hlog2",
        "log2f",
        "log2",
        "__log2f",
        lambda x: np.log2(x),
    ),
    MathCase(
        "log10_case",
        T.log10,
        1,
        "hlog10",
        "hlog10",
        "log10f",
        "log10",
        "__log10f",
        lambda x: np.log10(x),
    ),
    MathCase(
        "tan_case",
        T.tan,
        1,
        "htan",
        "htan",
        "tanf",
        "tan",
        "tanf",
        lambda x: np.tan(x),
    ),
    MathCase(
        "cos_case",
        T.cos,
        1,
        "hcos",
        "hcos",
        "cosf",
        "cos",
        "__cosf",
        lambda x: np.cos(x),
    ),
    MathCase(
        "sin_case",
        T.sin,
        1,
        "hsin",
        "hsin",
        "sinf",
        "sin",
        "__sinf",
        lambda x: np.sin(x),
    ),
    MathCase(
        "tanh_case",
        T.tanh,
        1,
        "htanh",
        "htanh",
        "tanhf",
        "tanh",
        "__tanhf",
        lambda x: np.tanh(x),
    ),
    MathCase(
        "pow_case",
        T.pow,
        2,
        "hpow",
        "hpow",
        "powf",
        "pow",
        "__powf",
        lambda x, y: np.power(x, y),
    ),
]


def make_mod(
    dtype: str, case: MathCase, enable_fast_math: bool
) -> tuple[tvm.target.Target, tvm.IRModule]:
    """Make a module for the given dtype and case."""
    target = tvm.target.Target("cuda")
    prim_func = make_prim_func(case.name, dtype, case.num_inputs, case.op)
    return target, tvm.IRModule.from_expr(prim_func.with_attr("target", target))


def expected_intrinsic(dtype: str, case: MathCase, enable_fast_math: bool) -> str:
    """Get the expected intrinsic for the given dtype and case."""
    if dtype == "float16":
        return case.default_intrinsic_f16
    elif dtype == "bfloat16":
        return case.default_intrinsic_bf16
    elif dtype == "float32":
        return case.fast_math_intrinsic_f32 if enable_fast_math else case.default_intrinsic_f32
    elif dtype == "float64":
        return case.default_intrinsic_f64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def check_lowered_ir(
    dtype: str, case: MathCase, enable_fast_math: bool
) -> tuple[tvm.target.Target, IRModule]:
    """Check the lowered IR for the given dtype and case."""
    target, mod = make_mod(dtype, case, enable_fast_math)
    with tvm.transform.PassContext(config={"tirx.enable_fast_math": enable_fast_math}):
        lowered_mod = tvm.tirx.transform.LowerIntrin()(mod)
    script = lowered_mod.script(show_meta=False)
    expected = expected_intrinsic(dtype, case, enable_fast_math)
    assert re.search(rf"""["']{re.escape(expected)}["']""", script)
    return target, lowered_mod


def check_cuda_source(
    target: tvm.target.Target,
    mod: IRModule,
    dtype: str,
    case: MathCase,
    enable_fast_math: bool,
) -> Executable:
    """Check the CUDA source for the given dtype and case."""
    with tvm.transform.PassContext(config={"tirx.enable_fast_math": enable_fast_math}):
        executable = tvm.compile(mod, target=target)
    source = executable.mod.imports[0].inspect_source()
    expected = expected_intrinsic(dtype, case, enable_fast_math)
    assert re.search(rf"(?<!_)\b{re.escape(expected)}\s*\(", source)
    return executable


def make_numpy_inputs(dtype: str, case: MathCase):
    """Make the numpy inputs for the given dtype and case."""
    lhs = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 10.0], dtype=dtype)
    if case.num_inputs == 1:
        return [lhs]
    elif case.num_inputs == 2:
        rhs = np.array([2.0, 3.0, 0.5, 1.5, 0.25, 0.5, 2.0, 1.0], dtype=dtype)
        return [lhs, rhs]
    else:
        raise ValueError(f"Unsupported number of inputs: {case.num_inputs}")


def check_runtime(dtype: str, case: MathCase, executable: Executable):
    """Check the runtime for the given dtype and case."""
    dev = tvm.cuda(0)

    np_inputs = make_numpy_inputs(dtype, case)
    expected = case.np_ref(*[arr.astype(dtype) for arr in np_inputs]).astype(dtype)

    tvm_inputs = [tvm.runtime.tensor(arr, device=dev) for arr in np_inputs]
    output = tvm.runtime.empty((VECTOR_N_INPUTS,), dtype, dev)

    executable(*tvm_inputs, output)
    dev.sync()

    actual = output.numpy()

    np.testing.assert_allclose(actual, expected, rtol=case.rtol, atol=case.atol)


@pytest.mark.parametrize("enable_fast_math", [False, True], ids=["default", "fast_math"])
def test_cuda_math_intrinsic_lowering_pass_context(enable_fast_math):
    check_lowered_ir("float32", MATH_CASES[0], enable_fast_math)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "float64"],
)
@pytest.mark.parametrize("case", MATH_CASES, ids=lambda case: f"{case.name}")
@pytest.mark.parametrize("enable_fast_math", [False, True], ids=["default", "fast_math"])
def test_cuda_math_intrinsic_lowering_source_and_runtime(dtype, case, enable_fast_math):
    if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
        pytest.skip("GPU does not support float16")
    if dtype == "bfloat16" and case.name.startswith("pow_"):
        pytest.skip("pow_argnames=case is only supported for float")

    target, lowered_mod = check_lowered_ir(dtype, case, enable_fast_math)
    executable = check_cuda_source(target, lowered_mod, dtype, case, enable_fast_math)
    check_runtime(dtype, case, executable)
