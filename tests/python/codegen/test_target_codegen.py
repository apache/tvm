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
# ruff: noqa: F841

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T


def test_buffer_store_predicate_not_supported():
    target = "c"

    @T.prim_func(s_tir=True)
    def func(b: T.handle):
        B = T.match_buffer(b, (8,), "float32")
        B.vstore([T.Ramp(0, 2, 4)], T.Broadcast(1.0, 4), predicate=T.Broadcast(T.bool(True), 4))

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm.target.Target(target):
            tvm.compile(func)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "from_device": 0}, marks=pytest.mark.gpu),
    ],
)
def test_buffer_store_predicate_not_supported_gpu(target):
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.prim_func(s_tir=True)
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (2, 3), "float32")
        B = T.match_buffer(b, (6,), "float32")
        T.func_attr({"global_symbol": "main"})
        for i_0 in T.thread_binding(3, thread="threadIdx.x"):
            B.vstore(
                [T.Ramp(i_0, 1, 4)], T.Broadcast(1.0, 4), predicate=T.Broadcast(T.bool(True), 4)
            )

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm.target.Target(target):
            tvm.compile(func)


def test_buffer_load_predicate_not_supported():
    target = "c"

    @T.prim_func(s_tir=True)
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (8,), "float32")
        B = T.match_buffer(b, (8,), "float32")
        for i_0 in range(4):
            B.vstore(
                [T.Ramp(0, 2, 4)],
                A.vload([T.Ramp(i_0, 1, 4)], predicate=T.Broadcast(T.bool(True), 4)),
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm.target.Target(target):
            tvm.compile(func)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "from_device": 0}, marks=pytest.mark.gpu),
    ],
)
def test_buffer_load_predicate_not_supported_gpu(target):
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.prim_func(s_tir=True)
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (8,), "float32")
        B = T.match_buffer(b, (8,), "float32")
        for i_0 in T.thread_binding(3, thread="threadIdx.x"):
            B.vstore(
                [T.Ramp(0, 2, 4)],
                A.vload([T.Ramp(i_0, 1, 4)], predicate=T.Broadcast(T.bool(True), 4)),
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm.target.Target(target):
            tvm.compile(func)


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_codegen_loop_step(target):
    if target != "c" and not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.prim_func(s_tir=True)
    def test_loop_step(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        for i in T.serial(3, 1024, step=96):
            C[i] = A[i] + B[i]

    with tvm.transform.PassContext(disabled_pass=["s_tir.CanonicalizeLoop"]):
        lib = tvm.compile(test_loop_step, target=target)

    src = lib.mod.inspect_source()
    if target == "c":
        assert src.find("for (int32_t i = 3; i < 1024; i += 96)") >= 0

    dev = tvm.cpu()
    a_np = np.random.rand(1024).astype("float32")
    b_np = np.random.rand(1024).astype("float32")
    c_np = np.zeros(1024, dtype="float32")
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(c_np, dev)

    lib(a_tvm, b_tvm, c_tvm)

    c_result = c_tvm.numpy()

    # Check that the loop executes at positions 3, 99, 195, 291, 387, 483, 579, 675, 771, 867, 963
    for i in range(3, 1024, 96):
        tvm.testing.assert_allclose(c_result[i], a_np[i] + b_np[i], rtol=1e-5)

    # Assert non-touched positions remain zero
    for i in range(0, 3):
        assert c_result[i] == 0.0
    for i in range(4, 1024):
        if (i - 3) % 96 != 0:
            assert c_result[i] == 0.0


def test_min_max_nan_preserving():
    dtype = "float32"
    uint_dtype = "uint32"

    @T.prim_func(s_tir=True)
    def max_func(
        A: T.Buffer((8,), dtype),
        B: T.Buffer((8,), dtype),
        C: T.Buffer((8,), dtype),
    ):
        T.func_attr({"tirx.noalias": True})
        for i in range(8):
            C[i] = T.max(A[i], B[i])

    @T.prim_func(s_tir=True)
    def min_func(
        A: T.Buffer((8,), dtype),
        B: T.Buffer((8,), dtype),
        C: T.Buffer((8,), dtype),
    ):
        T.func_attr({"tirx.noalias": True})
        for i in range(8):
            C[i] = T.min(A[i], B[i])

    a_np = np.array([0.0, 1.0, 0.0, 0.0, -0.0, 3.0, 2.0, -5.0], dtype=dtype)
    b_np = np.array([1.0, 0.0, 0.0, -0.0, 0.0, 2.0, 2.0, -4.0], dtype=dtype)
    a_bits = a_np.view(uint_dtype)
    b_bits = b_np.view(uint_dtype)
    a_bits[[0, 2]] = 0x7FC00011
    b_bits[[1, 2]] = 0x7FC00022

    dev = tvm.cpu()
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    targets = ["c"]
    if tvm.testing.device_enabled("llvm"):
        targets.append("llvm")

    for target in targets:
        for operation, func in [("min", min_func), ("max", max_func)]:
            c = tvm.runtime.empty((8,), dtype, dev)
            tvm.compile(func, target=target)(a, b, c)
            compare = a_np < b_np if operation == "min" else a_np > b_np
            expected = np.where(compare | np.isnan(a_np), a_np, b_np)
            np.testing.assert_array_equal(c.numpy().view(uint_dtype), expected.view(uint_dtype))


def _make_min_max_func(operation, const_side, dtype="float32", extent=1, const_value=0):
    a_buffer = tvm.tirx.decl_buffer((extent,), dtype, name="A")
    c_buffer = tvm.tirx.decl_buffer((extent,), dtype, name="C")
    index = tvm.tirx.Var("i", "int32")
    constant = tvm.tirx.const(const_value, dtype)
    dynamic = tvm.tirx.BufferLoad(a_buffer, [index])
    lhs, rhs = (constant, dynamic) if const_side == "lhs" else (dynamic, constant)
    result = {"min": tvm.tirx.min, "max": tvm.tirx.max}[operation](lhs, rhs)
    body = tvm.tirx.For(
        index,
        0,
        extent,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.BufferStore(c_buffer, result, [index]),
    )
    return tvm.tirx.PrimFunc([a_buffer, c_buffer], body).with_attr("global_symbol", "main")


@pytest.mark.parametrize(
    "target,operation,const_side",
    [
        ("c", "min", "lhs"),
        ("c", "max", "rhs"),
        ("llvm", "min", "rhs"),
        ("llvm", "max", "lhs"),
    ],
)
def test_min_max_float_imm_operand(target, operation, const_side):
    if target != "c" and not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    func = _make_min_max_func(operation, const_side, extent=7)
    compile_target = {"kind": "llvm", "opt-level": 0} if target == "llvm" else target
    compiled = tvm.compile(func, target=compile_target)

    a_np = np.array([np.nan, -0.0, 0.0, -1.0, 1.0, np.inf, -np.inf], dtype="float32")
    c = tvm.runtime.empty(a_np.shape, "float32", tvm.cpu())
    compiled(tvm.runtime.tensor(a_np), c)

    zero = np.zeros_like(a_np)
    lhs, rhs = (zero, a_np) if const_side == "lhs" else (a_np, zero)
    compare = lhs < rhs if operation == "min" else lhs > rhs
    expected = np.where(compare | np.isnan(lhs), lhs, rhs)
    np.testing.assert_array_equal(c.numpy().view("uint32"), expected.view("uint32"))

    if target == "llvm":
        predicate = {
            ("min", "lhs"): "olt",
            ("min", "rhs"): "ult",
            ("max", "lhs"): "ogt",
            ("max", "rhs"): "ugt",
        }[operation, const_side]
        llvm_ir = compiled.mod.inspect_source("ll")
        assert f"fcmp {predicate}" in llvm_ir
        assert "fcmp uno" not in llvm_ir
    else:
        predicate = {
            ("min", "lhs"): " < ",
            ("min", "rhs"): " >= ",
            ("max", "lhs"): " > ",
            ("max", "rhs"): " <= ",
        }[operation, const_side]
        result_lines = [
            line
            for line in compiled.mod.inspect_source().splitlines()
            if " = " in line and " ? " in line
        ]
        assert len(result_lines) == 1
        assert predicate in result_lines[0]
        assert "||" not in result_lines[0]
        assert "!=" not in result_lines[0]


def test_llvm_min_max_broadcast_float_imm_operand():
    if not tvm.testing.device_enabled("llvm"):
        pytest.skip("llvm not enabled")

    func = _make_min_max_func("min", "rhs", dtype="float32x4")
    llvm_ir = tvm.compile(func, target={"kind": "llvm", "opt-level": 0}).mod.inspect_source("ll")
    assert "fcmp ult <4 x float>" in llvm_ir
    assert "fcmp uno" not in llvm_ir


def test_llvm_min_max_nan_float_imm_operand():
    if not tvm.testing.device_enabled("llvm"):
        pytest.skip("llvm not enabled")

    for operation, const_side in [("min", "lhs"), ("max", "rhs")]:
        func = _make_min_max_func(operation, const_side, const_value=np.nan)
        llvm_ir = tvm.compile(func, target={"kind": "llvm", "opt-level": 0}).mod.inspect_source(
            "ll"
        )
        fcmp_lines = [line for line in llvm_ir.splitlines() if " fcmp " in line]
        if const_side == "lhs":
            assert not fcmp_lines
        else:
            assert len(fcmp_lines) == 1
            assert " fcmp uno " in fcmp_lines[0]


if __name__ == "__main__":
    tvm.testing.main()
