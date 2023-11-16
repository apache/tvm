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

import ctypes
from typing import Tuple, Callable

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax, rpc, te, tir, topi
from tvm.contrib import utils, cc, popen_pool
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax.testing.vm import check_saved_func
from tvm.runtime import ShapeTuple

EXEC_MODE = ["bytecode", "compiled"]


@pytest.fixture(params=EXEC_MODE)
def exec_mode(request):
    return request.param


def test_vm_compile_simple(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileStage0:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "test.vm.identity", x, y, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return y

    mod = TestVMCompileStage0
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    inp1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm["foo"](inp1, inp2)
    tvm.testing.assert_allclose(inp2.numpy(), inp1.numpy(), rtol=1e-7, atol=1e-7)


def test_match_check(exec_mode):
    @tvm.script.ir_module
    class TestMatchCheck:
        @R.function
        def foo(x: R.Tensor(["n", "m"], "int32"), y: R.Object) -> R.Tensor(["m", "n"], dtype=None):
            return y

    mod = TestMatchCheck
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x0 = tvm.nd.array(np.zeros((1, 2)).astype("int32"))
    y0 = tvm.nd.array(np.zeros((2, 1)).astype("float32"))
    y1 = tvm.nd.array(np.zeros((1, 2)).astype("float32"))
    y2 = tvm.nd.array(np.zeros((2, 1, 1)).astype("float32"))

    vm["foo"](x0, y0)

    with pytest.raises(RuntimeError, match=".*return.*"):
        vm["foo"](x0, y1)

    with pytest.raises(ValueError, match=".*return.*"):
        vm["foo"](x0, y2)


def test_vm_compile_stage2(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileStage2:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Shape:
            n, m = T.int64(), T.int64()
            _ = R.match_cast(x, R.Tensor((n, m), "float32"))
            return R.shape([n * 2, m * 3])

    mod = TestVMCompileStage2
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    arr = tvm.nd.array(np.random.rand(*shape).astype("float32"))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3

    # dtype mismatch
    with pytest.raises(ValueError, match=".*dtype.*"):
        vm["foo"](tvm.nd.array(np.zeros((1, 2)).astype("int32")))

    # ndim mismatch
    with pytest.raises(ValueError, match=".*match_cast.*ndim.*"):
        vm["foo"](tvm.nd.array(np.zeros((1,)).astype("float32")))

    # type mismach
    with pytest.raises(TypeError):
        vm["foo"]([])


def test_vm_compile_stage3(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileStage3:
        @R.function
        def foo(x: R.Tensor((32, 16), "float32")) -> R.Tensor:
            with R.dataflow():
                y = R.call_dps_packed("test.vm.identity", (x), R.Tensor((32, 16), dtype="float32"))
                R.output(y)
            return y

    mod = TestVMCompileStage3
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_e2e(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileE2E:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Tensor:
            with R.dataflow():
                n, m = T.int64(), T.int64()
                _ = R.match_cast(x, R.Tensor((n, m), "float32"))
                y = R.call_dps_packed("test.vm.tile", (x), R.Tensor((n, m * 2), dtype="float32"))
                R.output(y)
            return y

    mod = TestVMCompileE2E

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    shape = (32, 16)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "foo", inp)
    tvm.testing.assert_allclose(res.numpy(), np.tile(inp.numpy(), (1, 2)), rtol=1e-7, atol=1e-7)


def test_vm_compile_e2e_func_param_with_shape(exec_mode):
    @tvm.script.ir_module
    class TestVMCompileE2E2:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.int32()
            n = T.int32()
            k = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def func(
            x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")
        ) -> R.Tensor:
            m, k = T.int64(), T.int64()
            cls = TestVMCompileE2E2
            gv0 = R.call_tir(cls.tir_matmul, (x, w), R.Tensor((m, k), dtype="float32"))
            return gv0

    mod = TestVMCompileE2E2

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    data = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    res = check_saved_func(vm, "func", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_call_tir_inplace_e2e_simple(exec_mode):
    @tvm.script.ir_module
    class TestCallTIRInplaceE2ESimple:
        @T.prim_func
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            C: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
        ):
            # copies the contents of C into A, B, and out1
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[ax0, ax1])
                    T.writes(A[ax0, ax1], B[ax0, ax1], out1[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    B[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")
        ):
            res = R.call_tir_inplace(
                TestCallTIRInplaceE2ESimple.copy,
                (x, y, z),
                [0, 1, -1],
                [R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")],
            )
            return res

    mod = TestCallTIRInplaceE2ESimple

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    x = tvm.nd.array(np.zeros((2, 3)).astype(np.int32))
    y = tvm.nd.array(np.zeros((2, 3)).astype(np.int32))
    z = tvm.nd.array(np.ones((2, 3)).astype(np.int32))
    vm.set_input("main", x, y, z)
    vm.invoke_stateful("main")
    outs = vm.get_outputs("main")
    # check the expected aliasing (the last result is newly allocated)
    assert x == outs[0]
    assert y == outs[1]
    assert x != y
    assert x != outs[2]
    assert y != outs[2]
    tvm.testing.assert_allclose(x.numpy(), z.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(y.numpy(), z.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(outs[2].numpy(), z.numpy(), rtol=1e-7, atol=1e-7)


def test_call_tir_inplace_e2e_rw(exec_mode):
    # read and write from the same tensor
    @tvm.script.ir_module
    class TestCallTIRInplaceE2ERW:
        @T.prim_func
        def inplace_add(A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32")):
            # sums A and B, storing the result in A
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[ax0, ax1], B[ax0, ax1])
                    T.writes(A[ax0, ax1])
                    A[ax0, ax1] = A[ax0, ax1] + B[ax0, ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")
        ) -> R.Tensor((2, 3), "int32"):
            res = R.call_tir_inplace(
                TestCallTIRInplaceE2ERW.inplace_add, (x, y), [0], R.Tensor((2, 3), "int32")
            )
            return res

    mod = TestCallTIRInplaceE2ERW

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    x = tvm.nd.array(np.ones((2, 3)).astype(np.int32))
    y = tvm.nd.array(np.ones((2, 3)).astype(np.int32))
    vm.set_input("main", x, y)
    vm.invoke_stateful("main")
    out = vm.get_outputs("main")
    expected = tvm.nd.array(np.full((2, 3), 2).astype(np.int32))

    assert x == out
    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_emit_te_extern(exec_mode):
    if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
        print("skip because extern function is not available")
        return
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n, m], "float32"))
    y = relax.Var("y", R.Tensor([m, n], "float32"))

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    data = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    res = check_saved_func(vm, "rx_cblas_matmul", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_vm_emit_te_concat(exec_mode):
    # concatenate of two vectors of size (n,) and (m,)
    bb = relax.BlockBuilder()
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([m], "float32"))

    def te_func(A, B):
        C = te.compute((n + m), lambda i: tvm.tir.if_then_else(i < n, A[i], B[i - n]))
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    inp2 = tvm.nd.array(
        np.random.rand(
            2,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp, inp2)
    tvm.testing.assert_allclose(
        res.numpy(), np.append(inp.numpy(), inp2.numpy()), rtol=1e-7, atol=1e-7
    )


def test_vm_emit_te_dtype_change(exec_mode):
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    # convert a tensor with dtype of float32 to int16
    def te_func(A):
        B = te.compute((n,), lambda i: A[i].astype("int16"))
        return B

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        bb.emit_func_output(y)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    inp = tvm.nd.array(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp)
    np.testing.assert_allclose(res.numpy(), inp.numpy().astype("int16"))


def test_vm_emit_te_floor_symbolic_shape(exec_mode):
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    def te_func(A):
        C = te.compute((tir.floordiv(n, 2),), lambda i: A[i] + 1)
        return C

    with bb.function("rx_func", [x]):
        x1 = bb.emit_te(te_func, x)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape = (9,)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp)

    def expected_output():
        output_shape = (shape[0] // 2,)
        return inp.numpy()[: output_shape[0]] + 1

    tvm.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_emit_te_constant_param_cpu(exec_mode):
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    exec = relax.build(mod, "llvm", exec_mode=exec_mode)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)

    add_res = check_saved_func(vm, "main", tvm.nd.array(x_np, dev))
    tvm.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)


@tvm.testing.requires_gpu
def test_vm_emit_te_constant_param_gpu(exec_mode):
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    sch = tvm.tir.Schedule(mod, debug_mask="all")
    loops = sch.get_loops(sch.get_block(name="T_add", func_name="add"))
    sch.bind(loops[0], "threadIdx.x")

    exec = relax.build(sch.mod, "cuda", exec_mode=exec_mode)
    dev = tvm.cuda()
    vm = relax.VirtualMachine(exec, dev)

    add_res = check_saved_func(vm, "main", tvm.nd.array(x_np, dev))
    tvm.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)


def test_vm_relax_symbolic_shape(exec_mode):
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([(n // 2) + 1], "float32"))

    def te_func(A, B):
        C = te.compute((n,), lambda i: A[i] + B[i // 2])
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape1 = (5,)
    shape2 = (3,)
    inp = tvm.nd.array(np.random.rand(*shape1).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape2).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp, inp2)

    def expected_output():
        return inp.numpy() + np.repeat(inp2.numpy(), 2)[:5]

    tvm.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_relax_symbolic_shape_tuple(exec_mode):
    @I.ir_module
    class mod:
        @R.function
        def main(shape: R.Shape(["m", "n"])):
            m = T.int64()
            n = T.int64()
            return R.shape([2 * m, 3 * n])

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    func = vm["main"]

    assert func(ShapeTuple([2, 3])) == [4, 9]

    with pytest.raises(ValueError):
        func(ShapeTuple([2, 3, 4]))

    with pytest.raises(TypeError):
        func(R.prim_value(2))


def test_vm_relax_symbolic_prim_value(exec_mode):
    @I.ir_module
    class mod:
        @R.function
        def main(shape: R.Prim(value="n")):
            n = T.int64()
            return R.prim_value(n * n)

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    func = vm["main"]

    assert func(2) == 4

    with pytest.raises(tvm.TVMError):
        func(ShapeTuple([2]))


def test_vm_relax_multiple_symbolic_prim_value(exec_mode):
    """Like test_vm_relax_symbolic_prim_value, but with multiple variables"""

    @I.ir_module
    class mod:
        @R.function
        def main(
            # Provides definition of "n"
            _n: R.Prim(value="n"),
            # Requires definitions of both "n" and "m", but cannot
            # provide either.
            _shape: R.Shape(["n*2", "m*2"]),
            # Provides definition of "m"
            _m: R.Prim(value="m"),
        ):
            n = T.int64()
            m = T.int64()
            return R.shape([n * n, m + 1])

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    func = vm["main"]

    assert func(2, ShapeTuple([4, 12]), 6) == [4, 7]

    with pytest.raises(RuntimeError):
        func(2, ShapeTuple([4, 12]), 1)

    with pytest.raises(tvm.TVMError):
        func(ShapeTuple([2]))


@pytest.mark.xfail(reason="Current support for R.Prim with known value is primarily for int64")
@pytest.mark.parametrize("exec_mode", EXEC_MODE)
def test_vm_relax_prim_value_fp32(exec_mode):
    """A PrimValue may be R.prim('float32')

    Unlike shape tuples, which must contain int64, a PrimValue may be
    any type that can be represented as a single primitive value.
    """

    @I.ir_module
    class mod:
        @R.function
        def main(
            # First failure occurs during parsing.  The syntactic
            # sugar for symbolic variables assumes that all symbolic
            # variables are int64, rather than using the type that is
            # later declared.
            _x: R.Prim(value="half_fill_value"),
        ):
            half_fill_value = T.float32()
            # Second failure occurs when calling `relax.op.full`.  The
            # `fill_value` is expected to be a scalar constant
            # (R.Tensor with 0-dim shape), not a primitive value, even
            # though these are semantically the same.
            return R.full(shape=[16, 16], fill_value=R.prim_value(2 * half_fill_value))

    target = tvm.target.Target("llvm", host="llvm")
    # Third failure occurs here.  The current codegen assumes that all
    # symbolic variables are int64.
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    func = vm["main"]

    res = func(16.0).numpy()
    assert np.all(res == 32.0)


def test_vm_relax_dyn_tir_shape(exec_mode):
    # case where TIR variables are unbound in generated PrimFunc
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    def te_func(A):
        C = te.compute((n + 1), lambda i: A[i])
        return C

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n + 1,), dtype="float32", name="y")

        x1 = bb.emit_te(te_func, y)
        bb.emit_func_output(x1, params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    ex.export_library("exec.so")
    vm = relax.VirtualMachine(tvm.runtime.load_module("exec.so"), tvm.cpu())
    inp = tvm.nd.array(np.random.rand(2).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(3).astype(np.float32))

    res = check_saved_func(vm, "rx_func", inp, inp2)

    tvm.testing.assert_allclose(res.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuple(exec_mode):
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n,), dtype="float32", name="y")
        tup = relax.Tuple([x, y])
        item = tup[0]
        bb.emit_func_output([tup, item], params=[x, y])

    mod = bb.get()

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    shape = (5,)
    inp = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    inp2 = tvm.nd.array(np.random.rand(*shape).astype(np.float32))
    (res1, res2), res3 = vm["rx_func"](inp, inp2)

    tvm.testing.assert_allclose(res1.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res2.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res3.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuplegetitem(exec_mode):
    @tvm.script.ir_module
    class TestVMTupleGetItem:
        @R.function
        def tuple_get_item(
            x: R.Tensor(ndim=2, dtype="float32"),
            y: R.Tensor(ndim=2, dtype="float32"),
        ):
            t = (x, y)
            a = t[0]
            b = t[1]
            c = R.call_pure_packed(
                "test.vm.add", a, b, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return c

    mod = TestVMTupleGetItem
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    y_inp = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    res = check_saved_func(vm, "tuple_get_item", x_inp, y_inp)
    tvm.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy(), rtol=1e-7, atol=1e-7)


def test_lower_memory_alloc_storage_tensor(exec_mode):
    @tvm.script.ir_module
    class TestMemoryAllocStorageTensor:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = TestMemoryAllocStorageTensor
            storage = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            y = R.memory.alloc_tensor(storage, 0, R.shape([2, 3]), dtype="float32")
            # this is an impure operation, but the overall function is pure so we force purity
            _ = cls.copy(x, y)
            return y

        @T.prim_func
        def copy(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            for i0, i1 in T.grid(2, 3):
                with T.block("block"):
                    vi0, vi1 = T.axis.remap("SS", [i0, i1])
                    B[vi0, vi1] = A[vi0, vi1]

    mod = TestMemoryAllocStorageTensor
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    y = vm["main"](x)
    tvm.testing.assert_allclose(y.numpy(), x.numpy(), rtol=1e-7, atol=1e-7)


def test_sub_func_call(exec_mode):
    @tvm.script.ir_module
    class TestVMSubFunction:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.int32()
            n = T.int32()
            k = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def relax_matmul_tir(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Tensor((32, 32), dtype="float32"):
            cls = TestVMSubFunction
            with R.dataflow():
                gv0 = R.call_tir(cls.tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
                R.output(gv0)
            return gv0

        @R.function
        def relax_matmul_packed(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Object:
            gv0 = R.call_pure_packed(
                "test.vm.mul", x, w, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return gv0

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Object:
            cls = TestVMSubFunction
            gv0 = cls.relax_matmul_tir(x, w)
            gv1 = cls.relax_matmul_packed(gv0, gv0)
            return gv1

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(TestVMSubFunction, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(32, 32).astype(np.float32))
    y_inp = tvm.nd.array(np.random.rand(32, 32).astype(np.float32))
    res = check_saved_func(vm, "main", x_inp, y_inp)
    product = np.dot(x_inp.numpy(), y_inp.numpy())
    expected = product * product
    tvm.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_recursion(exec_mode):
    @tvm.script.ir_module
    class TestVMRecursion:
        @R.function
        def recursion(n: R.Tensor((1,), "float32")) -> R.Tensor:
            cond = R.call_pure_packed(
                "test.vm.equal_zero", n, sinfo_args=(R.Tensor(ndim=1, dtype="float32"))
            )
            if cond:
                res = R.const(1.0)
            else:
                gv0 = R.call_pure_packed(
                    "test.vm.subtract_one", n, sinfo_args=(R.Tensor(ndim=1, dtype="float32"))
                )
                tmp = TestVMRecursion.recursion(gv0)
                res = R.call_pure_packed(
                    "test.vm.add", tmp, tmp, sinfo_args=(R.Tensor(ndim=1, dtype="float32"))
                )
            return res

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(TestVMRecursion, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    inp = np.empty(1).astype("float32")
    recursion_runs = np.random.randint(1, 10)
    inp.fill(recursion_runs)
    inp = tvm.nd.array(inp)
    res = check_saved_func(vm, "recursion", inp)
    tvm.testing.assert_allclose(res.numpy(), np.power(2.0, recursion_runs), rtol=1e-7, atol=1e-7)


@tvm.testing.requires_gpu
def test_vm_to_device(exec_mode):
    @tvm.script.ir_module
    class TestToVDevice:
        @R.function
        def foo1(
            x: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            copied = R.to_vdevice(x, tvm.ir.VDevice("cuda", 0, "global"))
            return copied

        @R.function
        def foo2(
            x: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            copied = R.to_vdevice(x, tvm.ir.VDevice("llvm", 0, "global"))
            return copied

    mod = TestToVDevice
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    res_1 = check_saved_func(vm, "foo1", x_inp)
    res_2 = check_saved_func(vm, "foo2", x_inp)

    # check the copied tensor's device
    assert str(res_1.device) == "cuda(0)"
    assert str(res_2.device) == "cpu(0)"

    tvm.testing.assert_allclose(res_1.numpy(), x_inp.numpy())
    tvm.testing.assert_allclose(res_2.numpy(), x_inp.numpy())


def test_vm_closure(exec_mode):
    @tvm.script.ir_module
    class TestClosure:
        @R.function
        def lifted_func_1(x: R.Tensor((2, 3), "float32"), env: R.Tensor((2, 3), "float32")):
            return R.call_pure_packed("test.vm.add", x, env, sinfo_args=(R.Tensor()))

        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 3), "float32"),
        ):
            cls = TestClosure
            clo = R.make_closure(cls.lifted_func_1, (x,))
            res = R.invoke_pure_closure(clo, (y,), sinfo_args=(R.Tensor()))
            return res

    mod = TestClosure
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x_inp = tvm.nd.array(np.random.rand(2, 3).astype("float32"))
    y_inp = tvm.nd.array(np.array([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]], dtype="float32"))
    res = check_saved_func(vm, "main", x_inp, y_inp)
    tvm.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy())


def test_time_evaluator(exec_mode):
    @tvm.script.ir_module
    class TestTimeEvaluator:
        @R.function
        def main(x: R.Tensor((1,), "float32"), y: R.Tensor((1,), "float32")):
            return R.call_pure_packed(
                "test.vm.add", x, y, sinfo_args=(R.Tensor(ndim=1, dtype="float32"))
            )

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(TestTimeEvaluator, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    x = tvm.nd.array(np.random.rand(1).astype("float32"))
    y = tvm.nd.array(np.random.rand(1).astype("float32"))

    # ensure we can use time_evaluator with the stateful API
    vm.set_input("main", x, y)
    timing_res = vm.time_evaluator("invoke_stateful", tvm.cpu())("main")
    # just checking that it has some results at all
    assert timing_res.results

    # ensure we can use it with a closure
    vm.save_function("main", "saved_main", x, y)
    timing_res = vm.time_evaluator("saved_main", tvm.cpu())()
    assert timing_res.results


@tvm.script.ir_module
class TestVMSetInput:
    @T.prim_func
    def test_vm_mul(x: T.handle, y: T.handle, z: T.handle):
        T.func_attr({"global_symbol": "test_vm_mul"})
        m = T.int32()
        n = T.int32()
        A = T.match_buffer(x, (m, n))
        B = T.match_buffer(y, (m, n))
        C = T.match_buffer(z, (m, n))

        for i, j in T.grid(m, n):
            with T.block("mul"):
                vi = T.axis.spatial(m, i)
                vj = T.axis.spatial(n, j)
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    # test returning a tuple
    @R.function
    def test_vm_tuple(
        x: R.Tensor((), "int32")
    ) -> R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32")):
        return (x, x)

    # nested tuple too
    @R.function
    def test_vm_nested_tuple(
        x: R.Tensor((), "int32")
    ) -> R.Tuple(
        R.Tuple(
            R.Tensor((), "int32"),
            R.Tuple(
                R.Tensor((), "int32"),
            ),
        ),
        R.Tensor((), "int32"),
    ):
        return ((x, (x,)), x)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        cls = TestVMSetInput
        gv0 = R.call_tir(cls.test_vm_mul, (x, w), R.Tensor((32, 32), dtype="float32"))
        return gv0


def test_multi_systemlib(exec_mode):
    @tvm.script.ir_module
    class ModA:
        I.module_attrs({"system_lib_prefix": "libA_"})

        @T.prim_func
        def tir_init(x: T.Buffer((2), "float32")) -> None:
            for i in range(2):
                x[i] = T.float32(0)

        @R.function
        def main(s: R.Shape(["m"])) -> R.Tensor:
            m = T.int64()
            gv0 = R.call_tir(ModA.tir_init, (), R.Tensor((m + 1,), dtype="float32"))
            return gv0

    @tvm.script.ir_module
    class ModB:
        I.module_attrs({"system_lib_prefix": "libB_"})

        @T.prim_func
        def tir_init(x: T.Buffer((2), "float32")) -> None:
            for i in range(2):
                x[i] = T.float32(1)

        @R.function
        def main(s: R.Shape(["m"])) -> R.Tensor:
            m = T.int64()
            gv0 = R.call_tir(ModB.tir_init, (), R.Tensor((m,), dtype="float32"))
            return gv0

    target = tvm.target.Target("llvm", host="llvm")
    libA = relax.build(ModA, target, exec_mode=exec_mode)
    libB = relax.build(ModB, target, exec_mode=exec_mode)

    temp = utils.tempdir()
    pathA = temp.relpath("libA.a")
    pathB = temp.relpath("libB.a")
    path_dso = temp.relpath("mylibAll.so")
    libA.export_library(pathA, fcompile=cc.create_staticlib)
    libB.export_library(pathB, fcompile=cc.create_staticlib)

    # package two static libs together
    # check that they do not interfere with each other
    # even though they have shared global var names
    # intentionally craft same gvar function with different behaviors
    cc.create_shared(path_dso, ["-Wl,--whole-archive", pathA, pathB, "-Wl,--no-whole-archive"])

    def popen_check():
        # Load dll, will trigger system library registration
        ctypes.CDLL(path_dso)
        # Load the system wide library
        vmA = relax.VirtualMachine(tvm.runtime.system_lib("libA_"), tvm.cpu())
        vmB = relax.VirtualMachine(tvm.runtime.system_lib("libB_"), tvm.cpu())

        retA = vmA["main"](tvm.runtime.ShapeTuple([1]))
        retB = vmB["main"](tvm.runtime.ShapeTuple([2]))
        np.testing.assert_equal(retA.numpy(), np.array([0, 0]).astype("float32"))
        np.testing.assert_equal(retB.numpy(), np.array([1, 1]).astype("float32"))

    # system lib should be loaded in different process
    worker = popen_pool.PopenWorker()
    worker.send(popen_check)


def set_input_trial(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    vm.invoke_stateful("main")
    res0 = vm.get_outputs("main")

    data_dict = {"x": a, "w": b}
    vm.set_input("main", **data_dict)
    vm.invoke_stateful("main")
    res1 = vm.get_outputs("main")
    tvm.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)
    tvm.testing.assert_allclose(res0.numpy(), res1.numpy(), rtol=1e-7, atol=1e-7)

    # bug! If you don't bind the NDArray to a var, the memory will get corrupted.
    # Possibly due to object lifecycles and other FFI issues
    a = tvm.nd.array(np.array(2).astype("int32"), device)
    vm.set_input("test_vm_tuple", a)
    vm.invoke_stateful("test_vm_tuple")
    res2 = vm.get_outputs("test_vm_tuple")
    # the results are NDArrays wrapped around scalars,
    # so we have to get the scalar out of the NDArray
    assert tuple(map(lambda a: int(a.numpy()), res2)) == (2, 2)

    b = tvm.nd.array(np.array(1).astype("int32"), device)
    vm.set_input("test_vm_nested_tuple", b)
    vm.invoke_stateful("test_vm_nested_tuple")
    res3 = vm.get_outputs("test_vm_nested_tuple")
    assert len(res3) == 2 and len(res3[0]) == 2 and len(res3[0][1]) == 1
    result_cast = ((int(res3[0][0].numpy()), (int(res3[0][1][0].numpy()),)), int(res3[1].numpy()))
    assert result_cast == ((1, (1,)), 1)


def set_input_attempt_stateless(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: once you set inputs, you cannot run statelessly
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    # must use invoke stateful!
    vm["main"]()


def set_input_attempt_invoke(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: if the function needs inputs, you can't invoke directly
    vm.invoke_stateful("main")


def set_input_attempt_get(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # this should fail: you can't get outputs without invoking the function first
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    _ = vm.get_outputs("main")


def make_vm(mod, exec_mode, temp) -> Tuple[relax.VirtualMachine, tvm.runtime.Device]:
    """Returns a local VM for the given mod and the device"""
    target = tvm.target.Target("llvm", host="llvm")
    exec = relax.build(mod, target, exec_mode=exec_mode)
    libname = temp.relpath("exec.so")
    exec.export_library(libname)
    exec_loaded = tvm.runtime.load_module(libname)
    device = tvm.cpu()
    return relax.VirtualMachine(exec_loaded, device), device


def run_on_rpc(
    mod: tvm.IRModule,
    trial_func: Callable[[relax.VirtualMachine, tvm.runtime.Device], None],
    exec_mode: str,
):
    """
    Sets up a VM over localhost using the given mod and runs the given trial function.
    The trial function should take a VM and a device
    """
    target = tvm.target.Target("llvm", host="llvm")
    exec = relax.build(mod, target, exec_mode=exec_mode)
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    exec.export_library(path)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    # Adapted from relay/test_vm.py
    def check_remote(server):
        remote = rpc.connect(server.host, server.port, session_timeout=10)

        # Upload the serialized Executable.
        remote.upload(path)
        # Get a handle to remote Executable.
        rexec = remote.load_module("vm_library.so")

        device = remote.cpu()
        # Build a VM out of the executable and context.
        vm = relax.VirtualMachine(rexec, device=device)
        trial_func(vm, device)

    check_remote(rpc.Server("127.0.0.1"))


def test_set_input(exec_mode):
    temp = utils.tempdir()
    set_input_trial(*make_vm(TestVMSetInput, exec_mode, temp))


def test_set_input_tuple(exec_mode):
    @tvm.script.ir_module
    class MyMod:
        @R.function
        def main(x: R.Tuple([R.Tensor((32,), "float32"), R.Tensor((32,), "float32")])) -> R.Tensor:
            y = x[0]
            return y

    temp = utils.tempdir()
    vm, device = make_vm(MyMod, exec_mode, temp)
    device = tvm.cpu(0)
    a = tvm.nd.empty((32,), "float32", device=device)
    b = tvm.nd.empty((32,), "float32", device=device)
    vm.set_input("main", (a, b))
    vm.invoke_stateful("main")


def save_function_kwargs_trial(vm: relax.VirtualMachine, device: tvm.runtime.Device) -> None:
    # just checking that we can use kwargs for the args when saving a function
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", x=a, w=b)
    res0 = vm["saved_main"]()
    tvm.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)


def test_save_function_kwargs(exec_mode):
    temp = utils.tempdir()
    save_function_kwargs_trial(*make_vm(TestVMSetInput, exec_mode, temp))


def test_save_function_kwargs_rpc(exec_mode):
    run_on_rpc(TestVMSetInput, save_function_kwargs_trial, exec_mode)


def save_function_time_evaluator_trial(
    vm: relax.VirtualMachine, device: tvm.runtime.Device
) -> None:
    # just checking that the saved function can be called in the time evaluator
    a = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    b = tvm.nd.array(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", a, b)
    vm.time_evaluator("saved_main", device)()


def test_save_function_time_evaluator(exec_mode):
    temp = utils.tempdir()
    save_function_time_evaluator_trial(*make_vm(TestVMSetInput, exec_mode, temp))


def test_save_function_time_evaluator_rpc(exec_mode):
    run_on_rpc(TestVMSetInput, save_function_time_evaluator_trial, exec_mode)


# if you set an input, you should not be able to call statelessly


def test_set_input_stateless_failure(exec_mode):
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, exec_mode, temp)
    with pytest.raises(RuntimeError):
        set_input_attempt_stateless(*args)


def test_set_input_stateless_failure_rpc(exec_mode):
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_stateless, exec_mode)


def test_set_input_invoke_failure(exec_mode):
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, exec_mode, temp)
    with pytest.raises(ValueError):
        set_input_attempt_invoke(*args)


def test_set_input_invoke_failure_rpc(exec_mode):
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_invoke, exec_mode)


def test_set_input_get_failure(exec_mode):
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, exec_mode, temp)
    with pytest.raises(ValueError):
        set_input_attempt_get(*args)


def test_set_input_get_failure_rpc(exec_mode):
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_get, exec_mode)


if __name__ == "__main__":
    tvm.testing.main()
