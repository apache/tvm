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

import pytest
import tvm
from tvm import relax
from tvm.ir import structural_equal
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


def test_to_non_dataflow():
    @tvm.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.int64(), T.int64()
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "test.op.identity",
                    (x,),
                    R.Tensor(
                        (m, n),
                        dtype="float32",
                    ),
                )
                gv0 = R.call_dps_packed(
                    "test.op.identity",
                    (lv0,),
                    R.Tensor(
                        (m, n),
                        dtype="float32",
                    ),
                )
                R.output(gv0)
            return gv0

    mod = TestToNonDataflow

    old_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    x, lv0, gv0 = old_vars

    new_mod = relax.transform.ToNonDataflow()(mod)

    new_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal new_vars
            new_vars.append(e)

    relax.analysis.post_order_visit(new_mod["foo"], fvisit)

    assert x == new_vars[0]
    assert lv0 != new_vars[1]
    assert isinstance(lv0, relax.DataflowVar)
    assert not isinstance(new_vars[1], relax.DataflowVar)

    assert isinstance(gv0, relax.Var)
    assert isinstance(new_vars[2], relax.Var)
    assert gv0 == new_vars[2]


def test_call_tir_rewrite():
    @tvm.script.ir_module
    class TestCallTIRRewrite:
        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()
            gv0 = R.call_tir(TestCallTIRRewrite.exp, (x,), R.Tensor((m, n), dtype="float32"))
            return gv0

    mod = TestCallTIRRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, relax.Call)
    assert s0.op.name == "relax.call_tir"

    # after rewrite
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.sinfo_args[0].shape)
    s2 = block.bindings[1].value
    tvm.ir.expr.GlobalVar
    assert s2.op.name_hint == "exp"


def test_transform_remove_purity_checking():
    @tvm.script.ir_module
    class Before:
        @R.function
        def base(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.add(x, y)
            return z

        @R.function
        def use_call_pure_packed(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.call_pure_packed("vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            closure = R.make_closure(Before.base, ())
            res = R.invoke_pure_closure(closure, (x,), sinfo_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Object:
            y = R.print(format="I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                y = R.add(x, x)
                q = R.call_pure_packed(
                    "vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32"))
                )
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Object:
                x = R.print(format="Oops!")
                return x

            y = R.const(1, dtype="int32")
            z = nested()
            return y

    @tvm.script.ir_module
    class Expected:
        @R.function
        def base(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            y = R.add(x, x)
            z = R.add(x, y)
            return z

        @R.function
        def use_call_pure_packed(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            y = R.add(x, x)
            z = R.call_packed("vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            closure = R.make_closure(Expected.base, ())
            res = R.invoke_closure(closure, (x,), sinfo_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Object:
            y = R.print(format="I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})

            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                R.func_attr({"relax.force_pure": True})
                y = R.add(x, x)
                q = R.call_packed("vm.builtin.copy", y, sinfo_args=(R.Tensor((), dtype="int32")))
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Object:
                x = R.print(format="Oops!")
                return x

            y = R.const(1, dtype="int32")
            z = nested()
            return y

    new_mod = relax.transform.RemovePurityChecking()(Before)
    tvm.ir.assert_structural_equal(new_mod, Expected)


def test_call_dps_packed_rewrite():
    @tvm.script.ir_module
    class TestCallDPSPackedRewrite:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()
            gv0 = R.call_dps_packed("test.op.identity", (x,), R.Tensor((m, n), dtype="float32"))
            return gv0

    mod = TestCallDPSPackedRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, relax.Call)
    assert s0.op.name == "relax.call_dps_packed"

    # CallTIRRewrite also works for call_dps_packed
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.sinfo_args[0].shape)
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_call_tir_inplace_simple():
    # simple case: one inplace argument
    @tvm.script.ir_module
    class Input:
        @T.prim_func
        def zeros(A: T.Buffer((2, 3), "int32")):
            # just overwrites A with 0s
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.writes(A[ax0, ax1])
                    A[ax0, ax1] = T.int32(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(Input.zeros, x, 0, R.Tensor((2, 3), dtype="int32"))
            return gv0

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def zeros(A: T.Buffer((2, 3), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.writes(A[ax0, ax1])
                    A[ax0, ax1] = T.int32(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            R.func_attr({"relax.force_pure": True})
            _ = Expected.zeros(x)
            gv0 = x
            return gv0

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_multiple_args():
    @tvm.script.ir_module
    class Input:
        @T.prim_func
        def copy(
            A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32"), C: T.Buffer((2, 3), "int32")
        ):
            # copies the contents of C into A and B
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[ax0, ax1])
                    T.writes(A[ax0, ax1], B[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    B[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.copy,
                (x, y, z),
                [0, 1],
                [R.Tensor((2, 3), dtype="int32"), R.Tensor((2, 3), dtype="int32")],
            )
            return gv0

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def copy(A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(B[ax0, ax1])
                    T.writes(A[ax0, ax1])
                    A[ax0, ax1] = B[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            _ = Expected.copy(x, y, z)
            gv0 = (x, y)
            return gv0

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_some_new():
    @tvm.script.ir_module
    class Input:
        @T.prim_func
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            C: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
            out2: T.Buffer((2, 3), "int32"),
        ):
            # copies the contents of C into A, out1, and out2
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[ax0, ax1])
                    T.writes(A[ax0, ax1], out1[ax0, ax1], out2[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]
                    out2[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), dtype="int32")
        ):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.copy,
                (x, y, z),
                [0, -1, -1],
                [
                    R.Tensor((2, 3), dtype="int32"),
                    R.Tensor((2, 3), dtype="int32"),
                    R.Tensor((2, 3), dtype="int32"),
                ],
            )
            return gv0

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            C: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
            out2: T.Buffer((2, 3), "int32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[ax0, ax1])
                    T.writes(A[ax0, ax1], out1[ax0, ax1], out2[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]
                    out2[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), dtype="int32")
        ):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.builtin.alloc_tensor(R.shape([2, 3]), "int32", R.prim_value(0))
            gv1 = R.builtin.alloc_tensor(R.shape([2, 3]), "int32", R.prim_value(0))
            _ = Expected.copy(x, y, z, gv0, gv1)
            gv2 = (x, gv0, gv1)
            return gv2

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


@pytest.mark.xfail()
def test_call_tir_inplace_repeated_input():
    @tvm.script.ir_module
    class Input:
        @T.prim_func
        def func(
            A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32"), C: T.Buffer((2, 3), "int32")
        ):
            T.evaluate(0)

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.func,
                (x, y, z),
                # repeated 0 -> that's an error
                [0, 0],
                [R.Tensor((2, 3), dtype="int32"), R.Tensor((2, 3), dtype="int32")],
            )
            return gv0


@pytest.mark.xfail()
def test_call_tir_inplace_all_new():
    @tvm.script.ir_module
    class Input:
        @T.prim_func
        def func(A: T.Buffer((2, 3), "int32")):
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            R.func_attr({"relax.force_pure": True})
            # cannot make the only output a fresh one
            gv0 = R.call_tir_inplace(Input.func, x, -1, R.Tensor((2, 3), dtype="int32"))
            return gv0


def test_vm_builtin_lower():
    @tvm.script.ir_module
    class TestVMBuiltinLower:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            # we expected RemovePurityChecking to have been called first
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()
            alloc = R.builtin.alloc_tensor(R.shape([m, n]), runtime_device_index=0, dtype="float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    mod = TestVMBuiltinLower

    # after vm builtin lowering
    new_mod = relax.transform.VMBuiltinLower()(mod)
    func = new_mod["foo"]

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(func, tvm.relax.expr.Function)

    block = func.body.blocks[0]
    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.vm.alloc_storage"
    s2 = block.bindings[1].value
    assert isinstance(s2, relax.Call)
    s3 = block.bindings[2].value
    assert isinstance(s3, relax.Call)
    assert isinstance(s3.op, relax.ExternFunc)
    assert s3.op.global_symbol == "test.op.identity"


if __name__ == "__main__":
    pytest.main([__file__])
