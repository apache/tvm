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

import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def _make_dataflow_block_with_dual_role_var():
    prim_ty = tvm.ir.PrimType("int64")
    source = tvm.ir.Var("source", prim_ty)
    data = tvm.ir.Var("data", relax.TensorType(dtype="float32", ndim=1))
    extent = tvm.ir.Var("extent", prim_ty)
    matched_ty = relax.TensorType([extent], "float32")
    matched = relax.DataflowVar("matched", matched_ty)

    block = relax.DataflowBlock(
        [
            relax.VarBinding(extent, source),
            relax.MatchCast(matched, data, matched_ty),
        ]
    )
    func = relax.Function([source, data], relax.SeqExpr([block], extent), prim_ty)
    return relax.transform.Normalize()(tvm.IRModule.from_expr(func))


def test_dataflowblock_pass_rejects_deleting_binding_role_of_symbolic_var():
    mod = _make_dataflow_block_with_dual_role_var()

    @relax.transform.dataflowblock_pass(opt_level=0)
    def delete_ordinary_binding(block, _mod, _ctx):
        return relax.DataflowBlock([block.bindings[1]])

    with pytest.raises(tvm.error.InternalError, match="global-scope Var"):
        delete_ordinary_binding(mod)


def test_dataflowblock_pass_rejects_rewriting_match_cast_role_of_binding_var():
    mod = _make_dataflow_block_with_dual_role_var()

    @relax.transform.dataflowblock_pass(opt_level=0)
    def rewrite_match_cast_symbol(block, _mod, _ctx):
        old_match_cast = block.bindings[1]
        replacement = tvm.ir.Var("replacement", "int64")
        replacement_ty = relax.TensorType([replacement], "float32")
        new_match_cast = relax.MatchCast(
            relax.DataflowVar(old_match_cast.var.name, replacement_ty),
            old_match_cast.value,
            replacement_ty,
        )
        return relax.DataflowBlock([block.bindings[0], new_match_cast])

    with pytest.raises(tvm.error.InternalError, match="symbolic Var declared by a MatchCast"):
        rewrite_match_cast_symbol(mod)


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
        if isinstance(e, relax.Var) and not tvm.ir.is_prim_expr(e):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    x, lv0, gv0 = old_vars

    new_mod = relax.transform.ToNonDataflow()(mod)

    new_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var) and not tvm.ir.is_prim_expr(e):
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
        @T.prim_func(s_tir=True)
        def exp(A_handle: T.handle, B_handle: T.handle):
            m = T.int64()
            n = T.int64()
            A = T.match_buffer(A_handle, (m, n), "float32")
            B = T.match_buffer(B_handle, (m, n), "float32")
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
    tvm.ir.assert_structural_equal(s1.args[0], s0.ty_args[0].shape)
    s2 = block.bindings[1].value
    tvm.ir.expr.GlobalVar
    assert s2.op.name_hint == "exp"


def test_call_tir_rewrite_with_interspersed_primitive_argument():
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def scale_add(
            A: T.Buffer((16,), "float32"),
            scale: T.float32,
            C: T.Buffer((16,), "float32"),
            B: T.Buffer((16,), "float32"),
        ):
            for i in range(16):
                B[i] = A[i] + scale * C[i]

        @R.function
        def main(
            A: R.Tensor((16,), "float32"),
            scale: R.Prim("float32"),
            C: R.Tensor((16,), "float32"),
        ) -> R.Tensor((16,), "float32"):
            R.func_attr({"relax.force_pure": True})
            B = R.call_tir(Module.scale_add, (A, scale, C), R.Tensor((16,), "float32"))
            return B

    after = relax.transform.CallTIRRewrite()(Module)
    func = after["main"]
    bindings = func.body.blocks[0].bindings
    output_buffer = bindings[0].var
    call = bindings[1].value

    assert call.op.name_hint == "scale_add"
    tvm.ir.assert_structural_equal(
        call.args,
        [func.params[0], func.params[1], func.params[2], output_buffer],
    )


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
            z = R.call_pure_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            closure = R.make_closure(Before.base, ())
            res = R.invoke_pure_closure(closure, (x,), ty_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Any:
            y = R.print(format="I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                y = R.add(x, x)
                q = R.call_pure_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Any:
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
            z = R.call_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            closure = R.make_closure(Expected.base, ())
            res = R.invoke_closure(closure, (x,), ty_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Any:
            y = R.print(format="I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})

            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                R.func_attr({"relax.force_pure": True})
                y = R.add(x, x)
                q = R.call_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Any:
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
    tvm.ir.assert_structural_equal(s1.args[0], s0.ty_args[0].shape)
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_call_dps_packed_rewrite_nested_tuple_output():
    """Flatten nested outputs for the packed ABI, then rebuild their Relax structure."""
    input_ty = relax.TensorType((2, 3), "float32")
    flat_output_types = [
        relax.TensorType((2, 3), "float32"),
        relax.TensorType((4,), "int32"),
        relax.TensorType((5, 6), "float16"),
    ]
    output_ty = tvm.ir.TupleType([flat_output_types[0], tvm.ir.TupleType(flat_output_types[1:])])

    x = relax.Var("x", input_ty)
    call = relax.Call(
        tvm.ir.Op.get("relax.call_dps_packed"),
        [relax.ExternFunc("test.op.nested_outputs"), relax.Tuple([x])],
        ty_args=[output_ty],
    )
    builder = relax.BlockBuilder()
    with builder.function("main", [x], attrs={"relax.force_pure": True}):
        out = builder.emit(call)
        builder.emit_func_output(out)

    after = relax.transform.CallTIRRewrite()(builder.get())
    relax.analysis.well_formed(after)
    func = after["main"]
    block = func.body.blocks[0]

    alloc_bindings = block.bindings[:3]
    for binding, expected_ty in zip(alloc_bindings, flat_output_types):
        assert binding.value.op.name == "relax.builtin.alloc_tensor"
        tvm.ir.assert_structural_equal(binding.var.ty, expected_ty)

    packed_call = block.bindings[3].value
    assert packed_call.op.global_symbol == "test.op.nested_outputs"
    assert packed_call.args[0].same_as(func.params[0])
    assert all(
        arg.same_as(binding.var) for arg, binding in zip(packed_call.args[1:], alloc_bindings)
    )

    rebuilt = block.bindings[4].value
    assert rebuilt.fields[0].same_as(alloc_bindings[0].var)
    assert rebuilt.fields[1].fields[0].same_as(alloc_bindings[1].var)
    assert rebuilt.fields[1].fields[1].same_as(alloc_bindings[2].var)


def test_call_tir_inplace_simple():
    # simple case: one inplace argument
    @tvm.script.ir_module
    class Input:
        @T.prim_func(s_tir=True)
        def zeros(A: T.Buffer((2, 3), "int32")):
            # just overwrites A with 0s
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
        @T.prim_func(s_tir=True)
        def zeros(A: T.Buffer((2, 3), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
        @T.prim_func(s_tir=True)
        def copy(
            A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32"), C: T.Buffer((2, 3), "int32")
        ):
            # copies the contents of C into A and B
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
        @T.prim_func(s_tir=True)
        def copy(
            A: T.Buffer((2, 3), "int32"), B: T.Buffer((2, 3), "int32"), C: T.Buffer((2, 3), "int32")
        ):
            # copies the contents of C into A and B
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
            _ = Expected.copy(x, y, z)
            gv0 = (x, y)
            return gv0

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_some_new():
    @tvm.script.ir_module
    class Input:
        @T.prim_func(s_tir=True)
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            C: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
            out2: T.Buffer((2, 3), "int32"),
        ):
            # copies the contents of C into A, out1, and out2
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
        @T.prim_func(s_tir=True)
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            C: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
            out2: T.Buffer((2, 3), "int32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.sblock("T_zeros"):
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
            gv0: R.Tensor((2, 3), dtype="int32") = R.emit_with_ty(
                "relax.builtin.alloc_tensor",
                (R.shape([2, 3]), R.dtype("int32"), R.prim_value(0), R.str("global")),
                (R.Tensor((2, 3), dtype="int32"),),
            )
            gv1: R.Tensor((2, 3), dtype="int32") = R.emit_with_ty(
                "relax.builtin.alloc_tensor",
                (R.shape([2, 3]), R.dtype("int32"), R.prim_value(0), R.str("global")),
                (R.Tensor((2, 3), dtype="int32"),),
            )
            _ = Expected.copy(x, y, z, gv0, gv1)
            gv2 = (x, gv0, gv1)
            return gv2

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_repeated_input():
    with pytest.raises(tvm.error.DiagnosticError):

        @tvm.script.ir_module
        class Input:
            @T.prim_func(s_tir=True)
            def func(
                A: T.Buffer((2, 3), "int32"),
                B: T.Buffer((2, 3), "int32"),
                C: T.Buffer((2, 3), "int32"),
            ):
                T.evaluate(0)

            @R.function
            def foo(
                x: R.Tensor((2, 3), "int32"),
                y: R.Tensor((2, 3), "int32"),
                z: R.Tensor((2, 3), "int32"),
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


def test_call_tir_inplace_all_new():
    with pytest.raises(tvm.error.DiagnosticError):

        @tvm.script.ir_module
        class Input:
            @T.prim_func(s_tir=True)
            def func(A: T.Buffer((2, 3), "int32")):
                T.evaluate(0)

            @R.function
            def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
                R.func_attr({"relax.force_pure": True})
                # cannot make the only output a fresh one
                gv0 = R.call_tir_inplace(Input.func, x, -1, R.Tensor((2, 3), dtype="int32"))
                return gv0


def test_inplace_mutation_with_tuple_argument_raises_error():
    """TIR PrimFuncs do not support Tuple arguments

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where each argument in the tuple may be expressed in
    TIR.  Here, `[[A]]` specifies a tuple of arguments, where the
    first argument is itself a tuple.  Since PrimFuncs do not support
    Tuple arguments, this is invalid.

    This is a regression test.  In previous implementations, this
    triggered a segfault rather than raising an exception.

    """
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module(s_tir=True)
        class Module:
            @R.function
            def main(A: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
                cls = Module
                gv1 = R.call_tir_inplace(
                    cls.multiply_by_two,
                    [[A]],
                    out_ty=R.Tensor((16,), dtype="float32"),
                    inplace_indices=[0],
                )
                return gv1

            @T.prim_func(private=True, s_tir=True)
            def multiply_by_two(A: T.Buffer((16,), "float32")):
                for i in range(16):
                    A[i] = A[i] * T.float32(2)


def test_inplace_mutation_with_non_tensor_argument_raises_error():
    """In-place argument must be a tensor

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where each argument in the tuple may be expressed in
    TIR.  Here, the argument `A` is not a tensor.

    This is a regression test.  In previous implementations, this
    triggered a segfault rather than raising an exception.

    """
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module(s_tir=True)
        class Module:
            @R.function
            def main(A: R.Any):
                gv1 = R.call_tir_inplace(
                    Module.multiply_by_two,
                    [A],
                    out_ty=R.Tensor((16,), dtype="float32"),
                    inplace_indices=[0],
                )
                return gv1

            @T.prim_func(private=True, s_tir=True)
            def multiply_by_two(A: T.Buffer((16,), "float32")):
                for i in range(16):
                    A[i] = A[i] * T.float32(2)


def test_inplace_mutation_with_incompatible_tensor_shape_raises_error():
    """In-place argument must have compatible shape

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where the shape of each in-place argument is compatible
    with the corresponding output.  Here, the shape of argument `A` is
    different than the output's shape (`[32]` as opposed to `[16]`).

    """
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module(s_tir=True)
        class Module:
            @R.function
            def main(A: R.Tensor([32], dtype="float32")):
                gv1 = R.call_tir_inplace(
                    Module.multiply_by_two,
                    [A],
                    out_ty=R.Tensor((16,), dtype="float32"),
                    inplace_indices=[0],
                )
                return gv1

            @T.prim_func(private=True, s_tir=True)
            def multiply_by_two(A: T.Buffer((16,), "float32")):
                for i in range(16):
                    A[i] = A[i] * T.float32(2)


def test_inplace_mutation_with_incompatible_tensor_dtype_raises_error():
    """In-place argument must have compatible dtype

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where the shape of each in-place argument is compatible
    with the corresponding output.  Here, the dtype of argument `A` is
    different than the output's dtype (`int32` as opposed to `float32`).

    """
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module(s_tir=True)
        class Module:
            @R.function
            def main(A: R.Tensor([16], dtype="int32")):
                gv1 = R.call_tir_inplace(
                    Module.multiply_by_two,
                    [A],
                    out_ty=R.Tensor((16,), dtype="float32"),
                    inplace_indices=[0],
                )
                return gv1

            @T.prim_func(private=True, s_tir=True)
            def multiply_by_two(A: T.Buffer((16,), "float32")):
                for i in range(16):
                    A[i] = A[i] * T.float32(2)


if __name__ == "__main__":
    tvm.testing.main()
