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
import tvm.testing

from tvm import relax as rx
from tvm import tir
from tvm.script import ir as I, relax as R, tir as T

m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
x = rx.Var("x", R.Tensor([m, n], "float32"))
cond = rx.Var("cond", R.Tensor([], "bool"))


def build_function(blocks, params=[]):
    """Returns relax.function with given blocks"""
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    func = rx.Function([x, cond] + params, seq_expr, R.Tensor("float32")).with_attr(
        "global_symbol", "foo"
    )
    return func


def test_var():
    # Error: Var gv0 is not defined
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, gv0)
    bindings = [rx.VarBinding(gv1, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # Error: Var gv0 is defined more than once
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(gv0, call_node), rx.VarBinding(gv0, call_node2)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_dataflow_var():
    # Error: DataflowVar lv0 is not defined
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, lv0)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # Error: DataflowVar gv0 is defined more than once
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(lv0, call_node), rx.VarBinding(lv0, call_node2)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # Error: DataflowVar lv0 is defined outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # Error: DataflowVar lv0 is used outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(lv0, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_param_var():
    v0 = rx.Var("v0", R.Tensor([m, n], "float32"))
    v1 = rx.Var("v1", R.Tensor([m, n], "float32"))
    v2 = rx.Var("v2", R.Tensor([m, n], "float32"))
    bb = rx.BlockBuilder()
    with bb.function("func1", [v0, v1]):
        gv0 = bb.emit(rx.op.add(v0, v1))
        bb.emit_func_output(gv0)
    with bb.function("func2", [v0, v2]):
        gv0 = bb.emit(rx.op.add(v2, v1))
        bb.emit_func_output(gv0)
    mod = bb.get()
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_global_var():
    # Error: GlobalVar GlobalVar0 is not defined
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    globalvar = rx.GlobalVar("GlobalVar0")
    call_node = rx.Call(
        op=tvm.ir.Op.get("relax.call_tir"),
        args=[globalvar, rx.Tuple([x]), rx.ShapeExpr([m, n])],
    )
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_symbolic_var():
    # Error: Symbolic Var new_s is not defined
    new_s = tir.Var("new_s", "int64")
    gv0 = rx.Var("gv0", R.Tensor([m, new_s], "int64"))
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_symbolic_var_across_functions():
    # Error: Symbolic Var s presents across different functions
    s = tir.Var("s", "int64")
    v0 = rx.Var("v0", R.Tensor([5, s], "float32"))
    v1 = rx.Var("v1", R.Tensor([s, 7], "float32"))
    bb = rx.BlockBuilder()
    with bb.function("func1", [v0]):
        bb.emit_func_output(v0)
    with bb.function("func2", [v1]):
        bb.emit_func_output(v1)
    mod = bb.get()
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_symbolic_var_invalid_type():
    with pytest.raises(
        tvm.TVMError, match="the value in ShapeStructInfo can only have dtype of int64"
    ):
        dim = tir.Var("dim", "float32")
        y = rx.Var("y", R.Tensor([dim], "float32"))
        gv0 = rx.Var("gv0", R.Tensor([dim], "float32"))
        call_node = rx.op.add(y, y)
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks, [y])
        mod = tvm.IRModule({rx.GlobalVar("foo"): func})
        assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_seq_expr():
    # Error: SeqExpr in VarBinding
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    # build a SeqExpr
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, gv0)
    _bindings = [rx.VarBinding(gv1, call_node)]
    _blocks = [rx.BindingBlock(_bindings)]
    _seq_expr = rx.SeqExpr(_blocks, gv1)
    # build a Binding with the SeqExpr as value
    bindings = [rx.VarBinding(gv0, _seq_expr)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_recursive():
    scalar_struct_info = rx.TensorStructInfo(shape=[], dtype="int32")
    gv0 = rx.Var("gv0", scalar_struct_info)
    f = rx.Var("f", rx.FuncStructInfo([scalar_struct_info], scalar_struct_info))
    ipt = rx.Var("ipt", scalar_struct_info)
    x0 = rx.Var("x0", scalar_struct_info)
    x1 = rx.Var("x1", scalar_struct_info)
    x2 = rx.Var("x2", scalar_struct_info)
    y = rx.Var("y", scalar_struct_info)
    inner_block = rx.BindingBlock(
        [rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, rx.Call(f, [x0]))]
    )
    inner_func = rx.Function([ipt], rx.SeqExpr([inner_block], y), scalar_struct_info)
    outer_block = rx.BindingBlock(
        [
            rx.VarBinding(f, inner_func),
            rx.VarBinding(x1, rx.const(1, "int32")),
            rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, [x1]))),
            rx.VarBinding(gv0, x2),
        ]
    )
    func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_struct_info)
    mod = tvm.IRModule.from_expr(func)
    normalized = rx.transform.Normalize()(mod)
    assert rx.analysis.well_formed(normalized)


def test_if():
    # Error: Var defined in true/false branch is invisible in the outer scope
    # except the return Var, i.e the var in the last stmt
    # v_in_if is invisible in the outer scope
    v_in_if = rx.Var("v_in_if", R.Tensor([m, n], "float32"))
    # gv0 is visible in the outer scope
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    # build true branch
    true_bindings = [
        rx.VarBinding(v_in_if, rx.op.add(x, x)),
        rx.VarBinding(gv0, rx.op.multiply(x, x)),
    ]
    true_blocks = [rx.BindingBlock(true_bindings)]
    true_seq_expr = rx.SeqExpr(true_blocks, true_blocks[-1].bindings[-1].var)
    # build false branch
    false_bindings = [
        rx.VarBinding(v_in_if, rx.op.multiply(x, x)),
        rx.VarBinding(gv0, rx.op.add(x, x)),
    ]
    false_blocks = [rx.BindingBlock(false_bindings)]
    false_seq_expr = rx.SeqExpr(false_blocks, false_blocks[-1].bindings[-1].var)
    # build If node
    if_node = rx.If(cond=cond, true_branch=true_seq_expr, false_branch=false_seq_expr)
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    # try to call v_in_if defined in the true/false branch
    bindings = [rx.VarBinding(gv0, if_node), rx.VarBinding(gv1, v_in_if)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=True)


def test_if_non_seq_body():
    # Error: If node has a body that is not a seq node
    if_node = rx.If(cond=cond, true_branch=x, false_branch=x)
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # on the other hand, if they're wrapped in a seq node, it's fine
    seq = rx.SeqExpr([], x)
    new_if_node = rx.If(cond=cond, true_branch=seq, false_branch=seq)
    new_blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    new_if_node,
                )
            ]
        )
    ]
    new_func = build_function(new_blocks)
    new_mod = tvm.IRModule.from_expr(new_func)
    # apply normalization to fill in checked_type_
    normalized = rx.transform.Normalize()(new_mod)
    assert rx.analysis.well_formed(normalized, check_struct_info=True)


def test_if_complex_condition():
    # Error: If condition must be a leaf expression
    cond_tuple = rx.Tuple([cond])
    cond_idx = rx.TupleGetItem(cond_tuple, 0)
    if_node = rx.If(cond_idx, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    cond_var = rx.Var("q", R.Tensor([], "bool"))
    new_if = rx.If(cond_var, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(cond_var, cond_idx),
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    new_if,
                ),
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm.IRModule.from_expr(func)
    # apply normalization to fill in checked_type_
    normalized = rx.transform.Normalize()(mod)
    assert rx.analysis.well_formed(normalized, check_struct_info=True)


def test_tuple_get_item_nested():
    # Error: The tuple value in tuple get item must be a leaf expression
    nested_tup = rx.Var(
        "t", rx.TupleStructInfo([rx.TupleStructInfo([rx.TensorStructInfo([], "int32")])])
    )
    double_idx = rx.TupleGetItem(rx.TupleGetItem(nested_tup, 0), 0)
    ret_var = rx.Var("r", R.Tensor([], "int32"))
    f = rx.Function(
        [nested_tup],
        rx.SeqExpr([rx.BindingBlock([rx.VarBinding(ret_var, double_idx)])], ret_var),
        ret_struct_info=R.Tensor(ndim=0, dtype="int32"),
    )
    f = f.with_attr("global_symbol", "f")
    mod = tvm.IRModule.from_expr(f)
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # okay with an intermediate binding
    first_idx = rx.TupleGetItem(nested_tup, 0)
    idx_var = rx.Var("v", rx.TupleStructInfo([rx.TensorStructInfo([], "int32")]))
    second_idx = rx.TupleGetItem(idx_var, 0)
    new_f = rx.Function(
        [nested_tup],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [rx.VarBinding(idx_var, first_idx), rx.VarBinding(ret_var, second_idx)]
                )
            ],
            ret_var,
        ),
        ret_struct_info=R.Tensor(ndim=0, dtype="int32"),
    )
    new_f = new_f.with_attr("global_symbol", "new_f")
    mod = tvm.IRModule.from_expr(new_f)
    # normalize in order to fill in checked type
    normalized = rx.transform.Normalize()(mod)
    assert rx.analysis.well_formed(normalized, check_struct_info=True)


def test_complex_seq_body():
    # Error: seq expr with a body that is not a leaf expression is not permitted
    x = rx.Var("x", R.Tensor([], "int32"))
    y = rx.Var("y", R.Tensor([], "int32"))
    func = rx.Function(
        [x, y],
        rx.SeqExpr([], rx.op.add(x, y)),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    mod = tvm.IRModule.from_expr(func)
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # but if the result is bound, then it's okay
    z = rx.Var("z", R.Tensor([], "int32"))
    new_func = rx.Function(
        [x, y],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [
                        rx.VarBinding(
                            var=z,
                            value=rx.op.add(x, y),
                        )
                    ]
                )
            ],
            z,
        ),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    new_mod = tvm.IRModule.from_expr(new_func)
    # normalize in order to fill in checked type
    normalized = rx.transform.Normalize()(new_mod)
    assert rx.analysis.well_formed(normalized, check_struct_info=True)


def test_inline_prim_func():
    # Error: inline prim_func is disallowed in Relax IR
    x = rx.Var("x", R.Tensor([], "int32"))
    y = rx.Var("y", R.Tensor([], "int32"))
    new_func = rx.Function(
        [],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [
                        rx.VarBinding(
                            var=x,
                            value=tir.PrimFunc([], tir.Evaluate(0)),
                        ),
                        rx.VarBinding(
                            var=y,
                            value=rx.Call(
                                op=tvm.ir.Op.get("relax.call_tir"),
                                args=[
                                    rx.GlobalVar("GlobalVar0"),
                                    rx.Tuple([x, tir.PrimFunc([], tir.Evaluate(0))]),
                                    rx.ShapeExpr([]),
                                ],
                            ),
                        ),
                    ]
                )
            ],
            y,
        ),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    new_mod = tvm.IRModule.from_expr(new_func)
    assert not rx.analysis.well_formed(new_mod, check_struct_info=False)


def test_ANF():
    # Error: Nested Call
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, rx.op.add(x, x))
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)

    # Error: Call Node in Tuple
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    bindings = [rx.VarBinding(gv0, rx.Tuple((x, rx.op.add(x, x))))]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm.IRModule({rx.GlobalVar("foo"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_global_var_vs_gsymbol():
    # Error: gsymbol "main1" not equals to the name in global var "main"
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    bindings = [rx.VarBinding(gv0, x)]
    blocks = [rx.DataflowBlock(bindings)]
    func = rx.Function(
        [x],
        rx.SeqExpr(blocks, gv0),
        R.Tensor(ndim=2, dtype="float32"),
    ).with_attr("global_symbol", "main1")
    mod = tvm.IRModule({rx.GlobalVar("main"): func})
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_nested_dataflow():
    scalar_struct_info = rx.TensorStructInfo(shape=[], dtype="int32")
    gv0 = rx.Var("gv0", scalar_struct_info)
    f = rx.DataflowVar("f", rx.FuncStructInfo([], scalar_struct_info))
    x0 = rx.DataflowVar("x0", scalar_struct_info)
    x1 = rx.DataflowVar("x1", scalar_struct_info)
    x2 = rx.DataflowVar("x2", scalar_struct_info)
    y = rx.Var("y", scalar_struct_info)
    inner_block = rx.DataflowBlock([rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, x0)])
    inner_func = rx.Function([], rx.SeqExpr([inner_block], y), scalar_struct_info)
    outer_block = rx.DataflowBlock(
        [
            rx.VarBinding(x1, rx.const(1, "int32")),
            rx.VarBinding(f, inner_func),
            rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, []))),
            rx.VarBinding(gv0, x2),
        ]
    )
    func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_struct_info)
    mod = tvm.IRModule.from_expr(func)
    normalized = rx.transform.Normalize()(mod)
    assert rx.analysis.well_formed(normalized)


def test_sinfo_args_tir_var_used_before_define_call_packed():
    # Error: Symbolic Var m1, n1 are not defined
    m1 = tir.Var("m1", "int64")
    n1 = tir.Var("n1", "int64")
    call = R.call_packed("my_func", x, sinfo_args=R.Tensor((m1, n1), "float32"))
    func = build_function([rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])])
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_sinfo_args_tir_var_used_before_define_call_tir():
    # Error: Symbolic Var m1, n1 are not defined
    m1 = tir.Var("m1", "int64")
    n1 = tir.Var("n1", "int64")
    call = R.call_dps_packed("my_func", x, out_sinfo=R.Tensor((m1, n1), "float32"))
    func = build_function([rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])])
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod, check_struct_info=False)


def test_sinfo_erase_to_well_formed():
    # Error: The return sinfo contains undefined symbolic vars
    """
    @R.function
    def foo(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m1", "n1"), dtype="float32"):
        m = T.int64()
        n = T.int64()
        gv = R.call_dps_packed("my_func", (x,), out_sinfo=R.Tensor((m, n), dtype="float32"))
        return gv
    """
    m1 = tir.Var("m1", "int64")
    n1 = tir.Var("n1", "int64")
    call = R.call_dps_packed("my_func", x, out_sinfo=R.Tensor((m, n), "float32"))
    blocks = [rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])]
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    func = rx.Function([x], seq_expr, R.Tensor((m1, n1), "float32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)


def test_func_sinfo_well_formed():
    @R.function
    def foo():
        @R.function
        def local(x: R.Tensor(["m", "n"], "float32")):
            return x

        return local

    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(foo))
    assert rx.analysis.well_formed(mod)


def test_conditional_in_dataflow_block():
    # error: not allowed to have a conditional inside a dataflow block
    x = rx.Var("x", rx.TensorStructInfo([], dtype="int32"))
    y = rx.Var("y", rx.TensorStructInfo([], dtype="int32"))
    block = rx.DataflowBlock([rx.VarBinding(y, rx.If(rx.const(True, dtype="bool"), x, x))])
    func = rx.Function([x], rx.SeqExpr([block], y), R.Tensor((), dtype="int32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)


def test_unlabeled_impure():
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print(x, format="{}"))])
    # print is impure, but the function is not labeled as impure
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)


def test_labeled_impure():
    # the function is labeled impure so the impure operation is permitted
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print(x, format="{}"))])
    # print is impure, but the function is not labeled as impure
    func = rx.Function(
        [x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32"), is_pure=False
    ).with_attrs({"global_symbol": "foo"})
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert rx.analysis.well_formed(mod)


def test_force_pure():
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print(x, format="{}"))])
    # print is impure, but force_pure overrides the judgment
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attrs(
        {"global_symbol": "foo", "relax.force_pure": True}
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert rx.analysis.well_formed(mod)


def test_force_pure_improper():
    # we require both the is_pure and force_pure flags to be set together
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    # otherwise inoffensive, but the flags are wrong
    func = rx.Function(
        [x], rx.SeqExpr([], x), R.Tensor((), dtype="int32"), is_pure=False
    ).with_attrs({"global_symbol": "foo", "relax.force_pure": True})
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)


def test_impure_in_dataflow_block(capfd):
    # even if force_pure is set, an impure operation cannot appear in a dataflow block
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.DataflowVar("y")
    block = rx.DataflowBlock([rx.VarBinding(y, rx.op.print(x, format="{}"))])
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attrs(
        {"global_symbol": "foo", "relax.force_pure": True}
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)

    _stdout, stderr = capfd.readouterr()
    assert "R.print" in stderr


def test_well_formed_function():
    """Relax's well-formed check can be applied on a function"""

    @R.function
    def func(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
        return R.matmul(A, B)

    assert rx.analysis.well_formed(func)


def test_well_formed_function_referencing_global_var():
    """GlobalVar may refer to other functions in the module

    If validating that a IRModule is well-formed, the GlobalVar must
    have a definition.  If validating that a relax.Function is
    well-formed, no GlobalVar definitions are available.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
            return Module.subroutine(A, B)

        @R.function(private=True)
        def subroutine(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
            return R.matmul(A, B)

    assert rx.analysis.well_formed(Module)
    assert rx.analysis.well_formed(Module["main"])
    assert rx.analysis.well_formed(Module["subroutine"])


def test_pass_dltensor_arg_to_tir():
    """Relax may pass R.Tensor as DLTensor

    In TIR, a `DLTensor*` argument with unknown shape and dtype is
    represented as a `tir.Var` with
    `tvm::PrimType(DataType::Handle())`, and with no entry in the
    `PrimFuncNode::buffer_map`.  In Relax, this is represented as
    `R.Tensor`.  Calls from Relax to TIR that pass a tensor of unknown
    rank/shape are well-formed.

    In the test case below, a TIR function accepts an arbitrary
    `R.Tensor`, and returns a boolean value based on inspection of the
    runtime datatype.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor) -> R.Prim("bool"):
            return Module.is_bfloat16_dtype(A)

        @T.prim_func(private=True)
        def is_bfloat16_dtype(tensor: T.handle) -> T.bool:
            T.func_attr({"tir.is_scheduled": True, "tir.is_host_func": True})

            # From #include <tvm/tir/builtin.h>
            kArrTypeCode = T.meta_var(5)
            kArrTypeBits = T.meta_var(6)
            kArrTypeLanes = T.meta_var(7)

            # From #include <dlpack/dlpack.h>
            kDLBfloat = T.meta_var(4)

            type_code = T.tvm_struct_get(tensor, 0, kArrTypeCode, dtype="uint8")
            type_bits = T.tvm_struct_get(tensor, 0, kArrTypeBits, dtype="uint8")
            type_lanes = T.tvm_struct_get(tensor, 0, kArrTypeLanes, dtype="uint16")

            is_bfloat16: T.bool = (
                (type_code == kDLBfloat) and (type_bits == 16) and (type_lanes == 1)
            )
            return is_bfloat16

    assert rx.analysis.well_formed(Module)


def test_call_tir_with_matching_arguments():
    """R.call_tir is well-formed when called with matching arguments"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([16], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert rx.analysis.well_formed(Module)


def test_call_tir_input_ndim():
    """Arguments to R.call_tir must have the correct dimensionality

    Here, the `add_one` function expects a 1-d input tensor, but is
    called with a 2-d tensor.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([4, 4], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([16], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_output_ndim():
    """Output shape R.call_tir must have the correct dimensionality

    Here, the `add_one` function requires a 1-d output tensor, but is
    provided with a 2-d tensor.
    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([4, 4], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_input_shape():
    """Arguments to R.call_tir must have the correct shape

    Here, the `add_one` function expects an input tensor with 16
    elements, but is called with an input tensor with 32 elements.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([32], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([16], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_output_shape():
    """Output shape R.call_tir must have the correct shape

    Here, the `add_one` function requires an output tensor with 16
    elements, but is provided an output tensor with 32 elements.
    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([32], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_input_dtype():
    """Arguments to R.call_tir must have the correct dtype

    Here, the `add_one` function expects an input tensor containing
    float16 value, but is called with an input tensor containing
    float32 values.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([16], "float16"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_output_dtype():
    """Output shape R.call_tir must have the correct shape

    Here, the `add_one` function requires an output tensor that may be
    populated with float16 values, but is provided an output tensor
    that may be populated with float32 elements.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, out_sinfo=R.Tensor([16], "float32"))
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16"), B: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_with_correct_dynamic_output_shape():
    """Output shape R.call_tir may not be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  This is legal,
    since the output shape is determined by the `out_sinfo` parameter.

    Inability to verify the output shape does not mean that the output
    shape is invalid.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, out_sinfo=R.Tensor([2, 8], "float16"))
            return B

        @T.prim_func
        def reshape(A: T.Buffer(16, "float16"), B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            B = T.match_buffer(B_handle, [M, N], dtype="float16")

            for i, j in T.grid(M, N):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    assert rx.analysis.well_formed(Module)


@pytest.mark.xfail(reason="Not supported")
def test_call_tir_with_incorrect_dynamic_output_shape():
    """Output shape R.call_tir may not be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  Even though the
    IRModule will not provide well-defined output due to the
    out-of-bounds read from buffer A, catching this error is beyond
    the current scope of the Relax well-formed checker.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, out_sinfo=R.Tensor([16, 16], "float16"))
            return B

        @T.prim_func
        def reshape(A: T.Buffer(16, "float16"), B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            B = T.match_buffer(B_handle, [M, N], dtype="float16")

            for i, j in T.grid(M, N):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    assert not rx.analysis.well_formed(Module)


def test_call_tir_incorrect_dimensionality_of_output_shape():
    """Dimensionality may be verified

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.

    Even though the output shape may not be inferred from the input
    arguments, the output dimensionality can still be inferred from
    the PrimFunc signature.  The IRModule below is ill-formed, because
    the PrimFunc requires a 2-d output argument, but is provided with
    a 3-d output argument.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, out_sinfo=R.Tensor([2, 4, 2], "float16"))
            return B

        @T.prim_func
        def reshape(A: T.Buffer(16, "float16"), B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            B = T.match_buffer(B_handle, [M, N], dtype="float16")

            for i, j in T.grid(M, N):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    assert not rx.analysis.well_formed(Module)


@pytest.mark.xfail(reason="Not yet supported")
def test_call_tir_output_shape_with_mixed_static_and_dynamic():
    """Some dimensions of the R.call_tir output shape may be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  This is legal,
    since the output shape is taken from the `out_sinfo` parameter.

    Identifying this failure mode is not yet supported in the current
    implementation.  This is because the output is inferred as
    `R.Tensor(ndim=3, dtype="float16")`, and the explicit `out_sinfo`
    is a 3-d tensor.  The mismatch in the first dimension is not yet
    counted, because the entire tensor shape is removed by
    `EraseToWellDefined`.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([256], "float16")):
            B = R.call_tir(Module.reshape, A, out_sinfo=R.Tensor([8, 16, 2], "float16"))
            return B

        @T.prim_func
        def reshape(A: T.Buffer(256, "float16"), B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            B = T.match_buffer(B_handle, [16, M, N], dtype="float16")

            for i, j, k in T.grid(16, M, N):
                with T.block("compute"):
                    vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi * N * M + vj * N + vk]

    assert not rx.analysis.well_formed(Module)


def test_call_tir_with_correct_inferred_dynamic_output_shape():
    """Some dynamic output shapes of R.call_tir may be inferred

    Here, the `flatten` function is dynamic, and will flatten any 2-d
    TIR buffer.  Even though it is dynamic, the input shapes are
    sufficient to infer that `M==8` and `N==4`.  As a result, the
    output shape of `[M*N]` can be inferred to be `[32]`, and the
    shape specified in `out_sinfo` can be validated.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([8, 4], "float16")):
            B = R.call_tir(Module.flatten, A, out_sinfo=R.Tensor([32], "float16"))
            return B

        @T.prim_func
        def flatten(A_handle: T.handle, B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            A = T.match_buffer(A_handle, [M, N], dtype="float16")
            B = T.match_buffer(B_handle, [M * N], dtype="float16")

            for i in T.grid(M * N):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    assert rx.analysis.well_formed(Module)


def test_call_tir_with_incorrect_inferred_dynamic_output_shape():
    """Some dynamic output shapes of R.call_tir may be inferred

    Here, the `flatten` function is dynamic, and will flatten any 2-d
    TIR buffer.  Even though it is dynamic, the input shapes are
    sufficient to infer that `M==8` and `N==4`.  As a result, the
    output shape of `[M*N]` can be inferred to be `[32]`, and the
    shape specified in `out_sinfo` can be validated.

    This unit test is identical to the above test
    `test_call_tir_with_correct_inferred_dynamic_output_shape`, except
    that the output shape is explicitly specified as `[64]`, which is
    caught as a mismatch from the expected output shape.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([8, 4], "float16")):
            B = R.call_tir(Module.flatten, A, out_sinfo=R.Tensor([64], "float16"))
            return B

        @T.prim_func
        def flatten(A_handle: T.handle, B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            A = T.match_buffer(A_handle, [M, N], dtype="float16")
            B = T.match_buffer(B_handle, [M * N], dtype="float16")

            for i in T.grid(M * N):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    assert not rx.analysis.well_formed(Module)


def test_call_tir_with_dtensor_arguments():
    """R.call_tir and R.dist.call_tir share the same operation

    Both `R.call_tir` and `R.dist.call_tir` produce the same
    "relax.call_tir" operation, differing only in the StructInfo of
    their arguments.  Normalization of "relax.call_tir" must handle
    `R.DTensor` arguments.

    """

    # from tvm.script.parser import relax as R

    @I.ir_module
    class Module:
        I.module_attrs({"device_num": 4})
        I.module_global_infos({"mesh": [R.dist.device_mesh([4], I.Range(0, 4))]})

        @R.function
        def main(A: R.dist.DTensor([8, 4], "float16", "mesh[0]", "S[0]")):
            B = R.dist.call_tir(
                Module.flatten, A, out_sinfo=R.dist.DTensor([64], "float16", "mesh[0]", "S[0]")
            )
            return B

        @T.prim_func
        def flatten(A_handle: T.handle, B_handle: T.handle):
            M = T.int64()
            N = T.int64()
            A = T.match_buffer(A_handle, [M, N], dtype="float16")
            B = T.match_buffer(B_handle, [M * N], dtype="float16")

            for i in T.grid(M * N):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    assert rx.analysis.well_formed(Module)


def test_call_tir_inplace_with_correct_shapes():
    """R.call_tir_inplace is well-formed when called with matching arguments"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir_inplace(
                Module.add_one,
                A,
                inplace_indices=[0],
                out_sinfo=R.Tensor([16], "float16"),
            )
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    A[vi] = A[vi] + T.float16(1.0)

    assert rx.analysis.well_formed(Module)


def test_call_tir_inplace_with_incorrect_shapes():
    """R.call_tir_inplace is ill-formed when output shape does not match input"""

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir_inplace(
                Module.add_one,
                A,
                inplace_indices=[0],
                out_sinfo=R.Tensor([32], "float16"),
            )
            return B

        @T.prim_func
        def add_one(A: T.Buffer(16, "float16")):
            for i in range(16):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    A[vi] = A[vi] + T.float16(1.0)

    assert not rx.analysis.well_formed(Module)


def test_call_tir_inplace_with_some_allocated_outputs():
    """R.call_tir_inplace may contain some non-inplace outputs"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16"), B: R.Tensor([32], "float16")):
            out = R.call_tir_inplace(
                Module.add_one,
                (A, B),
                inplace_indices=[-1, 1],
                out_sinfo=[
                    R.Tensor([16], "float16"),
                    R.Tensor([32], "float16"),
                ],
            )
            return out

        @T.prim_func
        def add_one(
            A: T.Buffer(16, "float16"),
            B: T.Buffer(32, "float16"),
            C: T.Buffer(16, "float16"),
        ):
            for i in range(32):
                with T.block("inplace_B"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = B[vi] + T.float16(1.0)

            for i in range(16):
                with T.block("output_C"):
                    vi = T.axis.remap("S", [i])
                    C[vi] = A[vi] + T.float16(1.0)

    assert rx.analysis.well_formed(Module)


def test_var_binding_must_have_compatible_struct_info():
    """Variables must accurately describe their contents

    To be well-formed, the inferred struct info must not conflict with
    the StructInfo annotations.

    """

    # The function is equivalent to the TVMScript below.  However,
    # TVMScript applies additional checks that would catch this error
    # while parsing.  In order to validate the well-formed checker
    # itself, this test directly constructs the function withoutusing
    # TVMScript, skipping the TVMScript-specific checks.
    #
    # @R.function
    # def main(
    #     A: R.Tensor(shape=[128, 32], dtype="float32"),
    # ):
    #     B: R.Tensor(shape=[128, 32], dtype="int32") = A
    #     return B

    param = tvm.relax.Var("A", R.Tensor(shape=[128, 32], dtype="float32"))
    var = tvm.relax.Var("B", R.Tensor(shape=[128, 32], dtype="int32"))
    binding = tvm.relax.VarBinding(var, param)
    body = tvm.relax.SeqExpr([tvm.relax.BindingBlock([binding])], var)
    tvm.relax.expr._update_struct_info(body, var.struct_info)
    main = tvm.relax.Function([param], body)

    assert not rx.analysis.well_formed(main)


def test_var_binding_may_have_less_constrained_struct_info():
    """StructInfo of variable may be less specific than expression

    The StructInfo annotation of a variable is not required to be an
    exact match to the expression's StructInfo, and may provide less
    specific information than the inference would provide.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            B: R.Object = R.add(A, A)
            return B

    assert isinstance(
        Module["main"].body.blocks[0].bindings[0].var.struct_info, tvm.relax.ObjectStructInfo
    ), "Validity of this test requires a variable with R.Object struct info"

    assert rx.analysis.well_formed(Module)


def test_var_binding_with_incomplete_struct_info_must_be_consistent():
    """StructInfo of variable must be accurate

    Even though StructInfo annotation may be less specific, the
    information that they do contain must be correct.

    """

    # The function is equivalent to the TVMScript below.  However,
    # TVMScript applies additional checks that would catch this error
    # while parsing.  In order to validate the well-formed checker
    # itself, this test directly constructs the function withoutusing
    # TVMScript, skipping the TVMScript-specific checks.
    #
    #   @R.function
    #   def main(
    #       A: R.Tensor(shape=[128, 32], dtype="float32"),
    #   ):
    #       B: R.Tensor(ndim=3) = A
    #       return B

    param = tvm.relax.Var("A", R.Tensor(shape=[128, 32], dtype="float32"))
    var = tvm.relax.Var("B", R.Tensor(ndim=3, dtype="int32"))
    binding = tvm.relax.VarBinding(var, param)
    body = tvm.relax.SeqExpr([tvm.relax.BindingBlock([binding])], var)
    tvm.relax.expr._update_struct_info(body, var.struct_info)
    main = tvm.relax.Function([param], body)

    assert not rx.analysis.well_formed(main)


def test_incomplete_struct_info_must_be_consistent():
    """StructInfo annotations must be accurate

    Even though StructInfo annotation may be less specific, the
    information that they do contain must be correct.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(ndim=3) = R.add(A, B)
            return C

    assert not rx.analysis.well_formed(Module)


def test_struct_info_annotations_must_be_correct():
    """StructInfo annotations must be correct

    To be well-formed, the inferred struct info must not conflict with
    the StructInfo annotations.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(shape=[128, 32], dtype="int32") = R.add(A, B)
            return C

    assert not rx.analysis.well_formed(Module)


def test_struct_info_may_be_incomplete():
    """StructInfo annotations may be less specific

    The StructInfo annotations are not required to be an exact match
    to the inferred StructInfo, and may provide less specific
    information than the inference would provide.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Object = R.add(A, B)
            return C

    assert rx.analysis.well_formed(Module)


def test_incomplete_struct_info_must_be_consistent():
    """StructInfo annotations must be accurate

    Even though StructInfo annotation may be less specific, the
    information that they do contain must be correct.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(ndim=3) = R.add(A, B)
            return C

    assert not rx.analysis.well_formed(Module)


if __name__ == "__main__":
    tvm.testing.main()
