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
from tvm.script import relax as R
from tvm.script import tir as T

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


def test_impure_in_dataflow_block():
    # even if force_pure is set, an impure operation cannot appear in a dataflow block
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.DataflowVar("y")
    block = rx.DataflowBlock([rx.VarBinding(y, rx.op.print(x, format="{}"))])
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attrs(
        {"global_symbol": "foo", "relax.force_pure": True}
    )
    mod = rx.transform.Normalize()(tvm.IRModule.from_expr(func))
    assert not rx.analysis.well_formed(mod)


if __name__ == "__main__":
    tvm.testing.main()
