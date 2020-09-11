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
"""Test that type checker correcly computes types
   for expressions.
"""
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import op, transform, analysis
from tvm.relay import Any


def run_infer_type(expr, mod=None):
    if not mod:
        mod = tvm.IRModule.from_expr(expr)
        mod = transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(expr, relay.Function) else entry.body
    else:
        if isinstance(expr, relay.GlobalVar):
            gv = expr.name_hint
        else:
            func = expr
            if not isinstance(expr, relay.Function):
                func = relay.Function(analysis.free_vars(expr), expr)
            mod["main"] = func
            gv = "main"
        mod = transform.InferType()(mod)

        if isinstance(expr, (relay.GlobalVar, relay.Function)):
            return mod[gv]
        return mod[gv].body


def assert_has_type(expr, typ, mod=tvm.IRModule({})):
    checked_expr = run_infer_type(expr, mod)
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (checked_type, typ))


# initializes simple ADT for tests
def initialize_box_adt(mod):
    box = relay.GlobalTypeVar("box")
    tv = relay.TypeVar("tv")
    constructor = relay.Constructor("constructor", [tv], box)
    data = relay.TypeData(box, [tv], [constructor])
    mod[box] = data
    return (box, constructor)


def test_monomorphic_let():
    "Program: let %x = 1; %x"
    sb = relay.ScopeBuilder()
    x = sb.let("x", relay.const(1.0, "float64"))
    sb.ret(x)
    xchecked = run_infer_type(sb.get())
    assert xchecked.checked_type == relay.scalar_type("float64")


def test_single_op():
    "Program: fn (%x : float32) { let %t1 = f(%x); %t1 }"
    x = relay.var("x", shape=[])
    func = relay.Function([x], op.log(x))
    ttype = relay.TensorType([], dtype="float32")
    assert_has_type(func, relay.FuncType([ttype], ttype))


def test_add_broadcast_op():
    """
    Program:
        fn (%x: Tensor[(10, 4), float32], %y: Tensor[(5, 10, 1), float32])
            -> Tensor[(5, 10, 4), float32] {
            %x + %y
        }
    """
    x = relay.var("x", shape=(10, 4))
    y = relay.var("y", shape=(5, 10, 1))
    z = x + y
    func = relay.Function([x, y], z)
    t1 = relay.TensorType((10, 4), "float32")
    t2 = relay.TensorType((5, 10, 1), "float32")
    t3 = relay.TensorType((5, 10, 4), "float32")
    expected_ty = relay.FuncType([t1, t2], t3)
    assert_has_type(func, expected_ty)


def test_dual_op():
    """Program:
    fn (%x : Tensor[(10, 10), float32]) {
      let %t1 = log(x);
      let %t2 = add(%t1, %x);
      %t1
    }
    """
    tp = relay.TensorType((10, 10), "float32")
    x = relay.var("x", tp)
    sb = relay.ScopeBuilder()
    t1 = sb.let("t1", relay.log(x))
    t2 = sb.let("t2", relay.add(t1, x))
    sb.ret(t2)
    f = relay.Function([x], sb.get())
    fchecked = run_infer_type(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_decl():
    """Program:
    def @f(%x : Tensor[(10, 10), float32]) {
        log(%x)
    }
    """
    tp = relay.TensorType((10, 10))
    x = relay.var("x", tp)
    f = relay.Function([x], relay.log(x))
    fchecked = run_infer_type(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_recursion():
    """
    Program:
       def @f(%n: int32, %data: float32) -> float32 {
          if (%n == 0) {
              %data
          } else {
              @f(%n - 1, log(%data))
          }
       }
    """
    sb = relay.ScopeBuilder()
    f = relay.GlobalVar("f")
    ti32 = relay.scalar_type("int32")
    tf32 = relay.scalar_type("float32")
    n = relay.var("n", ti32)
    data = relay.var("data", tf32)

    with sb.if_scope(relay.equal(n, relay.const(0, ti32))):
        sb.ret(data)
    with sb.else_scope():
        sb.ret(f(relay.subtract(n, relay.const(1, ti32)), relay.log(data)))
    mod = tvm.IRModule()
    mod[f] = relay.Function([n, data], sb.get())
    assert "@f(%1, %2) /* ty=float32 */" in mod.astext()
    assert mod[f].checked_type == relay.FuncType([ti32, tf32], tf32)


def test_incomplete_call():
    tt = relay.scalar_type("int32")
    x = relay.var("x", tt)
    f = relay.var("f")
    func = relay.Function([x, f], relay.Call(f, [x]), tt)

    ft = run_infer_type(func)
    f_type = relay.FuncType([tt], tt)
    assert ft.checked_type == relay.FuncType([tt, f_type], tt)


def test_higher_order_argument():
    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    id_func = relay.Function([x], x, a, [a])

    b = relay.TypeVar("b")
    f = relay.Var("f", relay.FuncType([b], b))
    y = relay.Var("y", b)
    ho_func = relay.Function([f, y], f(y), b, [b])

    # id func should be an acceptable argument to the higher-order
    # function even though id_func takes a type parameter
    ho_call = ho_func(id_func, relay.const(0, "int32"))

    hc = run_infer_type(ho_call)
    expected = relay.scalar_type("int32")
    assert hc.checked_type == expected


def test_higher_order_return():
    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    id_func = relay.Function([x], x, a, [a])

    b = relay.TypeVar("b")
    nested_id = relay.Function([], id_func, relay.FuncType([b], b), [b])

    ft = run_infer_type(nested_id)
    assert ft.checked_type == relay.FuncType([], relay.FuncType([b], b), [b])


def test_higher_order_nested():
    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    id_func = relay.Function([x], x, a, [a])

    choice_t = relay.FuncType([], relay.scalar_type("bool"))
    f = relay.Var("f", choice_t)

    b = relay.TypeVar("b")
    z = relay.Var("z")
    top = relay.Function(
        [f], relay.If(f(), id_func, relay.Function([z], z)), relay.FuncType([b], b), [b]
    )

    expected = relay.FuncType([choice_t], relay.FuncType([b], b), [b])
    ft = run_infer_type(top)
    assert ft.checked_type == expected


def test_tuple():
    tp = relay.TensorType((10,))
    x = relay.var("x", tp)
    res = relay.Tuple([x, x])
    assert run_infer_type(res).checked_type == relay.TupleType([tp, tp])


def test_ref():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    r = relay.RefCreate(x)
    st = relay.scalar_type("float32")
    assert run_infer_type(r).checked_type == relay.RefType(st)
    g = relay.RefRead(r)
    assert run_infer_type(g).checked_type == st
    w = relay.RefWrite(r, y)
    assert run_infer_type(w).checked_type == relay.TupleType([])


def test_free_expr():
    return
    x = relay.var("x", "float32")
    y = relay.add(x, x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.scalar_type("float32")
    assert x.vid.same_as(yy.args[0].vid)


def test_type_args():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    ty_z = run_infer_type(z)
    ty_args = ty_z.type_args
    assert len(ty_args) == 2
    assert ty_args[0].dtype == "float32"
    assert ty_args[1].dtype == "float32"
    sh1 = ty_args[0].shape
    sh2 = ty_args[1].shape
    assert sh1[0].value == 10
    assert sh1[1].value == 10
    assert sh2[0].value == 1
    assert sh2[1].value == 10


def test_global_var_recursion():
    mod = tvm.IRModule({})
    gv = relay.GlobalVar("main")
    x = relay.var("x", shape=[])
    tt = relay.scalar_type("float32")

    func = relay.Function([x], relay.Call(gv, [x]), tt)
    mod[gv] = func

    ft = run_infer_type(gv, mod)
    assert ft.checked_type == relay.FuncType([tt], tt)


def test_equal():
    i = relay.var("i", shape=[], dtype="int32")
    eq = op.equal(i, relay.const(0, dtype="int32"))
    func = relay.Function([i], eq)
    ft = run_infer_type(func)

    assert ft.checked_type == relay.FuncType(
        [relay.scalar_type("int32")], relay.scalar_type("bool")
    )


def test_constructor_type():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    ct = run_infer_type(relay.Function([x], constructor(x), box(a), [a]), mod)
    expected = relay.FuncType([a], box(a), [a])
    assert ct.checked_type == expected


def test_constructor_call():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    box_unit = constructor(relay.Tuple([]))
    box_constant = constructor(relay.const(0, "float32"))

    ut = run_infer_type(box_unit, mod)
    ct = run_infer_type(box_constant, mod)
    assert ut.checked_type == box(relay.TupleType([]))
    assert ct.checked_type == box(relay.TensorType((), "float32"))


def test_adt_match():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    v = relay.Var("v", relay.TensorType((), "float32"))
    match = relay.Match(
        constructor(relay.const(0, "float32")),
        [
            relay.Clause(
                relay.PatternConstructor(constructor, [relay.PatternVar(v)]), relay.Tuple([])
            ),
            # redundant but shouldn't matter to typechecking
            relay.Clause(relay.PatternWildcard(), relay.Tuple([])),
        ],
    )

    mt = run_infer_type(match, mod)
    assert mt.checked_type == relay.TupleType([])


def test_adt_match_type_annotations():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    # the only type annotation is inside the match pattern var
    # but that should be enough info
    tt = relay.TensorType((2, 2), "float32")
    x = relay.Var("x")
    mv = relay.Var("mv", tt)
    match = relay.Match(
        constructor(x),
        [
            relay.Clause(
                relay.PatternConstructor(constructor, [relay.PatternVar(mv)]), relay.Tuple([])
            )
        ],
    )

    func = relay.Function([x], match)
    ft = run_infer_type(func, mod)
    assert ft.checked_type == relay.FuncType([tt], relay.TupleType([]))


def test_let_polymorphism():
    id = relay.Var("id")
    xt = relay.TypeVar("xt")
    x = relay.Var("x", xt)
    body = relay.Tuple([id(relay.const(1)), id(relay.Tuple([]))])
    body = relay.Let(id, relay.Function([x], x, xt, [xt]), body)
    body = run_infer_type(body)
    int32 = relay.TensorType((), "int32")
    tvm.ir.assert_structural_equal(body.checked_type, relay.TupleType([int32, relay.TupleType([])]))


def test_if():
    choice_t = relay.FuncType([], relay.scalar_type("bool"))
    f = relay.Var("f", choice_t)
    true_branch = relay.Var("True", relay.TensorType([Any(), 1], dtype="float32"))
    false_branch = relay.Var("False", relay.TensorType([Any(), Any()], dtype="float32"))
    top = relay.Function([f, true_branch, false_branch], relay.If(f(), true_branch, false_branch))
    ft = run_infer_type(top)
    tvm.ir.assert_structural_equal(ft.ret_type, relay.TensorType([Any(), 1], dtype="float32"))


def test_type_arg_infer():
    code = """
#[version = "0.0.5"]
def @id[A](%x: A) -> A {
  %x
}
def @main(%f: float32) -> float32 {
  @id(%f)
}
"""
    mod = tvm.parser.fromtext(code)
    mod = transform.InferType()(mod)
    tvm.ir.assert_structural_equal(mod["main"].body.type_args, [relay.TensorType((), "float32")])


if __name__ == "__main__":
    pytest.main([__file__])
