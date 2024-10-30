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
import numpy as np

import tvm
from tvm import IRModule, relay
from tvm.relay import op, transform
from tvm.relay.op import op as _op
from tvm.script import tir as T


def infer_mod(mod, annotate_spans=True):
    if annotate_spans:
        mod = relay.transform.AnnotateSpans()(mod)

    mod = transform.InferType()(mod)
    return mod


def infer_expr(expr):
    transform.InferTypeLocal(expr)
    return expr


def assert_has_type(expr, typ, mod=None):
    if not mod:
        mod = tvm.IRModule({})

    mod["main"] = expr
    mod = infer_mod(mod)
    checked_expr = mod["main"]
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (checked_type, typ))


def initialize_box_adt(mod):
    # initializes simple ADT for tests
    box = relay.GlobalTypeVar("box")
    tv = relay.TypeVar("tv")
    constructor = relay.Constructor("constructor", [tv], box)
    data = relay.TypeData(box, [tv], [constructor])
    mod[box] = data
    return box, constructor


def test_monomorphic_let():
    "Program: let %x = 1; %x"
    # TODO(@jroesch): this seems whack.
    sb = relay.ScopeBuilder()
    x = relay.var("x", dtype="float64", shape=())
    x = sb.let(x, relay.const(1.0, "float64"))
    sb.ret(x)
    xchecked = infer_expr(sb.get())
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
    fchecked = infer_expr(f)
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
    fchecked = infer_expr(f)
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
    mod = infer_mod(mod)
    assert "@f(%1, %2)" in mod.astext()
    assert mod["f"].checked_type == relay.FuncType([ti32, tf32], tf32)


def test_incomplete_call():
    tt = relay.scalar_type("int32")
    x = relay.var("x", tt)
    f_type = relay.FuncType([tt], tt)
    f = relay.var("f")
    func = relay.Function([x, f], relay.Call(f, [x]), tt)

    ft = infer_expr(func)
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

    hc = infer_expr(ho_call)
    expected = relay.scalar_type("int32")
    assert hc.checked_type == expected


def test_higher_order_return():
    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    id_func = relay.Function([x], x, a, [a])

    b = relay.TypeVar("b")
    nested_id = relay.Function([], id_func, relay.FuncType([b], b), [b])

    ft = infer_expr(nested_id)
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
    ft = infer_expr(top)
    assert ft.checked_type == expected


def test_tuple():
    tp = relay.TensorType((10,))
    x = relay.var("x", tp)
    res = relay.Tuple([x, x])
    assert infer_expr(res).checked_type == relay.TupleType([tp, tp])


def test_ref():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    r = relay.RefCreate(x)
    st = relay.scalar_type("float32")
    assert infer_expr(r).checked_type == relay.RefType(st)
    g = relay.RefRead(r)
    assert infer_expr(g).checked_type == st
    w = relay.RefWrite(r, y)
    assert infer_expr(w).checked_type == relay.TupleType([])


def test_free_expr():
    x = relay.var("x", "float32")
    y = relay.add(x, x)
    yy = infer_expr(y)
    tvm.ir.assert_structural_equal(yy.args[0], x, map_free_vars=True)
    assert yy.checked_type == relay.scalar_type("float32")
    assert x.vid.same_as(yy.args[0].vid)


def test_type_args():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)

    # InferTypeLocal does not support populating the type_args field
    mod = infer_mod(IRModule.from_expr(z))
    mod = infer_mod(mod, annotate_spans=False)
    ty_args = mod["main"].body.type_args
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
    mod = infer_mod(mod)
    func_ty = mod["main"].checked_type

    assert func_ty == relay.FuncType([tt], tt)


def test_equal():
    i = relay.var("i", shape=[], dtype="int32")
    eq = op.equal(i, relay.const(0, dtype="int32"))
    func = relay.Function([i], eq)
    ft = infer_expr(func)
    expected = relay.FuncType([relay.scalar_type("int32")], relay.scalar_type("bool"))
    assert ft.checked_type == expected

    assert ft.checked_type == relay.FuncType(
        [relay.scalar_type("int32")], relay.scalar_type("bool")
    )


def test_constructor_type():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    a = relay.TypeVar("a")
    x = relay.Var("x", a)
    func = relay.Function([x], constructor(x), box(a), [a])
    mod["main"] = func
    mod = infer_mod(mod)
    func_ty = mod["main"].checked_type
    box = mod.get_global_type_var("box")
    expected = relay.FuncType([a], box(a), [a])
    assert func_ty == expected


def test_constructor_call():
    mod = tvm.IRModule()
    box, constructor = initialize_box_adt(mod)

    box_unit = constructor(relay.Tuple([]))
    box_constant = constructor(relay.const(0, "float32"))

    func = relay.Function([], relay.Tuple([box_unit, box_constant]))
    mod["main"] = func
    mod = infer_mod(mod)
    ret_type = mod["main"].checked_type.ret_type.fields
    # NB(@jroesch): when we annotate spans the ast fragments before
    # annotation the previous fragments will no longer be directly equal.
    box = mod.get_global_type_var("box")
    expected1 = box(relay.TupleType([]))
    expected2 = box(relay.TensorType((), "float32"))
    assert ret_type[0] == expected1
    assert ret_type[1] == expected2


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

    func = relay.Function([], match)
    mod["main"] = func
    mod = infer_mod(mod)
    actual = mod["main"].checked_type.ret_type
    assert actual == relay.TupleType([])


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

    mod["main"] = relay.Function([x], match)
    mod = infer_mod(mod)
    ft = mod["main"].checked_type
    assert ft == relay.FuncType([tt], relay.TupleType([]))


def test_let_polymorphism():
    id = relay.Var("id")
    xt = relay.TypeVar("xt")
    x = relay.Var("x", xt)
    body = relay.Tuple([id(relay.const(1)), id(relay.Tuple([]))])
    body = relay.Let(id, relay.Function([x], x, xt, [xt]), body)
    body = infer_expr(body)
    int32 = relay.TensorType((), "int32")
    tvm.ir.assert_structural_equal(body.checked_type, relay.TupleType([int32, relay.TupleType([])]))


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
    mod = tvm.relay.fromtext(code)
    mod = transform.InferType()(mod)
    tvm.ir.assert_structural_equal(mod["main"].body.type_args, [relay.TensorType((), "float32")])


def test_dynamic_function():
    dy_tt = relay.TensorType([relay.Any()], "float32")
    s_tt = relay.TensorType([10], "float32")
    x = relay.Var("x", dy_tt)
    f = relay.Function([x], x + x)
    y = relay.Var("y", s_tt)
    c = f(y)

    mod = tvm.IRModule()
    mod["main"] = relay.Function([y], c)
    mod = transform.InferType()(mod)
    assert mod["main"].params[0].checked_type == s_tt

    data = relay.var(
        "data", shape=(relay.Any(), relay.Any(), relay.Any(), relay.Any()), dtype="float32"
    )
    weigth = relay.const(np.full((16, 16, 3, 3), 0.25), dtype="float32")
    x = relay.nn.conv2d(data, weigth, kernel_size=(3, 3), channels=16, groups=2)
    mod = tvm.IRModule.from_expr(x)
    mod = transform.InferType()(mod)


def test_custom_op_infer():
    """Tests infer type for custom_op"""
    op_name = "custom_log"
    _op.register(op_name, r"code(cal log of a tensor.)code")
    _op.get(op_name).set_num_inputs(1)
    _op.get(op_name).add_argument("data_0", "Tensor", "The input data tensor.")
    # call default relation functions
    _op.get(op_name).add_type_rel("Identity")
    _op.get(op_name).set_support_level(1)
    _op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
    _op.register_stateful(op_name, False)

    def clog(x):
        return relay.Call(_op.get(op_name), [x])

    tp = relay.TensorType((10, 10), "float32")
    x = relay.var("x", tp)
    sb = relay.ScopeBuilder()
    t1 = sb.let("t1", clog(x))
    t2 = sb.let("t2", relay.add(t1, x))
    sb.ret(t2)
    f = relay.Function([x], sb.get())
    fchecked = infer_expr(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_custom_add_broadcast_op():
    """Tests infer type for broadcast custom_op"""
    op_name = "custom_broadcast_add"
    _op.register(op_name, r"code(Add two tensor with inner broadcasting.)code")
    _op.get(op_name).set_num_inputs(2)
    _op.get(op_name).add_argument("data_0", "Tensor", "The input data tensor.")
    _op.get(op_name).add_argument("data_1", "Tensor", "The input data tensor.")
    # call default relation functions
    _op.get(op_name).add_type_rel("Broadcast")
    _op.get(op_name).set_support_level(1)
    _op.register_stateful(op_name, False)

    def broadcast_add(x, y):
        return relay.Call(_op.get(op_name), [x, y])

    x = relay.var("x", shape=(10, 4))
    y = relay.var("y", shape=(5, 10, 1))
    z = broadcast_add(x, y)
    func = relay.Function([x, y], z)
    t1 = relay.TensorType((10, 4), "float32")
    t2 = relay.TensorType((5, 10, 1), "float32")
    t3 = relay.TensorType((5, 10, 4), "float32")
    expected_ty = relay.FuncType([t1, t2], t3)
    assert_has_type(func, expected_ty)


def test_custom_op_rel_infer():
    """Tests infer type for custom_op"""

    def custom_log1_rel(arg_types, attrs):
        assert len(arg_types) == 1, "type relation arg number mismatch!"
        if attrs:
            assert isinstance(attrs, DictAttrs)
        inputa_type = arg_types[0]
        return relay.TensorType(inputa_type.shape, inputa_type.dtype)

    op_name = "custom_log1"
    _op.register(op_name, r"code(cal log of a tensor.)code")
    _op.get(op_name).set_num_inputs(1)
    _op.get(op_name).add_argument("data_0", "Tensor", "The input data tensor.")
    _op.get(op_name).set_attrs_type_key("DictAttrs")
    # call customized relation functions
    _op.get(op_name).add_type_rel("custom_log1", custom_log1_rel)
    _op.get(op_name).set_support_level(1)
    _op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
    _op.register_stateful(op_name, False)

    def clog(x):
        return relay.Call(_op.get(op_name), [x])

    tp = relay.TensorType((10, 10), "float32")
    x = relay.var("x", tp)
    sb = relay.ScopeBuilder()
    t1 = sb.let("t1", clog(x))
    t2 = sb.let("t2", relay.add(t1, x))
    sb.ret(t2)
    f = relay.Function([x], sb.get())
    fchecked = infer_expr(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_custom_op_rel_infer_exception():
    """Tests infer type for custom_op"""

    def custom_log1_rel(arg_types, attrs):
        assert len(arg_types) == 2, "type relation arg number mismatch!"
        return None

    op_name = "custom_log2"
    _op.register(op_name, r"code(cal log of a tensor.)code")
    _op.get(op_name).set_num_inputs(1)
    _op.get(op_name).add_argument("data_0", "Tensor", "The input data tensor.")
    _op.get(op_name).set_attrs_type_key("DictAttrs")
    # call customized relation functions
    _op.get(op_name).add_type_rel("custom_log2", custom_log1_rel)
    _op.get(op_name).set_support_level(1)
    _op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
    _op.register_stateful(op_name, False)

    def clog(x):
        return relay.Call(_op.get(op_name), [x])

    tp = relay.TensorType((10, 10), "float32")
    x = relay.var("x", tp)
    sb = relay.ScopeBuilder()
    t1 = sb.let("t1", clog(x))
    t2 = sb.let("t2", relay.add(t1, x))
    sb.ret(t2)
    f = relay.Function([x], sb.get())
    with pytest.raises(AssertionError) as cm:
        fchecked = infer_expr(f)
        assert "type relation arg number mismatch" in str(cm.execption)


def test_repeat_register():
    op_name = "custom_log3"
    _op.register(op_name, r"code(cal log of a tensor.)code")
    with pytest.raises(tvm.error.TVMError) as cm:
        _op.register(op_name)
        assert "Operator custom_log3 is registered before" in str(cm.execption)


@pytest.mark.parametrize("relay_op", [relay.op.argmax, relay.op.argmin])
@pytest.mark.parametrize(
    "shape_dtype",
    [
        ("int32", T.int32),
        ("int64", T.int64),
    ],
    ids=["int32", "int64"],
)
def test_argreduce_infer_return_type(relay_op, shape_dtype):
    x_shape = (1, 1)
    broadcast_shape = [1, 1]
    (sdtype, conv) = shape_dtype

    x = relay.var("data", relay.TensorType(x_shape, "float32"))
    broadcast_to = relay.op.broadcast_to(x, relay.const(broadcast_shape, dtype=sdtype))
    argmax = relay_op(broadcast_to, axis=[1])

    f = relay.Function([x], argmax)
    assert_has_type(
        f,
        relay.FuncType(
            [relay.TensorType(broadcast_shape, "float32")],
            relay.TensorType([conv(1)], dtype=sdtype),
        ),
    )


if __name__ == "__main__":
    tvm.testing.main()
