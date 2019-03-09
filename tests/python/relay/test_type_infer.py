"""Test that type checker correcly computes types
   for expressions.
"""
import tvm
import numpy as np
from tvm.relay.ir_pass import infer_type
from tvm import relay
from tvm.relay import op
from tvm.relay.scope_builder import ScopeBuilder


def assert_has_type(expr, typ, mod=relay.module.Module({})):
    checked_expr = infer_type(expr, mod)
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (
            checked_type, typ))


# initializes simple ADT for tests
def initialize_box_adt(mod):
    box = relay.GlobalTypeVar('box')
    tv = relay.TypeVar('tv')
    constructor = relay.Constructor('constructor', [tv], box)
    data = relay.TypeData(box, [tv], [constructor])
    mod[box] = data
    return (box, constructor)


def test_monomorphic_let():
    "Program: let x = 1; return x"
    sb = relay.ScopeBuilder()
    x = sb.let('x', relay.const(1.0, "float64"))
    sb.ret(x)
    xchecked = relay.ir_pass.infer_type(sb.get())
    assert xchecked.checked_type == relay.scalar_type("float64" )


def test_single_op():
    "Program: fn (x : float32) { let t1 = f(x); t1 }"
    x = relay.var('x', shape=[])
    func = relay.Function([x], op.log(x))
    ttype = relay.TensorType([], dtype='float32')
    assert_has_type(func, relay.FuncType([ttype], ttype))


def test_add_broadcast_op():
    """
    Program:
        fn (x: Tensor[(10, 4), f32], y: Tensor[(5, 10, 1), f32]) -> Tensor[(5, 10, 4), f32] {
            return x + y;
        }
    """
    x = relay.var('x', shape=(10, 4))
    y = relay.var('y', shape=(5, 10, 1))
    z = x + y
    func = relay.Function([x, y], z)
    t1 = relay.TensorType((10, 4), 'float32')
    t2 = relay.TensorType((5, 10, 1), 'float32')
    t3 = relay.TensorType((5, 10, 4), 'float32')
    expected_ty = relay.FuncType([t1, t2], t3)
    assert_has_type(func, expected_ty)


def test_dual_op():
    """Program:
       fn (x : Tensor[f32, (10, 10)]) {
         let t1 = log(x);
         let t2 = add(t1, x);
         return t1;
       }
    """
    tp = relay.TensorType((10, 10), "float32")
    x = relay.var("x", tp)
    sb = relay.ScopeBuilder()
    t1 = sb.let("t1", relay.log(x))
    t2 = sb.let("t2", relay.add(t1, x))
    sb.ret(t2)
    f = relay.Function([x], sb.get())
    fchecked = relay.ir_pass.infer_type(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_decl():
    """Program:
       def f(x : Tensor[(10, 10), f32]) {
           return log(x);
       }
    """
    tp = relay.TensorType((10, 10))
    x = relay.var("x", tp)
    f = relay.Function([x], relay.log(x))
    fchecked = relay.ir_pass.infer_type(f)
    assert fchecked.checked_type == relay.FuncType([tp], tp)


def test_recursion():
    """
    Program:
       def f(n: i32, data: f32) -> f32 {
          if (n == 0) {
              return data;
          } else {
              return f(n - 1, log(data));
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
    mod = relay.Module()
    mod[f] = relay.Function([n, data], sb.get())
    assert "%3 = @f(%1, %2)" in mod.astext()
    assert mod[f].checked_type == relay.FuncType([ti32, tf32], tf32)


def test_incomplete_call():
    tt = relay.scalar_type('int32')
    x = relay.var('x', tt)
    f = relay.var('f')
    func = relay.Function([x, f], relay.Call(f, [x]), tt)

    ft = relay.ir_pass.infer_type(func)
    f_type = relay.FuncType([tt], tt)
    assert ft.checked_type == relay.FuncType([tt, f_type], tt)


def test_higher_order_argument():
    a = relay.TypeVar('a')
    x = relay.Var('x', a)
    id_func = relay.Function([x], x, a, [a])

    b = relay.TypeVar('b')
    f = relay.Var('f', relay.FuncType([b], b))
    y = relay.Var('y', b)
    ho_func = relay.Function([f, y], f(y), b, [b])

    # id func should be an acceptable argument to the higher-order
    # function even though id_func takes a type parameter
    ho_call = ho_func(id_func, relay.const(0, 'int32'))

    hc = relay.ir_pass.infer_type(ho_call)
    expected = relay.scalar_type('int32')
    assert hc.checked_type == expected


def test_higher_order_return():
    a = relay.TypeVar('a')
    x = relay.Var('x', a)
    id_func = relay.Function([x], x, a, [a])

    b = relay.TypeVar('b')
    nested_id = relay.Function([], id_func, relay.FuncType([b], b), [b])

    ft = relay.ir_pass.infer_type(nested_id)
    assert ft.checked_type == relay.FuncType([], relay.FuncType([b], b), [b])


def test_higher_order_nested():
    a = relay.TypeVar('a')
    x = relay.Var('x', a)
    id_func = relay.Function([x], x, a, [a])

    choice_t = relay.FuncType([], relay.scalar_type('bool'))
    f = relay.Var('f', choice_t)

    b = relay.TypeVar('b')
    z = relay.Var('z')
    top = relay.Function(
        [f],
        relay.If(f(), id_func, relay.Function([z], z)),
        relay.FuncType([b], b),
        [b])

    expected = relay.FuncType([choice_t], relay.FuncType([b], b), [b])
    ft = relay.ir_pass.infer_type(top)
    assert ft.checked_type == expected


def test_tuple():
    tp = relay.TensorType((10,))
    x = relay.var("x", tp)
    res = relay.Tuple([x, x])
    assert (relay.ir_pass.infer_type(res).checked_type ==
            relay.TupleType([tp, tp]))


def test_ref():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    r = relay.RefCreate(x)
    st = relay.scalar_type("float32")
    assert relay.ir_pass.infer_type(r).checked_type == relay.RefType(st)
    g = relay.RefRead(r)
    assert relay.ir_pass.infer_type(g).checked_type == st
    w = relay.RefWrite(r, y)
    assert relay.ir_pass.infer_type(w).checked_type == relay.TupleType([])


def test_free_expr():
    x = relay.var("x", "float32")
    y = relay.add(x, x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.scalar_type("float32")
    assert x.vid.same_as(yy.args[0].vid)


def test_type_args():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    ty_z = relay.ir_pass.infer_type(z)
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
    mod = relay.Module({})
    gv = relay.GlobalVar("foo")
    x = relay.var('x', shape=[])
    tt = relay.scalar_type('float32')

    func = relay.Function([x], relay.Call(gv, [x]), tt)
    mod[gv] = func

    ft = relay.ir_pass.infer_type(gv, mod)
    assert mod[ft].checked_type == relay.FuncType([tt], tt)


def test_equal():
    i = relay.var('i', shape=[], dtype='int32')
    eq = op.equal(i, relay.const(0, dtype='int32'))
    func = relay.Function([i], eq)
    ft = relay.ir_pass.infer_type(func)

    assert ft.checked_type == relay.FuncType([relay.scalar_type('int32')], relay.scalar_type('bool'))


def test_constructor_type():
    mod = relay.Module()
    box, constructor = initialize_box_adt(mod)

    a = relay.TypeVar('a')
    x = relay.Var('x', a)
    ct = relay.ir_pass.infer_type(
        relay.Function([x], constructor(x), box(a), [a]), mod)
    expected = relay.FuncType([a], box(a), [a])
    assert ct.checked_type == expected


def test_constructor_call():
    mod = relay.Module()
    box, constructor = initialize_box_adt(mod)

    box_unit = constructor(relay.Tuple([]))
    box_constant = constructor(relay.const(0, 'float32'))

    ut = relay.ir_pass.infer_type(box_unit, mod)
    ct = relay.ir_pass.infer_type(box_constant, mod)
    assert ut.checked_type == box(relay.TupleType([]))
    assert ct.checked_type == box(relay.TensorType((), 'float32'))


def test_adt_match():
    mod = relay.Module()
    box, constructor = initialize_box_adt(mod)

    v = relay.Var('v', relay.TensorType((), 'float32'))
    match = relay.Match(constructor(relay.const(0, 'float32')),
                        [relay.Clause(
                            relay.PatternConstructor(constructor,
                                                     [relay.PatternVar(v)]),
                            relay.Tuple([])),
                         # redundant but shouldn't matter to typechecking
                         relay.Clause(relay.PatternWildcard(),
                                      relay.Tuple([]))])

    mt = relay.ir_pass.infer_type(match, mod)
    assert mt.checked_type == relay.TupleType([])


def test_adt_match_type_annotations():
    mod = relay.Module()
    box, constructor = initialize_box_adt(mod)

    # the only type annotation is inside the match pattern var
    # but that should be enough info
    tt = relay.TensorType((2, 2), 'float32')
    x = relay.Var('x')
    mv = relay.Var('mv', tt)
    match = relay.Match(constructor(x),
                        [relay.Clause(
                            relay.PatternConstructor(constructor,
                                                     [relay.PatternVar(mv)]),
                                                     relay.Tuple([]))])

    func = relay.Function([x], match)
    ft = relay.ir_pass.infer_type(func, mod)
    assert ft.checked_type == relay.FuncType([tt], relay.TupleType([]))


if __name__ == "__main__":
    test_free_expr()
    test_dual_op()
    test_single_op()
    test_recursion()
    test_monomorphic_let()
    test_decl()
    test_recursion()
    test_tuple()
    test_incomplete_call()
    test_free_expr()
    test_type_args()
    test_global_var_recursion()
    test_equal()
    test_ref()
    test_constructor_type()
    test_constructor_call()
    test_adt_match()
