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


def test_monomorphic_let():
    "Program: let x = 1; return x"
    sb = relay.ScopeBuilder()
    x = sb.let('x', relay.const(1.0, "float64"))
    sb.ret(x)
    xchecked = relay.ir_pass.infer_type(sb.get())
    assert xchecked.checked_type == relay.scalar_type("float64")


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
    pass
    # x = relay.var('x', shape=(10, 4))
    # y = relay.var('y', shape=(5, 10, 1))
    # z = x + y
    # func = relay.Function([x, y], z)
    # ttype = relay.TensorType((5, 5, 5), 'float32')
    # expected_ty = relay.FuncType([ttype, ttype], ttype)
    # assert_has_type(func.to_func(), expected_ty)


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

# This currently fails and should pass under the type system.
#
# This test is to illustrate problem with our weak form of
# unification.
#


def test_incomplete_call():
    sb = ScopeBuilder()
    x = relay.var('x', dtype='int32')
    f = relay.var('f')
    func = relay.Function([x, f], relay.Call(f, [x]))

    try:
        relay.ir_pass.infer_type(func)
        assert False
    except tvm.TVMError as e:
        assert True

def test_tuple():
    tp = relay.TensorType((10,))
    x = relay.var("x", tp)
    res = relay.Tuple([x, x])
    assert (relay.ir_pass.infer_type(res).checked_type ==
            relay.TupleType([tp, tp]))

def test_free_expr():
    x = relay.var("x", "float32")
    y = relay.add(x, x)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.scalar_type("float32")


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


def test_self_reference():
    """
    Program:
       def f(x) {
           return x;
       }
    """
    a = relay.TypeVar("a")
    x = relay.var("x", a)
    sb = relay.ScopeBuilder()

    f = relay.Function([x], x)
    fx = relay.Call(f, [x])
    assert relay.ir_pass.infer_type(x).checked_type == a
    assert relay.ir_pass.infer_type(f).checked_type == relay.FuncType([a], a)
    assert relay.ir_pass.infer_type(fx).checked_type == a


def test_global_var_cow_issue():
    mod = relay.Module({})
    gv = relay.GlobalVar("foo")
    x = relay.var('x', shape=[])
    func = relay.Function([x], relay.Call(gv, [x]),
                          relay.TensorType([], 'float32'))
    mod[gv] = func


def test_equal():
    i = relay.var('i', shape=[], dtype='int32')
    eq = op.equal(i, relay.const(0, dtype='int32'))
    # This should fail ....
    func = relay.Function([i], eq, ret_type=relay.TensorType([], 'int32'))


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
    test_self_reference()
    test_global_var_cow_issue()
