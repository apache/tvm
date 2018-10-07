"""Test that type checker correcly computes types
   for expressions.
"""
import tvm
import numpy as np
from tvm.relay.ir_pass import infer_type
from tvm.relay.ir_builder import IRBuilder, func_type
from tvm.relay.ir_builder import scalar_type, convert, tensor_type
from tvm.relay.env import Environment
from tvm.relay.op import log, add, equal, subtract, concatenate
from tvm.relay.expr import Function

def assert_has_type(expr, typ, env=Environment({})):
    checked_expr = infer_type(env, expr)
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (
            checked_type, typ))


def assert_decl_has_type(env, name, typ):
    func = env[name]
    assert func.checked_type == typ


def test_monomorphic_let():
    "Program: let x = 1; return x"
    b = IRBuilder()
    x = b.let('x', 1.0, value_type=scalar_type('float64'))
    b.ret(x)

    prog, env = b.get()
    assert_has_type(prog, scalar_type('float64'))

def test_dual_op():
    """Program:
       fn (x : Tensor[f32, (10, 10)]) {
         let t1 = log(x);
         let t2 = add(t1, x);
         return t1;
       }
    """
    b = IRBuilder()
    with b.function(('x', tensor_type(10, 10))) as func:
        x, = func.param_ids()
        t1 = b.let('t1', log(x))
        t2 = b.let('t2', add(t1, x))
        b.ret(t2)

    assert_has_type(func.to_func(),
                    func_type([tensor_type(10, 10)], tensor_type(10, 10)))


def test_decl():
    """Program:
       def f(x : Tensor[f32, (10, 10)]) {
           let lx = log(x);
           return lx;
       }
    """
    b = IRBuilder()
    x = b.param('x')
    with b.decl('f', x):
        lx = b.let('lx', log(x))
        b.ret(lx)
    _, env = b.get()
    assert_decl_has_type(env, 'f', func_type(['float32'], 'float32'))


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
       f(2, 10000);
    """
    b = IRBuilder()
    f = b.global_var('f')
    n = b.param('n', ty='int32')
    data = b.param('data', ty='float32')
    with b.decl(f, n, data):
        with b.if_scope(equal(n, convert(0))):
            b.ret(data)
        with b.else_scope():
            b.ret(f(subtract(n, convert(1)), log(data)))
    b.ret(f(convert(2.0), convert(10000.0)))
    assert_decl_has_type(b.env, 'f', func_type(
        ['int32', 'float32'], 'float32'))
    # TODO(@jroesch): need evaluator or new runtime
    # to execute this.

def test_concat():
    """
    Program:
        def try_concat2(x: Float(3, 2), y: Float(2, 2)) -> Float(5, 2) {
            return concatenate((x, y), axis=0);
        }
    """
    ib = IRBuilder()
    try_concat2 = ib.global_var('try_concat2')
    x = ib.param('x', ty=tensor_type(3, 2))
    y = ib.param('y', ty=tensor_type(2, 2))
    with ib.decl(try_concat2, x, y):
        ib.ret(concatenate((x, y), axis=0))
    fn_ty = func_type([tensor_type(3, 2), tensor_type(2, 2)], tensor_type(5, 2))
    assert_decl_has_type(ib.env, try_concat2, fn_ty)

if __name__ == "__main__":
    test_dual_op()
    test_recursion()
    test_monomorphic_let()
    test_decl()
    test_recursion()
    test_concat()
