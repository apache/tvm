"""Test that type checker correcly computes types
   for expressions.
"""
import tvm
import numpy as np
from nnvm import graph
from tvm.relay.ir_pass import check_expr
from tvm.relay.ir_builder import IRBuilder, float_type, int_type
from tvm.relay.ir_builder import func_type, tensor_type, into_ast
from tvm.relay.env import Environment
from tvm.relay.ir_pass import Monomorphize
from tvm.relay.op import log, add, equal, subtract
from tvm.relay.expr import Function
from tvm.relay import to_tvm
from tvm.contrib import graph_runtime
import nnvm


def has_type(expr, typ, env=Environment({})):
    checked_expr = check_expr(env, expr)
    return checked_expr.checked_type() == typ


def decl_has_type(env, name, typ):
    func = env.lookup(name)
    return func.checked_type() == typ


def run(env, expr, inputs, shape):
    if not isinstance(expr, Function):
        expr = Function([], None, expr, [])

    env.add("main", expr)
    env.transform(Monomorphize.to_pass())
    main = env.lookup("main")
    graph, lib, _ = to_tvm.compile(main)
    # We use NNVM to load the graph right now because it populates node_row_ptr field.
    nnvm_graph = nnvm.graph.load_json(graph)
    module = graph_runtime.create(nnvm_graph, lib, tvm.cpu(0))
    module.set_input(None, None, **inputs)
    module.run()
    out_nd_array = tvm.nd.array(np.empty(shape, dtype='float32'))
    return module.get_output(0, out=out_nd_array)


def test_monomorphic_let():
    "Program: let x = 1; return x"
    b = IRBuilder()
    x = b.let('x', 1.0, value_type=float_type(64))
    b.ret(x)

    prog, env = b.get()
    assert has_type(prog, float_type(64))
    run(env, prog, [], float_type(64))


def test_single_op():
    "Program: fn (x : float32) { let t1 = f(x); t1 }"
    b = IRBuilder()
    with b.function(('x', float_type())) as func:
        x, = func.param_ids()
        t1 = b.let('t1', log(x))
        b.ret(t1)
    assert has_type(func.to_func(), func_type([float_type()], float_type()))


def test_add_op():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    b = IRBuilder()
    x = b.param('x', tensor_type(5, 5, 5))
    y = b.param('y', tensor_type(5, 5, 5))
    with b.function(x, y) as func:
        b.ret(add(x.var, y.var))
    b.ret(func)
    prog, env = b.get()
    ttype = tensor_type(5, 5, 5)
    expected_ty = func_type([ttype, ttype], ttype)
    assert has_type(func.to_func(), expected_ty)
    x_data = tvm.nd.array(np.random.rand(5, 5, 5).astype('float32'))
    y_data = tvm.nd.array(np.random.rand(5, 5, 5).astype('float32'))
    result = run(env, prog, {'x': x_data, 'y': y_data}, (5, 5, 5))
    np.testing.assert_allclose(
        x_data.asnumpy() + y_data.asnumpy(), result.asnumpy())

def test_add_broadcast_op():
    """
    Program:
        fn (x: Tensor[(10, 4), f32], y: Tensor[(5, 10, 1), f32]) -> Tensor[(5, 10, 4), f32] {
            return x + y;
        }
    """
    b = IRBuilder()
    x = b.param('x', tensor_type(10, 4))
    y = b.param('y', tensor_type(5, 10, 1))
    with b.function(x, y) as func:
        b.ret(add(x.var, y.var))
    b.ret(func)
    prog, env = b.get()
    ttype = tensor_type(5, 5, 5)
    expected_ty = func_type([ttype, ttype], ttype)
    assert has_type(func.to_func(), expected_ty)
    x_data = tvm.nd.array(np.random.rand(5, 5, 5).astype('float32'))
    y_data = tvm.nd.array(np.random.rand(5, 5, 5).astype('float32'))
    result = run(env, prog, {'x': x_data, 'y': y_data}, (5, 10, 4))
    np.testing.assert_allclose(
        x_data.asnumpy() + y_data.asnumpy(), result.asnumpy())

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
    assert has_type(func.to_func(), func_type([float_type()], float_type()))


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
    assert decl_has_type(env, 'f', func_type([float_type()], float_type()))


def test_recursion():
    """
    Program:
       def f(n: i32, data: f32) -> f32 {
          if (n == 0) {
              return f(n - 1, log(data));
          } else {
              return data;
          }
       }
       f(2, 10000);
    """
    b = IRBuilder()
    f = b.global_var('f')
    n = b.param('n', ty=int_type())
    data = b.param('data', ty=float_type())
    with b.decl(f, n, data):
        with b.if_scope(equal(n, into_ast(0.0))):
            b.ret(f(subtract(n, into_ast(1)), log(data)))
        with b.else_scope():
            b.ret(data)
    b.ret(f(into_ast(2.0), into_ast(10000.0)))
    assert decl_has_type(b.env, 'f', func_type(
        [int_type(), float_type()], float_type()))
    # TODO(@jroesch): need evaluator or new runtime
    # to execute this.

if __name__ == "__main__":
    # test_monomorphic_let()
    # test_single_op()
    test_add_op()
    test_add_broadcast_op()
    # test_dual_op()
    # test_decl()
    # test_recursion()
