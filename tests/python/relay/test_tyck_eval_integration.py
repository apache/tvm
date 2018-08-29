"""Test that type checker correcly computes types
   for expressions.
"""
from tvm.relay.ir_pass import check_expr
from tvm.relay.ir_builder import IRBuilder, float_type, func_type, tensor_type
from tvm.relay.env import Environment
from tvm.relay.op import log, add

def has_type(expr, typ):
    env = Environment({})
    checked_expr = check_expr(env, expr)
    import pdb; pdb.set_trace()
    return checked_expr.checked_type() == typ

def test_monomorphic_let():
    "Program: let x = 1; return x"
    b = IRBuilder()
    x = b.let('x', 1.0, value_type=float_type(64))
    b.ret(x)

    prog = b.get()
    assert has_type(prog, float_type(64))

def test_single_op():
    "Program: fn (x : float32) { let t1 = f(x); t1 }"
    b = IRBuilder()
    with b.function(('x', float_type())) as func:
        x, = func.param_ids()
        t1 = b.let('t1', log(x))
        b.ret(t1)
    assert has_type(func.to_func(), func_type([float_type()], float_type()))

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

if __name__ == "__main__":
    # test_monomorphic_let()
    # test_single_op()
    test_dual_op()
