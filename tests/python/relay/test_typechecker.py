"""Test that type checker correcly computes types
   for expressions.
"""
import tvm.relay.make as mk
from tvm.relay.ir_builder import IRBuilder, float_type

def test_monomorphic_let():
    b = IRBuilder()
    # Program: let x = 1; x
    x = b.let('x', 1, value_type=float_type())
    b.ret(x)

    prog = b.get()
    e = check_expr(prog)
    e.get_type()


