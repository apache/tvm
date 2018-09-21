"""How to use Relay to implement a simple two-operator fusion pass.
==================================
**Author**: `Jared Roesch <https://homes.cs.washington.edu/~jroesch/>`_

In this tutorial, we will demonstrate how to write a fusion pass for
the Relay IR. We demonstrate many Relay features including defining a
new operator, a program transform, the NNVM compatibility layer,
and executing the original and transformed programs on the Relay
evaluator and TVM runtime system.
"""

################################################################
# Introduction
# -------------------------
#
# In this tutorial, we will demonstrate how to write a fusion pass for
# the Relay IR. We demonstrate many Relay features including defining a
# new operator, a program transform, the NNVM compatibility layer,
# and executing the original and transformed programs on the Relay
# evaluator and TVM runtime system.

from typing import Any, Dict

import numpy as np
import tvm
import topi

from relay import ir, make as mk
from relay.ir import OperatorId
from relay.opt import ItemVisitor, ExprVisitor
from relay.frontend.nnvm import Variable, symbol
from relay.frontend.nnvm import compiler
from relay.frontend.global_env import get_env
from relay.operators.register import func_ty_to_placeholders, register_op
from relay.eval import defn_to_pyfunc
from relay.tyck import check_expr

class ExprAtVisitor(ExprVisitor):
    """A demo visitor which adds a new traversal strategy."""
    expr_map: Dict[ir.LocalId, ir.Expr]

    def __init__(self):
        self.expr_map = {}

    def expr_at(self,id: ir.LocalId) -> ir.Expr:
        try:
            return self.expr_map[id]
        except KeyError:
            return id

    def visit_let(self, let: ir.Let) -> ir.Expr:
        self.expr_map[let.id] = let.value
        return super().visit_let(let)

# let x = 1 + 1;
# ... x will map to 1 + 1

class FuseTwo(ExprAtVisitor):
    """Rewrite b(a(x, y), z) into ab(x, y, z). """
    def __init__(self, a: OperatorId, b: OperatorId, a_and_b: OperatorId) -> None:
        self.a = a
        self.b = b
        self.a_and_b = a_and_b
        super().__init__()

    def visit_call(self, call: ir.Call) -> ir.Expr:
        func = call.fn
        if func == self.b:
            assert len(call.args) == 2 # An assumption of this fusion code.
            arg0 = self.expr_at(call.args[0])
            arg1 = self.expr_at(call.args[1])
            if isinstance(arg0, ir.Call) and arg0.fn == self.a:
                new_call = mk.Call(self.a_and_b, arg0.args[:] + [arg1])
            elif isinstance(arg1, ir.Call) and arg1.fn == self.a:
                new_call = mk.Call(self.a_and_b, arg1.args[:] + [arg0])
            else:
                new_call = super().visit_call(call)

            return new_call
        else:
            return super().visit_call(call)

def fma_compile(op_name: str, func_ty: ir.Type, attrs: ir.Attributes=None) -> Any:
    Inputs, ret_ty = func_ty_to_placeholders(func_ty)
    x, y, z  = Inputs
    Output = topi.multiply(topi.add(x, y), z)
    # this is not a python function call, but builds an AST
    schedule = tvm.create_schedule(Output.op)
    return [schedule, Inputs + [Output]]


def register_fma(env: Any) -> None:
    """Register TOPI's elementwise broadcast addition for the `+` operator."""
    shape = mk.TypeParam("s", ir.Kind.Shape)
    bt = mk.TypeParam("bt", ir.Kind.BaseType)
    in_out_type = mk.TensorType(bt, shape)
    fma_type = mk.TypeQuantifier(bt, mk.TypeQuantifier(shape, mk.TypeArrow([in_out_type, in_out_type, in_out_type], in_out_type)))
    # forall (bt: BaseTYpe) (s : Shape), Tensor[bt, s] -> Tensor[bt, s] -> Tensor[bt, s]
    # TODO: no reverse mode
    register_op(env, 'fma', fma_type, compiler=fma_compile)

# Get the global environment for demo purposes.
env = get_env()

register_fma(env)

# A small helper which applies just our transform to the Relay expression.
def transform(e):
    fuse = FuseTwo(env.add_id(), env.mul_id(), env.operator_id('fma'))
    e = fuse.visit(e)
    # Now let's use the type checker to make sure we didn't make a mistake.
    check_expr(env, e)
    return e

# We will use NNVM frontend.
x = Variable('x')
y = Variable('y')
z = x * (x + y)

relay_func = compiler.to_relay(z)

print(f"Relay Function:\n{compiler.pp(relay_func)}")

xform_func = transform(relay_func)

print(f"Transformed Function:\n{compiler.pp(xform_func)}")

# Use the evaluator.
norm = defn_to_pyfunc(env, relay_func)
xform = defn_to_pyfunc(env, xform_func)

x = np.random.uniform(size=(10, 5, 10)).astype('float32')
y = np.random.uniform(size=(10, 5, 10)).astype('float32')

norm_out = norm(x, y).asnumpy()
xform_out = xform(x, y).asnumpy()

np.testing.assert_allclose(norm_out, xform_out)

# Use the TVM runtime.

