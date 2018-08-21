from typing import Any
import numpy as np
import tvm
from . import type as ty
from . import expr
from . import make as mk


def convert(arg: Any, ctxt=tvm.cpu(0)) -> tvm.nd.NDArray:
    """Convert Python values into the appropriate types
       for the Relay evaluator.
    """
    if isinstance(arg, int):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, float):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, bool):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, np.ndarray):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, tvm.ndarray.NDArray):
        return arg
    else:
        # raise Exception(f"can't convert {type(arg)} to a Relay AST")
        raise Exception(f"unsupported argument type {type(arg)}")

def into_ast(arg: Any, ctxt=tvm.cpu(0)) -> expr.Expr:
    if isinstance(arg, tuple):
        raise Exception("..")
    else:
        value = convert(arg, ctxt)
        return mk.Constant(value)

class WithScope(object):
    """Auxiliary scope  with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()

def _mk_let(bindings, ret_value):
    let_expr = ret_value
    for var, value in reversed(list(bindings.items())):
        let_expr = mk.Let(var, value, let_expr, None)

    return let_expr

class IRBuilder():
    def __init__(self):
        self.bindings = [{}]
        self.scopes = [{}]
        self.ret_value = None

    def bind(self, name, type, value):
        lv = mk.LocalVar(name)
        self.scopes[-1][name] = lv
        self.bindings[-1][lv] = value
        return lv


    def let(self, name, value):
        if not isinstance(value, expr.Expr):
            value = into_ast(value)

        return self.bind(name, None, value)

    def function(self, params):
        def _on_exit():
            bindings = self.bindings.pop()
            scope = self.scopes.pop()
            import pdb
            pdb.set_trace()
        return WithScope(None, _on_exit)

    def ret(self, x):
        if not self.ret_value:
            self.ret_value = x
        else:
            raise Exception(
                "return value already set, a function can only have one return value")

    def get(self):
        """Get the full program"""
        bindings = self.bindings.pop()
        scope = self.scopes.pop()

        if self.bindings:
            raise Exception("...")
        if self.scopes:
            raise Exception("...")

        if not self.ret_value:
            raise Exception("...")

        return _mk_let(bindings, self.ret_value)

# def int_type():
#     return TensorType(IntType(32), ShapeSeq([]))

# def float_type():
#     return TensorType(FloatType(32), ShapeSeq([]))

# def bool_type():
#     return TensorType(BoolType(), ShapeSeq([]))

# def make_shape(dims):
#     return ShapeSeq([ShapeSingleton(dim) for dim in dims])


