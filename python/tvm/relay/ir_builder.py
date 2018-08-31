from typing import Any
import numpy as np
import tvm
from .type import FuncType, TensorType
from .expr import Expr, Call, Constant, Let, LocalVar, Param, Function
from .env import Environment
from . import op as _op

class ExprBuilder():
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, *args):
        return ExprBuilder(Call(self.expr, list(args), None, None))

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

def into_ast(arg: Any, ctxt=tvm.cpu(0)) -> Expr:
    if isinstance(arg, tuple):
        raise Exception("..")
    else:
        value = convert(arg, ctxt)
        return ExprBuilder(Constant(value))

class WithScope(object):
    """Auxiliary scope  with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


class PartialFunc():
    def __init__(self, params, ret_type, body, type_params):
        self.params = params
        self.ret_type = ret_type
        self.body = body
        self.type_params = type_params

    def param_ids(self):
        return [p.var for p in self.params]

    def to_func(self):
        return Function(
            self.params,
            self.ret_type,
            self.body,
            self.type_params)


def _mk_let(bindings, ret_value):
    let_expr = ret_value
    for var, value in reversed(list(bindings.items())):
        let_expr = Let(var, value, let_expr, None)

    return let_expr


class IRBuilder():
    def __init__(self):
        self.bindings = [{}]
        self.scopes = [{}]
        self.params = []
        self.ret_value = None
        self.env = Environment({})


    def bind(self, name, type, value):
        lv = LocalVar(name)
        self.scopes[-1][name] = lv
        self.bindings[-1][lv] = value
        return lv


    def let(self, name, value, value_type=None):
        if isinstance(value, Param):
            value = value.var

        if not (isinstance(value, Expr) or isinstance(value, ExprBuilder)):
            value = into_ast(value)

        if isinstance(value, ExprBuilder):
            value = value.expr

        return self.bind(name, value_type, value)

    def function(self, *params):
        relay_params = []
        for name, ty in params:
            lv = LocalVar(name)
            self.scopes[-1][name] = lv
            relay_params.append(Param(lv, ty))

        # self.params.append(relay_params)

        pfunc = PartialFunc(relay_params, None, None, [])

        def _on_exit():
            bindings = self.bindings.pop()
            scope = self.scopes.pop()
            ret_value = self.ret_value
            body = _mk_let(bindings, ret_value)
            self.ret_value = None
            pfunc.body = body


        return WithScope(pfunc, _on_exit)


    def ret(self, x):
        if not self.ret_value:
            self.ret_value = x
        else:
            raise Exception(
                "return value already set, a function can only have one return value")

    def param(self, name, ty=None):
        if not ty:
            ty = float_type()
        
        return Param(LocalVar(name), ty)

    # def params(*args):
    #      i = 0
    #      while i < args.size():
    #          arg = args[i]
    #          if isinstance(arg, str):
                

    def decl(self, name: str, *params):
        decl_builder = IRBuilder()

        def _on_exit():
            exp, sub_env = decl_builder.get()
            self.env.add(name, Function(params, None, exp))
            self.env.merge(sub_env)

        return WithScope(decl_builder, _on_exit)


    def get(self):
        """Get the full program"""
        bindings = self.bindings.pop()
        scope = self.scopes.pop()

        if self.bindings:
            raise Exception("IRBuilder: binding error")

        if self.scopes:
            raise Exception("IRBuilder: scoping error")

        if bindings and scope and not self.ret_value:
            raise Exception("IRBuilder: no return value set")

        return _mk_let(bindings, self.ret_value), self.env

def bool_dtype():
    return 'uint1'

def int_dtype(bits=32):
    return f'int1{bits}'

def float_dtype(bits=32):
    return f'float{bits}'

def uint_dtype(bits=32):
    return f'fuint{bits}'
    
def int_type(bits=32, lanes=1):
    # TODO(@jroesch, @tqchen) How do we set lanes?
    return TensorType(tvm.convert([]), int_dtype(bits))

def uint_type(bits=32, lanes=1):
    return TensorType(tvm.convert([]), uint_dtype(bits))

def float_type(bits=32, lanes=1):
    return TensorType(tvm.convert([]), float_dtype(bits))

def bool_type(lanes=1):
    return TensorType(tvm.convert([]), bool_dtype(bits))

def tensor_type(*shape, dtype='float32'):
    return TensorType(tvm.convert(shape), dtype)

def func_type(args, ret_type, type_params=[], type_constraints=[]):
    return FuncType(args, ret_type, type_params, type_constraints)
