# pylint: disable=no-else-return
"""IR builder for the Relay IR.

Enables users to construct Relay programs with a Python API.
"""
from collections import OrderedDict
import numpy as np
import tvm
from .ty import Type, FuncType, TensorType
from .expr import Expr, Constant, Let, Var, Param, Function, If
from .env import Environment


def _convert_to_value(arg, ctxt=tvm.cpu(0)):
    # type: (Any, tvm.Context) -> tvm.nd.NDArray
    """Convert Python values into the appropriate types
       for the Relay evaluator.
    """
    if isinstance(arg, bool): # bool is subclass of int
        return tvm.nd.array(np.array(arg, dtype='uint8'), ctxt)
    elif isinstance(arg, int):
        return tvm.nd.array(np.array(arg, dtype='int32'), ctxt)
    elif isinstance(arg, float):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, np.ndarray):
        return tvm.nd.array(arg, ctxt)
    elif isinstance(arg, tvm.ndarray.NDArray):
        return arg
    else:
        # raise Exception(f"can't convert {type(arg)} to a Relay AST")
        raise Exception("unsupported argument type {0}".format(type(arg)))


def _convert_type(rtype):
    if isinstance(rtype, str):
        return scalar_type(rtype)
    elif isinstance(rtype, Type):
        return rtype
    else:
        raise Exception(
            "unsupported conversion to Relay type {0}".format(type(rtype)))


def convert(arg):
    # type: (Any) -> Expr
    """Convert some Python objects into a Relay AST fragment.

    Parameters
    ----------
    arg: Any
        The Python object

    Returns
    -------
    expr: relay.Expr
        The converted expression.
    """
    if isinstance(arg, Expr):
        return arg
    elif isinstance(arg, tuple):
        return relay.Tuple([convert(el) for el in arg])
    elif isinstance(arg, PartialFunc):
        return arg.to_func()
    else:
        value = _convert_to_value(arg)
        return Constant(value)


class WithScope(object):
    """A wrapper for builder methods which introduce scoping."""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        if value:
            raise value
        else:
            self._exit_cb()


class PartialFunc(object):
    """A wrapper around functions while they are being built.

      Used by the builder as a user is building up a function,
      allows Function nodes which contain partially initialized
      state.
    """

    def __init__(self, params, ret_type, body, type_params):
        self.params = params
        self.ret_type = ret_type
        self.body = body
        self.type_params = type_params

    def param_ids(self):
        return [p.var for p in self.params]

    def to_func(self):
        """Converts a PartialFunc into a :py:class:`~relay.Function`."""
        return Function(
            self.params,
            self.ret_type,
            self.body,
            self.type_params)

#pylint: disable=invalid-name


def _mk_let(bindings, ret_value):
    let_expr = ret_value
    for var, (value, ty) in reversed(list(bindings.items())):
        let_expr = Let(var, value, let_expr, ty)

    return let_expr


class IRBuilder(object):
    """The IRBuilder class.

    Enables users to build up a Relay environment and program.

    Examples
    --------

    Program:
       fn (x : Tensor[f32, (10, 10)]) {
         let t1 = log(x);
         let t2 = add(t1, x);
         return t1;
       }

    ..code-block: python
        b = IRBuilder()
        with b.function(('x', tensor_type(10, 10))) as func:
            x, = func.param_ids()
            t1 = b.let('t1', log(x))
            t2 = b.let('t2', add(t1, x))
            b.ret(t2)
    """

    def __init__(self):
        self.bindings = [OrderedDict({})]
        self.scopes = [OrderedDict({})]
        self.params = []
        self.ret_values = [None]
        self.env = Environment({})

    def enter_scope(self, params=None):
        if not params:
            params = []

        self.bindings.append(OrderedDict({}))
        self.scopes.append(OrderedDict({}))
        self.params.append(params)
        self.ret_values.append(None)

    def exit_scope(self):
        bindings = self.bindings.pop()
        scopes = self.scopes.pop()
        params = self.params.pop()
        ret_value = self.ret_values.pop()
        return bindings, scopes, params, ret_value

    #pylint: disable=invalid-name
    def bind(self, name, value, ty):
        lv = Var(name)
        self.scopes[-1][name] = lv
        self.bindings[-1][lv] = (value, ty)
        return lv

    def let(self, name, value, value_type=None):
        if isinstance(value, Param):
            value = value.var

        if not isinstance(value, Expr):
            value = convert(value)

        return self.bind(name, value, value_type)

    def _convert_params(self, raw_params):
        relay_params = []
        for raw_param in raw_params:
            if isinstance(raw_param, Param):
                var = raw_param.var
                param = raw_param
            elif isinstance(raw_param, tuple):
                var, ty = raw_param
                if isinstance(var, str):
                    var = Var(var)
                ty = _convert_type(ty)
                param = Param(var, ty)
            elif isinstance(param, str):
                var = Var(raw_param)
                ty = None
                param = Param(var, ty)
            else:
                raise Exception("unknown parameter type")

            self.scopes[-1][var.name_hint] = var
            relay_params.append(param)

        return relay_params

    def function(self, *params):
        """Construct a Relay function."""

        relay_params = self._convert_params(params)

        self.enter_scope()

        pfunc = PartialFunc(relay_params, None, None, [])

        def _on_exit():
            bindings, _, _, ret_value = self.exit_scope()
            body = _mk_let(bindings, ret_value)
            pfunc.body = body

        return WithScope(pfunc, _on_exit)

    def ret(self, x):
        """Set `x` to be the return value of the current function."""
        if not self.ret_values[-1]:
            self.ret_values[-1] = convert(x)
        else:
            raise Exception(
                "return value already set, a function can only have one return value")

    def if_scope(self, cond):
        """Construct the if branch an if expression with scoping."""
        self.enter_scope()

        def _on_exit():
            bindings, _, _, ret_value = self.exit_scope()
            assert self.ret_values[-1] is None
            true_branch = _mk_let(bindings, ret_value)
            self.ret_values[-1] = If(cond, true_branch, None)

        return WithScope(10, _on_exit)

    def else_scope(self):
        """Construct the else branch of an if expression with scoping."""
        self.enter_scope()

        def _on_exit():
            bindings, _, _, ret_value = self.exit_scope()
            partial_if = self.ret_values[-1]
            assert isinstance(
                partial_if, If) and partial_if.false_branch is None
            false_branch = _mk_let(bindings, ret_value)
            self.ret_values[-1] = If(
                partial_if.cond,
                partial_if.true_branch,
                false_branch)

        return WithScope(10, _on_exit)

    def param(self, name, ty=None):
        if not ty:
            ty = scalar_type('float32')
        else:
            ty = _convert_type(ty)

        return Param(Var(name), ty)

    def global_var(self, name):
        # type: (str) -> GlobalVar
        """Construct a global var with `name` as its name hint.

        Parameters
        ----------
        name: str
            The name of the global variable.

        Returns
        -------
        global_var: relay.GlobalVar
            The global variable with `name`.

        """
        return self.env.global_var(name)

    def decl(self, name, *params, **kwargs):
        """Create a global function.

        Parameters
        ----------
        name: str or GlobalVar
            The name of the function.
        params: params
            The parameters of the function.

        Returns
        -------
        with_scope: Scope for the function.
        """

        ret_type = kwargs.get('ret_type', None)

        self.enter_scope()

        def _on_exit():
            bindings, _, _, ret_value = self.exit_scope()
            exp = _mk_let(bindings, ret_value)
            self.env.add(name, Function(params, ret_type, exp))

        return WithScope(10, _on_exit)

    def get(self):
        """Get the full program.

        Returns
        ----------
        (prog, env) : (relay.Expr, relay.Environment)
            A pair of the partial program, and the modified environment.
        """
        bindings = self.bindings.pop()
        scope = self.scopes.pop()

        if self.bindings:
            raise Exception("IRBuilder: binding error")

        if self.scopes:
            raise Exception("IRBuilder: scoping error")

        if bindings and scope and not self.ret_values:
            raise Exception("IRBuilder: no return value set")

        return _mk_let(bindings, self.ret_values[-1]), self.env


def scalar_type(dtype):
    """Construct a Relay scalar type.

    Parameters
    ----------
    dtype: dtype
        The dtype of the scalar type.

    Returns:
    scalar_type: relay.Type
        The scalar type.
    """
    return TensorType(tvm.convert([]), dtype)


def tensor_type(*shape, **kwargs):
    """Construct a Relay Tensor type.

    Parameters
    ----------
    shape: list of tvm.Expr
        The shape of the Tensor type.
    dtype: dtype
        The dtype of the Tensor type.

    Returns
    -------
    tensor_type: relay.Type
        The resulting tensor types.
    """
    dtype = kwargs.get('dtype', 'float32')

    return TensorType(tvm.convert(shape), dtype)


def func_type(args, ret_type, type_params=None):
    """Construct a Relay function type.

    Parameters
    ----------
    args: list of relay.Type
        The argument types.

    ret_type: relay.Type
        The return type.

    type_params: list of relay.TypeParam
        The type parameters.

    Returns
    -------
    func_type: The function type.
    """
    if not type_params:
        type_params = []

    args = [_convert_type(arg) for arg in args]
    ret_type = _convert_type(ret_type)
    return FuncType(args, ret_type, type_params, [])
