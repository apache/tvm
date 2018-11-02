#pylint: disable=no-else-return
"""An interface to the Realy interpreter."""
from __future__ import absolute_import
import numpy as np
from .. import register_func, nd
from .base import NodeBase, register_relay_node
from . import build_module
from . import _make
from . import _interpreter
from . import ir_pass
from .env import Environment
from .expr import Call, Constant, GlobalVar, Function, const
from .scope_builder import ScopeBuilder
from .._ffi.base import integer_types
from ..contrib import graph_runtime as tvm_runtime
from .. import cpu

class Value(NodeBase):
    """Base class of all values.
    """

    @staticmethod
    @register_func("relay.from_scalar")
    def from_scalar(i, dtype=None):
        """Convert a Python scalar to a Relay scalar."""
        if dtype is None:
            if isinstance(i, integer_types):
                dtype = 'int32'
            elif isinstance(i, float):
                dtype = 'float32'
            elif isinstance(i, bool):
                dtype = 'uint8'
            else:
                raise Exception("unable to infer dtype {0}".format(type(i)))

        return TensorValue(nd.array(np.array(i, dtype=dtype)))


@register_relay_node
class TupleValue(Value):
    def __init__(self, *fields):
        self.__init_handle_by_constructor__(
            _make.TupleValue, fields)

    def __getitem__(self, field_no):
        return self.fields[field_no]


@register_relay_node
class Closure(Value):
    pass


@register_relay_node
class TensorValue(Value):
    """A Tensor value produced by the evaluator."""

    def __init__(self, data):
        """Allocate a new TensorValue and copy the data from `array` into
           the new array.
        """
        if isinstance(data, np.ndarray):
            data = nd.array(data)

        self.__init_handle_by_constructor__(
            _make.TensorValue, data)

    def as_ndarray(self):
        """Convert a Relay TensorValue into a tvm.ndarray."""
        return self.data

    def asnumpy(self):
        """Convert a Relay TensorValue into a numpy.ndarray."""
        return self.data.asnumpy()

    def __eq__(self, other):
        return self.data == other.data


def _arg_to_ast(arg):
    if isinstance(arg, TensorValue):
        return Constant(arg.data)
    elif isinstance(arg, np.ndarray):
        return Constant(nd.array(arg))
    elif isinstance(arg, Constant):
        return arg
    else:
        return const(arg)

class Executor(object):
    """An abstract interface for executing Relay programs."""

    def __init__(self, env=None):
        """
        Parameters
        ----------
        env: relay.Environment
            The environment.
        """
        if env is None:
            self.env = Environment({})
        else:
            self.env = env


    def optimize(self, expr):
        # TODO: We need to move this optimization code into the optimizer/pass manager
        ck_expr = ir_pass.infer_type(expr, env=self.env)
        fused_expr = ir_pass.fuse_ops(self.env, ck_expr)
        ck_fused = ir_pass.infer_type(fused_expr, env=self.env)
        return ck_fused

    def _make_executor(self, _):
        """
        Construct a Python function that implements the evaluation
        of expression.

        Parameters
        ----------
        expr: relay.Expr
            The Relay expression to execute.

        Returns
        -------
        executor: function
            A Python function which implements the behavior of `expr`.
        """
        raise Exception("abstract method: please implement me.")

    def evaluate(self, expr, params=None):
        """
        Evaluate a Relay expression on the interpreter.

        Parameters
        ----------
        expr: tvm.relay.Expr
            The expression to evaluate.
        """
        if params:
            scope_builder = ScopeBuilder()
            for key, value in params:
                scope_builder.let(key, value)
            scope_builder.ret(expr)
            expr = scope_builder.get()

        if isinstance(expr, Function):
            assert not ir_pass.free_vars(expr)

        return self._make_executor(expr)


class Interpreter(Executor):
    """
    A wrapper around the Relay interpreter, implements the excecutor interface.
    """
    def __init__(self, env=None):
        Executor.__init__(self, env)

    def _make_executor(self, expr):
        def _interp_wrapper(*args):
            relay_args = []
            for arg in args:
                relay_args.append(_arg_to_ast(arg))

            if isinstance(expr, GlobalVar):
                func = self.env[expr]
                func = self.optimize(func)
                self.env._add(expr, func, True)
                opt_expr = Call(expr, relay_args)
                return _interpreter.evaluate(self.env, opt_expr)
            else:
                call = Call(expr, relay_args)
                opt_expr = self.optimize(call)
                return _interpreter.evaluate(self.env, opt_expr)

        return _interp_wrapper


class GraphRuntime(Executor):
    """A wrapper around the TVM graph runtime, implements the Executor interface."""
    def __init__(self, env=None):
        Executor.__init__(self, env)

    def _make_executor(self, expr):
        def _graph_wrapper(*args):
            func = self.optimize(expr)
            graph_json, mod, params = build_module.build(func, env=self.env)
            assert params is None
            gmodule = tvm_runtime.create(graph_json, mod, cpu(0))
            # Create map of inputs.
            inputs = {}
            for i, arg in enumerate(args):
                inputs[func.params[i].name_hint] = arg
            # Set the inputs here.
            gmodule.set_input(**inputs)
            # Run the module, and fetch the output.
            gmodule.run()
            return gmodule.get_output(0)

        return _graph_wrapper

def create_executor(mode='debug', env=None):
    if mode == 'debug':
        return Interpreter(env)
    elif mode == 'graph':
        return GraphRuntime(env)
    else:
        raise Exception("unknown mode {0}".format(mode))
