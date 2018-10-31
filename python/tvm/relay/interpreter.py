#pylint: disable=no-else-return
"""An interface to the Realy interpreter."""
from __future__ import absolute_import
import numpy as np
from .. import register_func, nd
from .base import NodeBase, register_relay_node
from . import _make
from . import _interpreter
from . import ir_pass
from .env import Environment
from .expr import Call, Constant, GlobalVar
from . import const
from .scope_builder import ScopeBuilder
from .._ffi.base import integer_types
from . import graph_runtime
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


class Interpreter(object):
    def __init__(self, mode='debug', env=None):
        if env is None:
            self.env = Environment({})
        else:
            self.env = env

        self.mode = mode


    def optimize(self, expr):
        # TODO: We need to move this optimization code into the optimizer/pass manager
        ck_expr = ir_pass.infer_type(expr, env=self.env)
        fused_expr = ir_pass.fuse_ops(self.env, ck_expr)
        ck_fused = ir_pass.infer_type(fused_expr, env=self.env)
        return ck_fused


    def evaluate(self, expr, params=None):
        """
        Evaluate a Relay expression on the interpreter.

        Parameters
        ----------
        expr: tvm.relay.Expr
            The expression to evaluate.
        """
        if params:
            sb = ScopeBuilder()
            for key, value in params:
                sb.let(key, value)
            sb.ret(expr)
            expr = sb.get()

        if self.mode == 'debug':
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
        elif self.mode == 'graph':
            def _graph_wrapper(*args):
                func = self.optimize(expr)
                graph_json, mod, params = graph_runtime.build(self.env, func)
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
        else:
            raise Exception("unknown interpreter mode: {0}".format(self.mode))


