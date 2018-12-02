#pylint: disable=no-else-return
"""The Python interface to the Relay reference interpreter."""
from __future__ import absolute_import

import numpy as np

from . import _backend
from .. import _make, ir_pass
from ... import register_func, nd
from ..base import NodeBase, register_relay_node
from ..expr import Call, Constant, GlobalVar, Function, const
from ..scope_builder import ScopeBuilder

class Value(NodeBase):
    """Base class of all values.
    """
    @staticmethod
    @register_func("relay.from_scalar")
    def from_scalar(value, dtype=None):
        """Convert a Python scalar to a Relay scalar."""
        return TensorValue(const(value, dtype).data)


@register_relay_node
class TupleValue(Value):
    """A tuple value produced by the interpreter."""
    def __init__(self, *fields):
        self.__init_handle_by_constructor__(
            _make.TupleValue, fields)

    def __getitem__(self, field_no):
        return self.fields[field_no]


@register_relay_node
class Closure(Value):
    """A closure produced by the interpreter."""
    pass


@register_relay_node
class TensorValue(Value):
    """A Tensor value produced by the interpreter."""

    def __init__(self, data):
        """Allocate a new TensorValue and copy the data from `array` into
           the new array.
        """
        if isinstance(data, np.ndarray):
            data = nd.array(data)

        self.__init_handle_by_constructor__(
            _make.TensorValue, data)

    def asnumpy(self):
        """Convert a Relay TensorValue into a numpy.ndarray."""
        return self.data.asnumpy()

    def __eq__(self, other):
        return self.data == other.data


def _arg_to_ast(arg):
    if isinstance(arg, TensorValue):
        return Constant(arg.data.copyto(_nd.cpu(0)))
    elif isinstance(arg, np.ndarray):
        return Constant(nd.array(arg))
    elif isinstance(arg, Constant):
        return arg
    else:
        return const(arg)


class Executor(object):
    """An abstract interface for executing Relay programs."""
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
        executor: function,
            A Python function which implements the behavior of `expr`.
        """
        raise NotImplementedError()

    def evaluate(self, expr, binds=None):
        """
        Evaluate a Relay expression on the executor.

        Parameters
        ----------
        expr: tvm.relay.Expr
            The expression to evaluate.

        binds: Map[tvm.relay.Var, tvm.relay.Expr]
            Additional binding of free variable.

        Returns
        -------
        val : Union[function, Value]
            The evaluation result.
        """
        if binds:
            scope_builder = ScopeBuilder()
            for key, value in binds.items():
                scope_builder.let(key, _arg_to_ast(value))
            scope_builder.ret(expr)
            expr = scope_builder.get()

        if isinstance(expr, Function):
            assert not ir_pass.free_vars(expr)

        if isinstance(expr, (Function, GlobalVar)):
            return self._make_executor(expr)

        # normal expression evaluated by running a function.
        func = Function([], expr)
        return self._make_executor(func)()


class Interpreter(Executor):
    """
    Simple interpreter interface.

    Parameters
    ----------
    mod : tvm.relay.Module
        The module to support the execution.

    ctx : tvm.TVMContext
        The runtime context to run the code on.

    target : tvm.Target
        The target option to build the function.
    """
    def __init__(self, mod, ctx, target):
        self.mod = mod
        self.ctx = ctx
        self.target = target
        self._intrp = _backend.CreateInterpreter(mod, ctx, target)

    def optimize(self, expr):
        """Optimize an expr.

        Parameters
        ----------
        expr : Expr
            The expression to be optimized.

        Returns
        -------
        opt_expr : Expr
            The optimized expression.
        """
        # TODO: We need to move this optimization code into the optimizer/pass manager
        ck_expr = ir_pass.infer_type(expr, mod=self.mod)
        fused_expr = ir_pass.fuse_ops(ck_expr)
        ck_fused = ir_pass.infer_type(fused_expr, mod=self.mod)
        return ck_fused

    def _make_executor(self, expr):
        def _interp_wrapper(*args):
            relay_args = []
            for arg in args:
                relay_args.append(_arg_to_ast(arg))

            if isinstance(expr, GlobalVar):
                func = self.mod[expr]
                func = self.optimize(func)
                self.mod._add(expr, func, True)
                opt_expr = Call(expr, relay_args)
                return self._intrp(opt_expr)
            else:
                call = Call(expr, relay_args)
                opt_expr = self.optimize(call)
                return self._intrp(opt_expr)
        return _interp_wrapper
