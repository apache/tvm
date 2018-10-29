#pylint: disable=no-else-return
"""An interface to the Realy interpreter."""
from __future__ import absolute_import
import numpy as np
from .. import register_func, nd
from .base import NodeBase, register_relay_node
from . import _make
from . import _eval
from . import ir_pass
from .expr import Call, Constant, GlobalVar
from . import const


class Value(NodeBase):
    """Base class of all values.
    """

    @staticmethod
    @register_func("relay.from_scalar")
    def from_scalar(i, dtype=None):
        """Convert a Python scalar to a Realy scalar."""
        if dtype is None:
            if isinstance(i, int):
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


def apply_passes(expr, env=None):
    ck_expr = ir_pass.infer_type(expr, env=env)
    fused_expr = ir_pass.fuse_ops(env, ck_expr)
    return fused_expr


def evaluate(env, expr, *args):
    # assert len(args) == 0
    relay_args = []
    for arg in args:
        relay_args.append(_arg_to_ast(arg))

    # TODO: We need to move this optimization code into the optimizer/pass manager
    if isinstance(expr, GlobalVar):
        func = env[expr]
        func = apply_passes(func, env)
        env._add(expr, func, True)
        opt_expr = Call(expr, relay_args)
        # import pdb; pdb.set_trace()
        return _eval.evaluate(env, opt_expr)
    else:
        expr = Call(expr, relay_args)
        opt_expr = apply_passes(expr, env)
        return _eval.evaluate(env, opt_expr)
