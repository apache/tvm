# pylint: disable=protected-access, no-member, invalid-name, too-many-locals
# pylint: disable=redefined-builtin, undefined-variable, unused-import
"""Functions defined in TVM."""
from __future__ import absolute_import as _abs

from numbers import Integral as _Integral

from ._ctypes._types import TVMType
from ._ctypes._node import register_node, NodeBase
from ._ctypes._node import convert_to_node as _convert_to_node
from ._ctypes._function import Function
from ._ctypes._function import _init_api_functions, register_func, get_global_func
from ._ctypes._function import convert_to_tvm_func as _convert_tvm_func
from . import _api_internal
from . import make as _make
from . import expr as _expr
from . import tensor as _tensor
from . import collections as _collections

int32 = "int32"
float32 = "float32"
handle = "handle"

def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)


def load_json(json_str):
    """Load tvm object from json_str.

    Parameters
    ----------
    json_str : str
        The json string

    Returns
    -------
    node : Node
        The loaded tvm node.
    """
    return _api_internal._load_json(json_str)


def save_json(node):
    """Load tvm object as json string.

    Parameters
    ----------
    node : Node
        A TVM Node object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return _api_internal._save_json(node)


def Var(name="tindex", dtype=int32):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type
    """
    return _api_internal._Var(name, dtype)


def placeholder(shape, dtype=None, name="placeholder"):
    """Construct an empty tensor object.

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    dtype: str, optional
        The data type of the tensor

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: tensor.Tensor
        The created tensor
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    dtype = float32 if dtype is None else dtype
    return _api_internal._Placeholder(
        shape, dtype, name)


def compute(shape, fcompute, name="compute"):
    """Construct a new tensor by computing over the shape domain.

    The compute rule is result[axis] = fcompute(axis)

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    fcompute: lambda function of *indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: tensor.Tensor
        The created tensor
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    ndim = len(shape)
    code = fcompute.__code__

    if fcompute.__code__.co_argcount == 0:
        arg_names = ["i%d" % i for i in range(ndim)]
    else:
        arg_names = code.co_varnames[:code.co_argcount]

    if ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

    dim_var = [_IterVar((0, s), x, 0) for x, s in zip(arg_names, shape)]
    body = fcompute(*[v.var for v in dim_var])
    body = convert(body)
    op_node = _api_internal._ComputeOp(
        name, dim_var, body)
    return op_node.output(0)


def scan(init, update, state_placeholder, name="scan"):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    init: Tensor or list of Tensor
        The initial condition of first init.shape[0] timestamps

    update: Tensor or list of Tensor
        The update rule of the scan given by symbolic tensor.

    state_placeholder: Tensor or list of Tensor
        The placeholder variables used by update.

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    # The following code is equivalent to numpy.cumsum
    m = tvm.Var("m")
    n = tvm.Var("n")
    X = tvm.placeholder((m, n), name="X")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    res = tvm.scan(s_init, s_update, s_state)
    """
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")
    axis = _IterVar((init[0].shape[0], update[0].shape[0]), "%s.idx" % name, 3)
    op = _api_internal._ScanOp(name, axis, init, update, state_placeholder)
    res = [op.output(i) for i in range(len(update))]
    return (res[0] if len(res) == 1 else res)


def extern(shape, inputs, fcompute,
           name="extern", dtype=None):
    """Compute several tensor via extern function.

    Parameters
    ----------
    shape: Shape tuple or list of shapes.
        The shape of the outputs.

    inputs: list of Tensor
        The inputs

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement to do the computation.

    name: str, optional
        The name hint of the tensor

    dtype: str or list of str, optional
        The data types of outputs,
        by default dtype will be same as inputs.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.
    """
    if isinstance(shape[0], _expr.Expr):
        shape = [shape]
    input_placeholders = []
    output_placeholders = []
    types = set()
    for t in inputs:
        if not isinstance(t, _tensor.Tensor):
            raise ValueError("expect inputs to be tensor")
        input_placeholders.append(
            Buffer(t.shape, t.dtype, t.op.name))
        types.add(t.dtype)

    if dtype is None:
        if len(types) != 1:
            raise ValueError("Cannot infer output type, please provide dtype argument")
        infered_type = types.pop()
        dtype = [infered_type for _ in shape]

    for shp, dt in zip(shape, dtype):
        output_placeholders.append(Buffer(shp, dt, name))
    body = fcompute(input_placeholders, output_placeholders)
    if isinstance(body, _expr.Expr):
        body = _make.Evaluate(body)

    op = _api_internal._ExternOp(
        name, inputs, input_placeholders, output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return (res[0] if len(res) == 1 else res)


def call_packed(*args):
    """Build expression by call an external packed function

    Parameters
    ----------
    args : list
        Positional arguments.
    """
    args = convert(args)
    return _make.Call(
        int32, "tvm_call_packed", args, 4, None, 0)


def Buffer(shape, dtype=None,
           name="buffer",
           ptr=None,
           strides=None):
    """Create a new symbolic buffer

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    ptr : Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    Returns
    -------
    buffer : Buffer
        The created buffer
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    dtype = float32 if dtype is None else dtype
    strides = () if strides is None else strides
    if ptr is None:
        ptr = Var(name, "handle")

    return _api_internal._Buffer(
        name, ptr, shape, strides, dtype)


def _IterVar(dom, name, iter_type, thread_tag=''):
    """Internal function to create IterVar

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of iteration variable.

    iter_type : int
        The type of iteration.

    thread_tag : str
        The thread tag of the iteration variable.

    Returns
    -------
    iter_var : IterVar
       The result itervar
    """
    if dom is not None:
        if isinstance(dom, (list, tuple)):
            if len(dom) != 2:
                raise ValueError("need to list of ranges")
            dom = Range(dom[0], dom[1])

        if not isinstance(dom, _collections.Range):
            raise ValueError("dom need to be Range")
    name = name if name else 'iter'
    var = Var(name)
    return _api_internal._IterVar(dom, var, iter_type, thread_tag)


def thread_axis(dom, tag, name=''):
    """Create a new IterVar to represent thread index.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    tag : str
        The thread tag

    name : str, optional
        The name of the var.
    """
    name = name if name else tag
    return _IterVar(dom, name, 1, tag)


def reduce_axis(dom, name="rv"):
    """Create a new IterVar for reduction.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of the variable.
    """
    return _IterVar(dom, name, 2)


def sum(expr, axis):
    """Create a sum expression over axis

    Parameters
    ----------
    expr : Expr
        The source expression.

    axis : IterVar
        The reduction IterVar axis
    """
    axis = axis if isinstance(axis, list) else [axis]
    x = _make.Reduce("Add", expr, axis)
    return x


def min(lhs, rhs=None, axis=None):
    """Create a min expression.

    Parameters
    ----------
    lhs : Expr
        The left hand expression.

    rhs : Expr, optional
        The right hand expression.

    axis : IterVar, optional
        The reduction IterVar axis
    """
    if rhs and axis:
        raise ValueError("Can only take one argument, rhs or axis")
    if isinstance(rhs, (_collections.IterVar, list)):
        axis, rhs = rhs, axis
    if rhs:
        return _make.Min(lhs, rhs)
    axis = axis if isinstance(axis, list) else [axis]
    x = _make.Reduce("Min", expr, axis)
    return x


def max(lhs, rhs=None, axis=None):
    """Create a max expression.

    Parameters
    ----------
    lhs : Expr
        The left hand expression.

    rhs : Expr, optional
        The right hand expression.

    axis : IterVar, optional
        The reduction IterVar axis
    """
    if rhs and axis:
        raise ValueError("Can only take one argument, rhs or axis")
    if isinstance(rhs, (_collections.IterVar, list)):
        axis, rhs = rhs, axis
    if rhs:
        return _make.Max(lhs, rhs)
    axis = axis if isinstance(axis, list) else [axis]
    x = _make.Reduce("Max", expr, axis)
    return x


def Schedule(ops):
    """Create a schedule for list of ops

    Parameters
    ----------
    ops : list of Operations
        The source expression.
    """
    if not isinstance(ops, (list, _collections.Array)):
        ops = [ops]
    return _api_internal._Schedule(ops)


def convert(value):
    """Convert value to TVM node or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Node or function
        Converted value in TVM
    """
    if isinstance(value, (Function, NodeBase)):
        return value

    if callable(value):
        return _convert_tvm_func(value)
    else:
        return _convert_to_node(value)


_init_api_functions("tvm")
