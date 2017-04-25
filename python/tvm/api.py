"""Functions defined in TVM."""
# pylint: disable=invalid-name,unused-import,redefined-builtin
from __future__ import absolute_import as _abs

from numbers import Integral as _Integral

from ._ctypes._types import TVMType
from ._ctypes._node import register_node, NodeBase
from ._ctypes._node import convert_to_node as _convert_to_node
from ._ctypes._function import Function
from ._ctypes._function import _init_api, register_func, get_global_func
from ._ctypes._function import convert_to_tvm_func as _convert_tvm_func
from . import _api_internal
from . import _base
from . import make as _make
from . import expr as _expr
from . import tensor as _tensor
from . import schedule as _schedule
from . import collections as _collections

int32 = "int32"
float32 = "float32"
handle = "handle"


def min_value(dtype):
    return _api_internal._min_value(dtype)


def max_value(dtype):
    return _api_internal._max_value(dtype)


def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)


def convert(value):
    """Convert value to TVM node or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Node or Function
        Converted value in TVM
    """
    if isinstance(value, (Function, NodeBase)):
        return value

    if callable(value):
        return _convert_tvm_func(value)
    else:
        return _convert_to_node(value)


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


def var(name="tindex", dtype=int32):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type

    Returns
    -------
    var : Var
        The result symbolic variable.
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
    tensor: Tensor
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

    fcompute: lambda function of indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor
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


def scan(init, update, state_placeholder, inputs=None, name="scan"):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    init: Tensor or list of Tensor
        The initial condition of first init.shape[0] timestamps

    update: Tensor or list of Tensor
        The update rule of the scan given by symbolic tensor.

    state_placeholder: Tensor or list of Tensor
        The placeholder variables used by update.

    inputs: Tensor or list of Tensor, optional
        The list of inputs to the scan. This is not required, but can
        be useful for the compiler to detect scan body faster.

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    .. code-block:: python

      # The following code is equivalent to numpy.cumsum
      m = tvm.var("m")
      n = tvm.var("n")
      X = tvm.placeholder((m, n), name="X")
      s_state = tvm.placeholder((m, n))
      s_init = tvm.compute((1, n), lambda _, i: X[0, i])
      s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
      res = tvm.scan(s_init, s_update, s_state, X)
    """
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if isinstance(inputs, _tensor.Tensor):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")
    axis = _IterVar((init[0].shape[0], update[0].shape[0]), "%s.idx" % name, 3)
    op = _api_internal._ScanOp(name, axis, init, update, state_placeholder, inputs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res


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
            decl_buffer(t.shape, t.dtype, t.op.name))
        types.add(t.dtype)

    if dtype is None:
        if len(types) != 1:
            raise ValueError("Cannot infer output type, please provide dtype argument")
        infered_type = types.pop()
        dtype = [infered_type for _ in shape]

    for shp, dt in zip(shape, dtype):
        output_placeholders.append(decl_buffer(shp, dt, name))
    body = fcompute(input_placeholders, output_placeholders)
    if isinstance(body, _expr.Expr):
        body = _make.Evaluate(body)

    op = _api_internal._ExternOp(
        name, inputs, input_placeholders, output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return res[0] if len(res) == 1 else res


def decl_buffer(shape, dtype=None,
                name="buffer",
                data=None,
                strides=None,
                byte_offset=None,
                offset_alignment=0):
    """Decleare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    data : Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    byte_offset: Expr, optional
        The offset in bytes to data pointer.

    offset_alignment: int, optional
        The alignment of offset

    Returns
    -------
    buffer : Buffer
        The created buffer
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    dtype = float32 if dtype is None else dtype
    strides = () if strides is None else strides
    if data is None:
        data = var(name, "handle")

    return _api_internal._Buffer(
        name, data, shape, strides, dtype, byte_offset, offset_alignment)


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
    v = var(name)
    return _api_internal._IterVar(dom, v, iter_type, thread_tag)


def thread_axis(dom=None, tag='', name=''):
    """Create a new IterVar to represent thread index.

    Parameters
    ----------
    dom : Range or str
        The domain of iteration
        When str is passed, dom is set to None and str is used as tag

    tag : str, optional
        The thread tag

    name : str, optional
        The name of the var.

    Returns
    -------
    axis : IterVar
        The thread itervar.
    """
    if isinstance(dom, _base.string_types):
        tag, dom = dom, None
    if len(tag) == 0:
        raise ValueError("tag must be given as Positional or keyword argument")
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

    Returns
    -------
    axis : IterVar
        An iteration variable representing the value.
    """
    return _IterVar(dom, name, 2)


class comm_reducer(object):
    def __init__(self, fcombine, fidentity):
        self.fcombine = fcombine
        self.fidentity = fidentity
        code = fcombine.__code__
        assert fcombine.__code__.co_argcount == 2
        self.arg_vars = [var(name) for name in code.co_varnames]
        result = fcombine(*[v for v in self.arg_vars])
        self.result = convert(result)

    def __call__(self, expr, axis, where=None):
        id_elem = self.fidentity(expr.dtype)
        assert isinstance(id_elem, _expr.Expr)
        reducer = _make.CommReducer(self.arg_vars, self.result, id_elem)
        return reducer(expr, axis, where)


_init_api("tvm.api")
sum = comm_reducer(lambda x, y: x+y, lambda t: const(0, dtype=t))
min = comm_reducer(lambda lhs, rhs: _make.Min(lhs, rhs), lambda t: max_value(t))
max = comm_reducer(lambda lhs, rhs: _make.Max(lhs, rhs), lambda t: min_value(t))
