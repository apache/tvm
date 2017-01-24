# pylint: disable=protected-access, no-member, invalid-name
# pylint: disable=redefined-builtin, undefined-variable, unused-import
"""Functions defined in TVM."""
from __future__ import absolute_import as _abs
from numbers import Integral as _Integral
from ._ctypes._api import _init_api_module, convert, register_func, get_global_func
from . import _api_internal
from . import make as _make
from . import expr as _expr
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
    arg_names = fcompute.__code__.co_varnames

    if fcompute.__code__.co_argcount == 0 and len(arg_names) == 1:
        arg_names = ["i%d" % i for i in range(ndim)]
    if ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

    dim_var = [IterVar((0, s), x) for x, s in zip(arg_names, shape)]
    body = fcompute(*[v.var for v in dim_var])
    body = convert(body)
    op_node = _api_internal._ComputeOp(
        name, dim_var, body)
    return _api_internal._Tensor(
        shape, body.dtype, op_node, 0)


def Buffer(shape, dtype=None,
           name="buffer",
           ptr=None,
           strides=None):
    """Create a new buffer

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


def IterVar(dom=None, name=None, thread_tag=''):
    """Create a iteration variable

    Parameters
    ----------
    dom : Range
       The domain of iteration.

    name : str
       The name of iteration variable.

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
    if name is None:
        name = thread_tag if thread_tag else name
    name = name if name else 'iter'
    return _api_internal._IterVar(dom, name, thread_tag)


def sum(expr, rdom):
    """Create a sum expression over rdom

    Parameters
    ----------
    expr : Expr
        The source expression.

    rdom : RDomain
        The reduction domainx
    """
    rdom = rdom if isinstance(rdom, list) else [rdom]
    x = _make.Reduce("Add", expr, rdom)
    return x


def min(expr, rdom):
    """Create a min expression over rdom

    Parameters
    ----------
    expr : Expr
        The source expression.

    rdom : RDomain
        The reduction domainx
    """
    rdom = rdom if isinstance(rdom, list) else [rdom]
    x = _make.Reduce("Min", expr, rdom)
    return x


def max(expr, rdom):
    """Create a min expression over rdom

    Parameters
    ----------
    expr : Expr
        The source expression.

    rdom : RDomain
        The reduction domainx
    """
    rdom = rdom if isinstance(rdom, list) else [rdom]
    x = _make.Reduce("Max", expr, rdom)
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


_init_api_module("tvm")
