from __future__ import absolute_import as _abs
from numbers import Number as _Number, Integral as _Integral
from ._ctypes._api import _init_function_module, convert
from . import _function_internal
from . import make as _make
from . import expr as _expr
from . import collections as _collections

int32 = "int32"
float32 = "float32"

def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _function_internal._const(value, dtype)


def Var(name="tindex", dtype=int32):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type
    """
    return _function_internal._Var(name, dtype)


def placeholder(shape, dtype = None, name="TensorObj"):
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
    dtype = float32 if dtype is None else dtype
    return _function_internal._Tensor(
        shape, name, dtype, None, 0)


def compute(shape, fcompute, name="TensorCompute"):
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
    if isinstance(shape, _expr.Expr):
        shape = (shape, )

    ndim = len(shape)
    arg_names = fcompute.__code__.co_varnames
    if ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension")

    dim_var = [IterVar((0, s), x) for x, s in zip(arg_names, shape)]
    body = fcompute(*[v.var for v in dim_var])
    body = convert(body)
    op_node = _function_internal._ComputeOp(
        name, dim_var, body)
    return _function_internal._Tensor(
        shape, name, body.dtype, op_node, 0)


def IterVar(dom, name='iter', thread_tag=''):
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
    if isinstance(dom, (list, tuple)):
        if len(dom) != 2:
            raise ValueError("need to list of ranges")
        dom = Range(dom[0], dom[1])

    if not isinstance(dom, _collections.Range):
        raise ValueError("dom need to be Range")

    return _function_internal._IterVar(dom, name, thread_tag)


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
    x =  _make.Reduce("Add", expr, rdom)
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
    x =  _make.Reduce("Min", expr, rdom)
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
    x =  _make.Reduce("Max", expr, rdom)
    return x


def Schedule(tensor, scope="global"):
    return _function_internal._Schedule(tensor, scope)


_init_function_module("tvm")
