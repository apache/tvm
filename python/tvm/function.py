from __future__ import absolute_import as _abs
from numbers import Number as _Number, Integral as _Integral
from ._ctypes._api import _init_function_module
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


def convert(value):
    """Convert a value to expression."""
    if isinstance(value, _Number):
        return const(value)
    elif isinstance(value, (list, tuple)):
        value = [convert(x) for x in value]
        return _function_internal._Array(*value)
    else:
        return value


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
    ndim = len(shape)
    arg_names = fcompute.__code__.co_varnames
    if ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension")
    dim_var = [Var(x) for x in arg_names]
    body = fcompute(*dim_var)
    dom = [Range(0, x) for x in shape]
    op_node = _function_internal._ComputeOp(
        dom, name, dim_var, body)
    return _function_internal._Tensor(
        shape, name, body.dtype, op_node, 0)


def RDomain(dom):
    """Create a reduction domain given domain

    Parameters
    ----------
    dom : list of Range or list of pairs
       The reduction domain.

    Returns
    -------
    rdom : RDomain
       The result rdomain
    """
    if not isinstance(dom, (list, tuple)):
        dom = [dom]
    elif not isinstance(dom[0], (list, tuple)):
        dom = [dom]
    dnorm = []
    for x in dom:
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("need to list of ranges")
            dnorm.append(Range(x[0], x[1]))
        else:
            dnorm.append(x)
    dnorm = convert(dnorm)
    return _function_internal._RDomain(dnorm)


def sum(expr, rdom):
    """Create a sum expression over rdom

    Parameters
    ----------
    expr : Expr
        The source expression.

    rdom : RDomain
        The reduction domainx
    """
    assert isinstance(rdom, _collections.RDomain)
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
    assert isinstance(expr, _expr.Expr)
    assert isinstance(rdom, _collections.RDomain)
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
    assert isinstance(expr, _expr.Expr)
    assert isinstance(rdom, _collections.RDomain)
    x =  _make.Reduce("Max", expr, rdom)
    return x


def Schedule(tensor, scope="global"):
    return _function_internal._Schedule(tensor, scope)


def Split(dim, factor, over_rdom=False):
    return _function_internal._DimSplit(dim, factor, over_rdom)


_init_function_module("tvm")
