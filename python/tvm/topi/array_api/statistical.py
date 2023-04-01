from typing import Sequence, Union

from tvm._ffi import register_func
from tvm.ir import Array
from tvm.te import Tensor

from .. import reduction as legacy_reduction
from ..broadcast import divide as legacy_divide
from ..math import sqrt as legacy_sqrt


@register_func("topi.array_api.max")
def max(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return legacy_reduction.max(
        data=x,
        axis=axis,
        keepdims=keepdims,
    )


@register_func("topi.array_api.min")
def min(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return legacy_reduction.min(
        data=x,
        axis=axis,
        keepdims=keepdims,
    )


@register_func("topi.array_api.sum")
def sum(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return legacy_reduction.sum(
        data=x,
        axis=axis,
        keepdims=keepdims,
    )


@register_func("topi.array_api.prod")
def prod(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return legacy_reduction.prod(
        data=x,
        axis=axis,
        keepdims=keepdims,
    )


@register_func("topi.array_api.mean")
def mean(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return _te_mean(
        x=x,
        axis=_get_real_axis(len(x.shape), axis),
        keepdims=keepdims,
    )


@register_func("topi.array_api.variance")
def variance(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return _te_variance(
        x=x,
        axis=_get_real_axis(len(x.shape), axis),
        keepdims=keepdims,
    )


@register_func("topi.array_api.std")
def std(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is not None and not axis:
        return x
    return _te_std(
        x=x,
        axis=_get_real_axis(len(x.shape), axis),
        keepdims=keepdims,
    )


def _te_mean(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Sequence[int],
    keepdims: bool,
) -> Tensor:
    from tvm.tir import const  # pylint: disable=import-outside-toplevel

    shape_prod = const(1, "int64")
    for dim in map(int, axis):
        shape_prod = shape_prod * x.shape[dim]
    res_sum = legacy_reduction.sum(x, axis, keepdims)
    return legacy_divide(res_sum, shape_prod)


def _te_variance(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Sequence[int],
    keepdims: bool,
) -> Tensor:
    dev = x - _te_mean(x, axis, keepdims=True)
    return _te_mean(dev * dev, axis, keepdims)


def _te_std(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Sequence[int],
    keepdims: bool,
) -> Tensor:
    return legacy_sqrt(_te_variance(x, axis, keepdims))


def _get_real_axis(ndim, axis) -> Sequence[int]:
    if axis is None:
        return list(range(ndim))
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(axis, (list, tuple, Array))
    axis = [(ele % ndim + ndim) % ndim for ele in axis]
    axis = list(set(axis))
    axis.sort()
    return axis
