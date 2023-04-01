from typing import Sequence, Union

from tvm._ffi import register_func
from tvm.te import Tensor

from .. import reduction as legacy_reduction
from .. import transform as legacy_transform


@register_func("topi.array_api.argmax")
def argmax(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    return legacy_reduction.argmax(x, axis, keepdims)


@register_func("topi.array_api.argmin")
def argmin(
    x: Tensor,  # pylint: disable=invalid-name
    axis: Union[None, int, Sequence[int]] = None,
    keepdims: bool = False,
) -> Tensor:
    return legacy_reduction.argmin(x, axis, keepdims)


@register_func("topi.array_api.where")
def where(
    condition: Tensor,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    return legacy_transform.where(condition, x, y)
