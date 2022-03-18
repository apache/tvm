# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Basic tensor operations."""
# pylint: disable=redefined-builtin, unused-argument
from tvm import target
from tvm.runtime import ndarray as _nd
from tvm.runtime import Device as _Device
from tvm.te.hybrid import script

from . import _make
from .dyn import _make as _dyn_make
from ..expr import Tuple, Expr, Constant
from . import op as reg


def _make_virtual_device(device):
    if isinstance(device, _Device):
        return target.VirtualDevice(device)
    if isinstance(device, str):
        return target.VirtualDevice(_nd.device(device))
    raise ValueError("expecting a Device or device name, but received a %s" % (type(device)))


# We create a wrapper function for each operator in the
# python side to call into the positional _make.OpName function.
#
# We make this decision so that we can:
# - Have declare python docstring for each function
# - Enable keyword arguments easily
# - Not put too much burden on FFI to support complicated features
#   like default value and keyword arguments


def log(data):
    """Compute elementwise log of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.log(data)


def log2(data):
    """Compute elementwise log to the base 2 of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.log2(data)


def log10(data):
    """Compute elementwise log to the base 10 of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.log10(data)


def tan(data):
    """Compute elementwise tan of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.tan(data)


def cos(data):
    """Compute elementwise cos of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.cos(data)


def cosh(data):
    """Compute elementwise cosh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.cosh(data)


def sin(data):
    """Compute elementwise sin of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sin(data)


def sinh(data):
    """Compute elementwise sinh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sinh(data)


def acos(data):
    """Compute elementwise acos of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.acos(data)


def acosh(data):
    """Compute elementwise acosh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.acosh(data)


def asin(data):
    """Compute elementwise asin of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.asin(data)


def asinh(data):
    """Compute elementwise asinh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.asinh(data)


def atan(data):
    """Compute elementwise atan of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.atan(data)


def atanh(data):
    """Compute elementwise atanh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.atanh(data)


def exp(data):
    """Compute elementwise exp of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.exp(data)


def erf(data):
    """Compute elementwise error function of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.erf(data)


def sqrt(data):
    """Compute elementwise sqrt of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sqrt(data)


def rsqrt(data):
    """Compute elementwise rsqrt of data.

    .. math::

      1/sqrt(x)

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.rsqrt(data)


def sigmoid(data):
    """Compute elementwise sigmoid of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sigmoid(data)


def floor(data):
    """Compute element-wise floor of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.floor(data)


def ceil(data):
    """Compute element-wise ceil of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ceil(data)


def trunc(data):
    """Compute element-wise trunc of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.trunc(data)


def round(data):
    """Compute element-wise round of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.round(data)


def abs(data):
    """Compute element-wise absolute of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.abs(data)


def sign(data):
    """Compute element-wise absolute of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sign(data)


def tanh(data):
    """Compute element-wise tanh of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.tanh(data)


def negative(data):
    """Compute element-wise negative of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.negative(data)


def logical_not(data):
    """Compute element-wise logical not of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_not(data)


def bitwise_not(data):
    """Compute element-wise bitwise not of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.bitwise_not(data)


def add(lhs, rhs):
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code:: python

      x = relay.Var("a") # shape is [2, 3]
      y = relay.Var("b") # shape is [2, 1]
      z = relay.add(x, y)  # result shape is [2, 3]
    """
    return _make.add(lhs, rhs)


def subtract(lhs, rhs):
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.multiply(lhs, rhs)


def divide(lhs, rhs):
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.divide(lhs, rhs)


def floor_divide(lhs, rhs):
    """Floor division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.floor_divide(lhs, rhs)


def trunc_divide(lhs, rhs):
    """Trunc division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.trunc_divide(lhs, rhs)


def power(lhs, rhs):
    """Power with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.power(lhs, rhs)


def mod(lhs, rhs):
    """Mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.mod(lhs, rhs)


def floor_mod(lhs, rhs):
    """Floor mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.floor_mod(lhs, rhs)


def trunc_mod(lhs, rhs):
    """Trunc mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.trunc_mod(lhs, rhs)


def logical_and(lhs, rhs):
    """logical AND with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_and(lhs, rhs)


def logical_or(lhs, rhs):
    """logical OR with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_or(lhs, rhs)


def logical_xor(lhs, rhs):
    """logical XOR with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.logical_xor(lhs, rhs)


def bitwise_and(lhs, rhs):
    """bitwise AND with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.bitwise_and(lhs, rhs)


def bitwise_or(lhs, rhs):
    """bitwise OR with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.bitwise_or(lhs, rhs)


def bitwise_xor(lhs, rhs):
    """bitwise XOR with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.bitwise_xor(lhs, rhs)


def equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs == rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs != rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.not_equal(lhs, rhs)


def less(lhs, rhs):
    """Broadcasted elementwise test for (lhs < rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less(lhs, rhs)


def less_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs <= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less_equal(lhs, rhs)


def greater(lhs, rhs):
    """Broadcasted elementwise test for (lhs > rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater(lhs, rhs)


def greater_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs >= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater_equal(lhs, rhs)


def maximum(lhs, rhs):
    """Maximum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Minimum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.minimum(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.right_shift(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.left_shift(lhs, rhs)


def zeros(shape, dtype):
    """Fill array with zeros.

    Parameters
    ----------
    shape : tuple of int or relay.Expr
        The shape of the target.

    dtype : data type
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    if isinstance(shape, Constant):
        shape = list(shape.data.numpy())
    if isinstance(shape, Expr):
        return _dyn_make.zeros(shape, dtype)
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(shape, (list, tuple)):
        shape = list(shape)
    return _make.zeros(shape, dtype)


def zeros_like(data):
    """Returns an array of zeros, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.zeros_like(data)


def ones(shape, dtype):
    """Fill array with ones.

    Parameters
    ----------
    shape : tuple of int or relay.Expr
        The shape of the target.

    dtype : data type
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    if isinstance(shape, Constant):
        shape = list(shape.data.numpy())
    if isinstance(shape, Expr):
        return _dyn_make.ones(shape, dtype)
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(shape, (list, tuple)):
        shape = list(shape)
    return _make.ones(shape, dtype)


def ones_like(data):
    """Returns an array of ones, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ones_like(data)


def clip(a, a_min, a_max):
    """Clip the elements in `a` between `a_min` and `a_max`.
    `a_min` and `a_max` are cast to `a`'s dtype.

    Parameters
    ----------
    a : relay.Expr
        The input tensor.
    a_min : float
        The clip minimum.
    a_max : float
        The clip maximum.

    Returns
    -------
    result : relay.Expr
        `a` with elements clipped between `a_min` and `a_max`.

    Examples
    --------
    .. code:: python

      x = relay.Constant(tvm.nd.array([0, 1, 5, 3, 4, 2]))
      relay.clip(x, 1., 4.)
      # [1, 1, 4, 3, 4, 2]
    """
    return _make.clip(a, a_min, a_max)


def fixed_point_multiply(data, multiplier, shift):
    """Fixed point multiplication between data and a fixed point
    constant expressed as multiplier * 2^(-shift), where multiplier
    is a Q-number with 31 fractional bits

    Parameters
    ----------
    data : relay.Expr
        The input tensor.
    multiplier : int
        The integer multiplier of the fixed point constant.
    shift : int
        The integer shift of the fixed point constant.

    Returns
    -------
    result : relay.Expr
        The output of the fixed point multiplication
    """
    return _make.fixed_point_multiply(data, multiplier, shift)


def concatenate(data, axis):
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors.
    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated tensor.
    """
    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    return _make.concatenate(Tuple(data), axis)


def einsum(data, equation):
    """Evaluates the Einstein summation convention on data

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors.
    equation : str
        The einsum expression string.

    Returns
    -------
    result : relay.Expr
        The output tensor from the einsum op.
    """
    data = list(data)
    if not data:
        raise ValueError("relay.einsum requires data to be non-empty.")
    if not isinstance(equation, str):
        raise ValueError("einsum `equation` must be a str")
    return _make.einsum(Tuple(data), equation)


def stack(data, axis):
    """Join a sequence of arrays along a new axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], relay.Expr)
        A list of tensors or a Relay expression that evaluates to a tuple of tensors.

    axis : int
        The axis in the result array along which the input arrays are stacked.

    Returns
    -------
    ret : relay.Expr
        The stacked tensor.
    """
    if not data:
        raise ValueError("relay.stack requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    if not isinstance(data, Expr):
        data = Tuple(list(data))
    return _make.stack(data, axis)


def copy(data):
    """Copy a tensor.

    Parameters
    ----------
    data : relay.Expr
        The tensor to be copied.

    Returns
    -------
    result: relay.Expr
        The copied result.
    """
    return _make.copy(data)


@script
def _copy_shape_func_tensor(data_shape):
    ndim = data_shape.shape[0]
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = data_shape[i]
    return out


@script
def _copy_shape_func_scalar(data_shape):
    out = output_tensor((), "int64")
    return out


@reg.register_shape_func("copy", False)
def copy_shape_func(attrs, inputs, _):
    """
    Shape function for copy op.
    """
    input = inputs[0]
    if len(input.shape) == 0:
        return [_copy_shape_func_scalar(input)]
    return [_copy_shape_func_tensor(input)]


def device_copy(data, src_device, dst_device):
    """Copy data from the source device to the destination device. This
    operator helps data transferring between difference devices for
    heterogeneous execution.

    Parameters
    ----------
    data : tvm.relay.Expr
        The tensor to be copied.

    src_device : Union[:py:class:`Device`, str]
        The source device where the data is copied from.

    dst_device : Union[:py:class:`Device`, str]
        The destination device where the data is copied to.

    Returns
    -------
    result : tvm.relay.Expr
        The copied result.
    """
    return _make.DeviceCopy(
        data, _make_virtual_device(src_device), _make_virtual_device(dst_device)
    )


def shape_of(data, dtype="int32"):
    """Get shape of a tensor.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.relay.Expr
        The shape tensor.
    """
    return _make.shape_of(data, dtype)


def ndarray_size(data, dtype="int32"):
    """Get number of elements of input tensor.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.relay.Expr
        The number of elements of input tensor.
    """
    return _make.ndarray_size(data, dtype)


def isnan(data):
    """Check nan in input data element-wise.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.isnan(data)


def isfinite(data):
    """Compute element-wise finiteness of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.isfinite(data)


def isinf(data):
    """Compute element-wise infiniteness of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.isinf(data)
