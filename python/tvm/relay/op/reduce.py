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
"""Reduce operators."""
# pylint: disable=redefined-builtin

from ..expr import Tuple, TupleWrapper
from . import _make
from .tensor import exp, log, sqrt
from .transform import squeeze


def argmax(data, axis=None, keepdims=False, exclude=False, select_last_index=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    select_last_index : bool
        Whether to select the last index or the first index if the max element appears in
        multiple indices, default is False (first index).

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmax(data, axis, keepdims, exclude, select_last_index)


def argmin(data, axis=None, keepdims=False, exclude=False, select_last_index=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    select_last_index : bool
        Whether to select the last index or the first index if the min element appears in
        multiple indices, default is False (first index).

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmin(data, axis, keepdims, exclude, select_last_index)


def sum(data, axis=None, keepdims=False, exclude=False):
    """Computes the sum of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.sum(data, axis, keepdims, exclude)


def all(data, axis=None, keepdims=False, exclude=False):
    """Computes the logical AND of boolean array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input boolean tensor

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data = relay.Constant(tvm.nd.array([[[ True,  True,  True],
                                           [ True,  True,  True],
                                           [False,  True, False]],
                                          [[ True, False, False],
                                           [ True,  True, False],
                                           [False,  True,  True]]]))

        relay.all(data, axis=1)
        # [[False,  True, False],
        # [False, False, False]]

        relay.all(data, axis=0)
        # [[ True, False, False],
        # [ True,  True, False],
        # [False,  True, False]]

    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.all(data, axis, keepdims, exclude)


def any(data, axis=None, keepdims=False, exclude=False):
    """Computes the logical OR of boolean array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input boolean tensor

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data = relay.Constant(tvm.nd.array([[[ True,  True,  True],
                                            [ True,  True,  True],
                                            [False,  True, False]],
                                            [[ True, False, False],
                                            [ True,  True, False],
                                            [False,  True,  True]]]))

        relay.any(data, axis=1)
        # [[True, True, True],
        # [True,  True, True]]

        relay.any(data, axis=0)
        # [[ True, True, True],
        # [ True,  True, True],
        # [False,  True, True]]

    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.any(data, axis, keepdims, exclude)


def max(data, axis=None, keepdims=False, exclude=False):
    """Computes the max of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which the max operation is performed.
        The default, axis=None, will find the max element from all of the elements of the input
        array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.max(data, axis, keepdims, exclude)


def min(data, axis=None, keepdims=False, exclude=False):
    """Computes the min of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a minimum operation is performed.
        The default, axis=None, will find the minimum element from all
        of the elements of the input array. If axis is negative it counts from
        the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.min(data, axis, keepdims, exclude)


def mean(data, axis=None, keepdims=False, exclude=False):
    """Computes the mean of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.mean(data, axis, keepdims, exclude)


def variance(data, axis=None, keepdims=False, exclude=False, unbiased=False):
    """Computes the variance of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a variance operation is performed.
        The default, axis=None, will compute the variance of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    unbiased : bool
        If this is set to True, the unbiased estimation will be used.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    m = mean(data, axis, True, exclude)
    return _make._variance(data, m, axis, keepdims, exclude, unbiased)


def std(data, axis=None, keepdims=False, exclude=False, unbiased=False):
    """Computes the standard deviation of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a standard deviation operation is performed.
        The default, axis=None, will compute the standard deviation of all elements in the
        input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    unbiased : bool
        If this is set to True, the unbiased estimation will be used.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    m = mean(data, axis, True, exclude)
    return sqrt(_make._variance(data, m, axis, keepdims, exclude, unbiased))


def mean_variance(data, axis=None, keepdims=False, exclude=False, unbiased=False):
    """Computes the mean and variance of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a mean and variance operation is performed.
        The default, axis=None, will compute the mean and variance of all elements in
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    unbiased : bool
        If this is set to True, the unbiased estimation will be used.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    m = mean(data, axis, True, exclude)
    var = _make._variance(data, m, axis, keepdims, exclude, unbiased)
    if not keepdims:
        m = squeeze(m, axis=axis)
    return TupleWrapper(Tuple((m, var)), 2)


def mean_std(data, axis=None, keepdims=False, exclude=False):
    """Computes the mean and standard deviation of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a mean and standard deviation operation is performed.
        The default, axis=None, will compute the mean and standard deviation of all elements in
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    m = mean(data, axis, True, exclude)
    s = sqrt(_make._variance(data, m, axis, keepdims, exclude, False))
    if not keepdims:
        m = squeeze(m)
    return TupleWrapper(Tuple((m, s)), 2)


def prod(data, axis=None, keepdims=False, exclude=False):
    """Computes the products of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a product is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.prod(data, axis, keepdims, exclude)


def logsumexp(data, axis=None, keepdims=False):
    """Compute the log of the sum of exponentials of input elements over given axes.

       This function is more numerically stable than log(sum(exp(input))).
       It avoids overflows caused by taking the exp of large inputs and underflows
       caused by taking the log of small inputs.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a standard deviation operation is performed.
        The default, axis=None, will compute the log of the sum of exponentials of all elements
        in the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """

    axis = [axis] if isinstance(axis, int) else axis
    max_x = max(data, axis, True)
    exp_x = exp(data - max_x)
    sum_x = sum(exp_x, axis, True)
    out_x = log(sum_x) + max_x
    if not keepdims:
        out_x = squeeze(out_x, axis)
    return out_x
