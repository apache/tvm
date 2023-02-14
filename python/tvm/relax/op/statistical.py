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
# pylint: disable=redefined-builtin
"""Statistical operators."""
from typing import List, Optional, Union

from . import _ffi_api
from ..expr import Expr


def max(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the max of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a max operation is performed.
        The default, axis=None, will compute the max of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.max(x, axis, keepdims)  # type: ignore


def mean(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the mean of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.mean(x, axis, keepdims)  # type: ignore


def min(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the min of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a min operation is performed.
        The default, axis=None, will compute the min of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.min(x, axis, keepdims)  # type: ignore


def prod(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the product of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a product is performed.
        The default, axis=None, will compute the product of all elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.prod(x, axis, keepdims)  # type: ignore


def std(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the standard deviation of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a standard deviation is performed.
        The default, axis=None, will compute the std of all elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.std(x, axis, keepdims)  # type: ignore


def sum(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the sum of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.sum(x, axis, keepdims)  # type: ignore


def variance(x: Expr, axis: Optional[Union[int, List[int]]] = None, keepdims: bool = False) -> Expr:
    """Computes the variance of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a variance operation is performed.
        The default, axis=None, will compute the variance of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.variance(x, axis, keepdims)  # type: ignore
