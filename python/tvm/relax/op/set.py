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
# pylint: disable=import-outside-toplevel, redefined-builtin, unused-argument
"""Set operators."""
from typing import Optional, Union

import numpy as np  # type: ignore
import tvm

from . import _ffi_api
from ..expr import Expr, PrimValue


def unique(
    x: Expr,
    sorted: Union[bool, Expr] = True,
    return_index: Union[bool, Expr] = False,
    return_inverse: Union[bool, Expr] = False,
    return_counts: Union[bool, Expr] = False,
    axis: Optional[Union[int, Expr]] = None,
) -> Expr:
    """Find the unique elements in a given tensor.
    In addition, it optionally returns
    - the indices of the input tensor that give the unique values;
    - the indices of the unique tensor that reconstruct the input tensor;
    - the number of times each unique value comes up in the input tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor.

    sorted : Union[bool, Expr]
        Whether to sort the unique elements in ascending order before
        returning as output.

    return_index : Union[bool, Expr]
        Whether to return an additional tensor with indices for where elements in
        the unique tensor come from the original input.

    return_inverse : Union[bool, Expr]
        Whether to return an additional tensor with indices for where elements in
        the original input ended up in the returned unique list.

    return_counts : Union[bool, Expr]
        Whether to return an additional tensor with counts of each unique elements.

    axis : Optional
        The dimension to apply unique.
        If not specified, the unique values of the flattened input are returned.

    Returns
    -------
    ret : relax.Expr
        The created relax call with
    """

    if isinstance(sorted, bool):
        sorted = PrimValue(sorted)
    if isinstance(return_index, bool):
        return_index = PrimValue(return_index)
    if isinstance(return_inverse, bool):
        return_inverse = PrimValue(return_inverse)
    if isinstance(return_counts, bool):
        return_counts = PrimValue(return_counts)
    if axis is not None and isinstance(axis, int):
        axis = PrimValue(axis)
    return _ffi_api.unique(  # type: ignore
        x, sorted, return_index, return_inverse, return_counts, axis
    )


@tvm.register_func("relax.run.unique")
def numpy_unique(
    x: tvm.nd.array,
    sorted: int,
    return_index: int,
    return_inverse: int,
    return_counts: int,
    axis: Optional[int] = None,
) -> tvm.nd.array:
    """Returns the unique elements of the input tensor.

    Uses numpy.unique to compute unique elements.
    """
    import builtins

    # TODO(prakalp): add support for returning a tuple when return_inverse or return_counts is True
    if bool(return_index) or bool(return_inverse) or bool(return_counts):
        raise NotImplementedError("missing support return_inverse or return_counts set to true")
    x_numpy = x.numpy()
    # TODO(prakalp): use torch.unique instead of numpy when torch is installed in ci.
    output_sorted_numpy, indices = np.unique(x_numpy, return_index=True, axis=axis)

    if sorted:
        return tvm.nd.array(output_sorted_numpy)
    output_numpy = np.take(x_numpy, builtins.sorted(indices), axis=axis)
    return tvm.nd.array(output_numpy)


def nonzero(x: Expr) -> Expr:
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor.

    Returns
    -------
    result : relax.Expr
        A 2-D tensor containing indices of non-zero elements.

    Note
    ----
    This function is equivalent to `onnx.nonzero`.

    Examples
    --------

    .. code-block:: python

        x = [[0, 1],
             [2, 0]]
        nonzero(x) = [[0, 1],
                      [1, 0]]

    """
    return _ffi_api.nonzero(x)  # type: ignore


@tvm.register_func("relax.run.nonzero")
def numpy_nonzero(x: tvm.nd.array) -> tvm.nd.array:
    np_result = np.atleast_1d(x.numpy()).nonzero()
    return tvm.nd.array(np.stack(np_result, axis=0))
