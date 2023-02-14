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
from typing import Optional

import numpy as np  # type: ignore
import tvm

from . import _ffi_api
from ..expr import Expr


def unique(
    x: Expr,
    sorted: bool = True,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None,
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

    sorted : bool
        Whether to sort the unique elements in ascending order before
        returning as output.

    return_index : bool
        Whether to return an additional tensor with indices for where elements in
        the unique tensor come from the original input.

    return_inverse : bool
        Whether to return an additional tensor with indices for where elements in
        the original input ended up in the returned unique list.

    return_counts : bool
        Whether to return an additional tensor with counts of each unique elements.

    axis : Optional
        The dimension to apply unique.
        If not specified, the unique values of the flattened input are returned.

    Returns
    -------
    ret : relax.Expr
        The created relax call with
    """

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
    axis: Optional[int],
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
    output_sorted_numpy, indices = np.unique(x_numpy, return_index=True)
    if sorted:
        return tvm.nd.array(output_sorted_numpy)
    output_numpy = [x_numpy.flatten()[index] for index in builtins.sorted(indices, reverse=True)]
    return tvm.nd.array(output_numpy)
