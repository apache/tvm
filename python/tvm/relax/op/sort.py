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
"""Sortings operators."""
from typing import List, Optional

from . import _ffi_api
from ..expr import Expr


def sort(x: Expr, axis: int = -1, descending: bool = False):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    x : relax.Expr
        The input tensor.

    axis : Optional[int]
        Axis along which to sort the input tensor.
        By default the last axis of the input is used.

    descending : Optional[bool]
        Whether to sort in descending order, the default is False

    Returns
    -------
    out : relax.Expr
        Sorted tensor.

    """
    return _ffi_api.sort(x, axis, descending)  # type: ignore
