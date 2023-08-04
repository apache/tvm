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
"""Operators with mask."""
from . import _ffi_api
from .create import full_like
from ..expr import Expr


def masked_fill(x: Expr, mask: Expr, value: Expr):
    """Fill a tensor by a specified value in places defined by a mask.
    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.
    mask : relax.Expr
        The mask.
    value : relax.Expr
        The value to set in the input tensor.
    Returns
    -------
    result : relax.Expr
        The filled tensor.
    """
    values = full_like(x, value)  # type: ignore
    return _ffi_api.where(mask, values, x)  # type: ignore
