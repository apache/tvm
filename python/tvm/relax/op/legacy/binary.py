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
# pylint: disable=redefined-builtin, invalid-name
"""Relax binary arithmetic and comparison operators."""
from ...expr import Expr
from . import _ffi_api

###################### Comparison operators ######################


def maximum(x1: Expr, x2: Expr) -> Expr:
    """Element-wise maximum
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.maximum(x1, x2)


def minimum(x1: Expr, x2: Expr) -> Expr:
    """Element-wise minimum
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.minimum(x1, x2)
