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
"""Relax unary arithmetic operators."""
from ...expr import Expr
from ...utils import args_converter
from . import _ffi_api

###################### Arithmetic operators ######################


def sigmoid(x: Expr) -> Expr:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sigmoid(x)  # type: ignore


def sign(x: Expr) -> Expr:
    """Returns an indication of the sign of a number for each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sign(x)  # type: ignore


@args_converter.auto
def clip(x: Expr, min: Expr, max: Expr) -> Expr:
    """Clips tensor values to a specified min and max.

    Parameters
    ----------
    x : relax.Expr
        The input data

    min : relax.Expr
        The minimum value

    max : relax.Expr
        The maximum value

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.clip(x, min, max)  # type: ignore
