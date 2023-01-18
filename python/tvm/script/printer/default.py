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
"""The printer configuration"""
from typing_extensions import Literal

from . import _ffi_api


def ir_prefix(  # pylint: disable=invalid-name
    ir: Literal["ir", "tir"],
    prefix: str,
) -> None:
    """Set the prefix for the IR. If not set, the prefix for "tvm.ir" is "I", and for "tir" is "T.

    Parameters
    ----------
    ir : str
        The IR type, either "ir" or "tir".

    prefix : str
        The prefix to use.
    """
    _ffi_api.DefaultIRPrefix(ir, prefix)  # type: ignore  # pylint: disable=no-member


def buffer_dtype(dtype: str) -> None:
    """Set the default dtype for buffer. If not set, it is "float32".

    Parameters
    ----------
    dtype : str
        The default dtype for buffer.
    """
    _ffi_api.DefaultBufferDtype(dtype)  # type: ignore  # pylint: disable=no-member


def int_dtype(dtype: str) -> None:
    """Set the default dtype for integers. If not set, it is "int32".

    Parameters
    ----------
    dtype : str
        The default dtype for buffer.
    """
    _ffi_api.DefaultBufferDtype(dtype)  # type: ignore  # pylint: disable=no-member


def float_dtype(dtype: str) -> None:
    """Set the default dtype for buffer. If not set, there is no default,
    which means every floating point numbers will be wrapped with its precise dtype.

    Parameters
    ----------
    dtype : str
        The default dtype for buffer.
    """
    _ffi_api.DefaultFloatDtype(dtype)  # type: ignore  # pylint: disable=no-member


def verbose_expr(verbose: bool) -> None:
    """Whether or not to verbose print expressions. If not, the definition of every variable in an
    expression will be printed as separate statements. Otherwise, the result will be a one-liner.

    Parameters
    ----------
    dtype : str
        The default dtype for buffer.
    """
    _ffi_api.VerboseExpr(verbose)  # type: ignore  # pylint: disable=no-member
