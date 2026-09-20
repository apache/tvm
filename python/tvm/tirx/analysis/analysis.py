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
"""Wrapping existing analysis utils."""

# pylint: disable=invalid-name

from tvm.ir import IRModule
from tvm.tirx.expr import Var
from tvm.tirx.stmt import Expr

from .. import Stmt
from ..function import PrimFunc
from . import _ffi_api


def verify_ssa(func: PrimFunc) -> bool:
    """Verify if the func is in SSA form.

    Parameters
    ----------
    func: tvm.tirx.PrimFunc
        The module to be verified.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.verify_ssa(func)  # type: ignore


def verify_memory(func: PrimFunc) -> bool:
    """Verify if func contains illegal host side direct memory access.

    Parameters
    ----------
    func: tvm.tirx.PrimFunc
        The module to be verified.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.verify_memory(func)  # type: ignore


def undefined_vars(node: Stmt | Expr, defs: list[Var] | None = None) -> list[Var]:
    """Find undefined vars in a TIR statement or expression.

    Parameters
    ----------
    node: Union[Stmt, Expr]
        The TIR statement or expression to be checked.

    defs: Optional[List[Var]]
        The vars that is defined

    Returns
    -------
    result : List[Var]
        The undefined vars.
    """
    defs = defs or []
    return _ffi_api.UndefinedVars(node, defs)  # type: ignore # pylint: disable=no-member


def verify_well_formed(obj: PrimFunc | IRModule, assert_mode: bool = True) -> bool:
    """Verify definitions and buffer-load types in ordinary TIRX.

    Use ``tvm.s_tir.analysis.verify_well_formed`` for schedulable blocks.

    Parameters
    ----------
    obj: Union[tvm.tirx.PrimFunc, tvm.ir.IRModule]
        The function or module to be verified.

    assert_mode: bool
        The indicator if it raises an error when the function is not well-formed.

    Returns
    -------
    result: bool
        Whether it is a well-formed TIR function.
    """
    return _ffi_api.VerifyWellFormed(obj, assert_mode)  # type: ignore # pylint: disable=no-member


def verify_tirx_well_formed(
    obj: PrimFunc | IRModule, assert_mode: bool = True, device_func: bool = False
) -> bool:
    """Verify if the given TIRX is well-formed.

    Parameters
    ----------
    obj: Union[tvm.tirx.PrimFunc, tvm.ir.IRModule]
        The function or module to be verified.

    assert_mode: bool
        The indicator if it raises an error when the function is not well-formed.

    device_func: bool
        The indicator if it is a device function.

    Returns
    -------
    result: bool
        Whether it is a well-formed TIRX function.
    """
    return _ffi_api.VerifyTIRxWellFormed(obj, assert_mode, device_func)  # type: ignore # pylint: disable=no-member
