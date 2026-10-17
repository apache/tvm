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

from tvm_tirx.ir import Expr, IRModule, Stmt
from tvm_tirx.tirx.expr import Var

from ..function import Function
from . import _ffi_api


def verify_memory(func: Function) -> bool:
    """Verify if func contains illegal host side direct memory access.

    Parameters
    ----------
    func: tvm_tirx.tirx.Function
        The module to be verified.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.VerifyMemory(func)  # type: ignore


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


def check_well_formed(obj: Function | IRModule, assert_mode: bool = True) -> bool:
    """Check whether a TIRx function is well formed.

    Parameters
    ----------
    obj: Union[tvm_tirx.tirx.Function, tvm_tirx.ir.IRModule]
        The function or module to be verified.

    assert_mode: bool
        Whether to raise on invalid IR. Defaults to True.

    Returns
    -------
    result: bool
        True for valid IR; False for invalid IR when assert_mode is False.

    Raises
    ------
    RuntimeError
        If the IR is invalid and assert_mode is True.
    """
    return _ffi_api.CheckWellFormed(obj, assert_mode)  # type: ignore # pylint: disable=no-member
