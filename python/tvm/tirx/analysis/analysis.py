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
from ..function import Function
from . import _ffi_api


def verify_memory(func: Function) -> bool:
    """Verify if func contains illegal host side direct memory access.

    Parameters
    ----------
    func: tvm.tirx.Function
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


def verify_well_formed(obj: Function | IRModule, assert_mode: bool = True) -> bool:
    """Verify definitions, buffer-load types, loop control, tile ops, and scope IDs.

    Use ``tvm.s_tir.analysis.verify_well_formed`` for schedulable blocks.

    Parameters
    ----------
    obj: Union[tvm.tirx.Function, tvm.ir.IRModule]
        The function or module to be verified.

    assert_mode: bool
        The indicator if it raises an error when the function is not well-formed.

    Returns
    -------
    result: bool
        Whether it is a well-formed TIR function.
    """
    return _ffi_api.VerifyWellFormed(obj, assert_mode)  # type: ignore # pylint: disable=no-member
