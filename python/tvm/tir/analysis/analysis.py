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
from typing import List, Optional, Union

from tvm.ir import IRModule
from tvm.tir.expr import Var
from tvm.tir.stmt import PrimExpr

from .. import Stmt
from ..function import PrimFunc
from . import _ffi_api


def expr_deep_equal(lhs: PrimExpr, rhs: PrimExpr) -> bool:
    """Deeply compare two nested expressions.

    Parameters
    ----------
    lhs : PrimExpr
        The left operand.

    rhs : PrimExpr
        The right operand.

    Returns
    -------
    result : bool
        The comparison result

    Note
    ----

    This function does not remap variable bindings, it will not
    return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
    Use py:func:`tvm.ir.structural_equal` to handle structural variable remapping.

    Due to the restriction of not remapping variables, this function can run
    faster than StructuralEqual and can be used as a utility function during arithmetic
    simplifications.

    Always consider py:func:`tvm.ir.structural_equal` first, which handles
    the structural remapping.

    See Also
    --------
    tvm.ir.structural_equal
    """
    return _ffi_api.expr_deep_equal(lhs, rhs)  # type: ignore


def verify_ssa(func: PrimFunc) -> bool:
    """Verify if the func is in SSA form.

    Parameters
    ----------
    func: tvm.tir.PrimFunc
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
    func: tvm.tir.PrimFunc
        The module to be verified.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.verify_memory(func)  # type: ignore


def undefined_vars(node: Union[Stmt, PrimExpr], defs: Optional[List[Var]] = None) -> List[Var]:
    """Find undefined vars in a TIR statement or expression.

    Parameters
    ----------
    node: Union[Stmt, PrimExpr]
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


def verify_well_formed(obj: Union[PrimFunc, IRModule], assert_mode: bool = True) -> bool:
    """Verify if the given TIR is well-formed. The verification includes:
        - Check if expressions not contain vars that is defined outside the block.

    Parameters
    ----------
    obj: Union[tvm.tir.PrimFunc, tvm.ir.IRModule]
        The function or module to be verified.

    assert_mode: bool
        The indicator if it raises an error when the function is not well-formed.

    Returns
    -------
    result: bool
        Whether it is a well-formed TIR function.
    """
    return _ffi_api.VerifyWellFormed(obj, assert_mode)  # type: ignore # pylint: disable=no-member
