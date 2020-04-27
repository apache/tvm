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

from . import _ffi_api


def expr_deep_equal(lhs, rhs):
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
    return _ffi_api.expr_deep_equal(lhs, rhs)


def verify_ssa(func):
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
    return _ffi_api.verify_ssa(func)


def verify_memory(func):
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
    return _ffi_api.verify_memory(func)


def verify_gpu_code(func, constraints):
    """Verify if module contains illegal host side direct memory access.

    Parameters
    ----------
    func: tvm.tir.PrimFunc
        The module to be verified.

    constraints : Dict[str, int]
        The attribute constraints.

    Returns
    -------
    result : bool
        The result of verification.
    """
    return _ffi_api.verify_gpu_code(func, constraints)
