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
from typing import Dict, List

from tvm.tir.stmt import Block, BufferRegion
from tvm.tir.stmt import PrimExpr
from tvm.tir.expr import Var
from . import _ffi_api
from ..function import PrimFunc
from .. import Buffer, Stmt


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


def verify_gpu_code(func: PrimFunc, constraints: Dict[str, int]) -> None:
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
    return _ffi_api.verify_gpu_code(func, constraints)  # type: ignore


def get_block_access_region(
    block: Block, buffer_var_map: Dict[Var, Buffer]
) -> List[List[BufferRegion]]:
    """Detect which regions of tensors in this block are read or written to.
       Regions are sorted by order of appearance in the AST.

    Parameters
    ----------
    block: tvm.tir.Block
        The block in which we are detecting read/write regions.

    buffer_var_map : Dict[Var, Buffer]
        The outside buffers which may access the block. Mapping from buffer var to the buffer

    Returns
    -------
    result : List[List[BufferRegion]]
        Array of access regions. There are three arrays of BufferRegion:
            - first: read regions
            - second: write regions
            - third: opaque regions
    """
    return _ffi_api.GetBlockAccessRegion(block, buffer_var_map)  # type: ignore


def get_block_read_write_region(
    block: Block, buffer_var_map: Dict[Var, Buffer]
) -> List[List[BufferRegion]]:
    """Auto detect the block read/write region according to its body stmt.
       An opaque access will be counted as both a read and a write access

    Parameters
    ----------
    block: tvm.tir.Block
        The block in which we are detecting read/write regions.

    buffer_var_map : Dict[Var, Buffer]
        The outside buffers which may access the block. Mapping from buffer var to the buffer

    Returns
    -------
    result : List[List[BufferRegion]]
        An array only consisting of the read regions and write regions of the input block
    """
    return _ffi_api.GetBlockReadWriteRegion(block, buffer_var_map)  # type: ignore


def calculate_workspace_bytes(func: PrimFunc, workspace_byte_alignment: int) -> int:
    """Calculate the workspace size in bytes needed by the TIR allocates inside the TIR
    PrimFunc.

    Parameters
    ----------
    func: tvm.tir.PrimFunc
        The function to be detected.
    workspace_byte_alignment : int
        The byte alignment required for each tensor

    Returns
    -------
    result : int
        Workspace size in bytes.
    """
    return _ffi_api.calculate_workspace_bytes(func, workspace_byte_alignment)  # type: ignore


def detect_buffer_access_lca(func: PrimFunc) -> Dict[Buffer, Stmt]:
    """Detect the lowest common ancestor(LCA) of buffer access, including both high-level
    access(BufferLoad, BufferStore) and low-level access(Load, Store and opaque access).
    The LCA may be a For loop or a Block.

    Parameters
    ----------
    func: tvm.tir.PrimFunc
        The function to be detected.

    Returns
    -------
    result : Dict[Buffer, Stmt]
        Map from buffer to the LCA of all access to it.
    """
    return _ffi_api.detect_buffer_access_lca(func)  # type: ignore # pylint: disable=no-member
