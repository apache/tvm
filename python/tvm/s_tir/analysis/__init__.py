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
"""Analysis utilities for Schedulable TensorIR (S-TIR)."""
# pylint: disable=invalid-name
from typing import Dict, List, Optional

from tvm.ir import IRModule
from tvm.tir.expr import Var
from tvm.tir.stmt import SBlock, BufferRegion

from tvm.tir import Buffer, Stmt
from tvm.tir.function import PrimFunc
from . import _ffi_api


def get_sblock_access_region(
    block: SBlock, buffer_var_map: Dict[Var, Buffer]
) -> List[List[BufferRegion]]:
    """Detect which regions of tensors in this block are read or written to.
       Regions are sorted by order of appearance in the AST.

    Parameters
    ----------
    block: tvm.tir.SBlock
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
    return _ffi_api.GetSBlockAccessRegion(block, buffer_var_map)  # type: ignore


def get_sblock_read_write_region(
    block: SBlock, buffer_var_map: Dict[Var, Buffer]
) -> List[List[BufferRegion]]:
    """Auto detect the block read/write region according to its body stmt.
       An opaque access will be counted as both a read and a write access

    Parameters
    ----------
    block: tvm.tir.SBlock
        The block in which we are detecting read/write regions.

    buffer_var_map : Dict[Var, Buffer]
        The outside buffers which may access the block. Mapping from buffer var to the buffer

    Returns
    -------
    result : List[List[BufferRegion]]
        An array only consisting of the read regions and write regions of the input block
    """
    return _ffi_api.GetSBlockReadWriteRegion(block, buffer_var_map)  # type: ignore


def detect_buffer_access_lca(func: PrimFunc) -> Dict[Buffer, Stmt]:
    """Detect the lowest common ancestor(LCA) of buffer access, including both high-level
    access (BufferLoad, BufferStore) and low-level access (BufferLoad, BufferStore and opaque
    access).
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


def find_anchor_sblock(mod: IRModule) -> Optional[SBlock]:
    """Find the "anchor block" of the given module.

    We define the anchor block to be the block with (1) an init statement and (2) having
    the biggest flops count. The latter condition is only used when there are multiple blocks
    with an init statement.

    For example, if the input module is conv2d + fused spatial blocks, conv2d is the anchor block.
    The input module may not contain more than one such block. For example, a module having
    two conv2d is not allowed as an input.

    However, a module created from winograd convolution has multiple blocks with an init statement
    (input transform, batched GEMM, and output transform). We use the second condition, the flops
    count, to determine that the batched GEMM block is the anchor block.

    Parameters
    ----------
    mod: tvm.ir.IRModule
        The input TIR module.
    Returns
    -------
    anchor_block: Optional[SBlock]
        The anchor block if found, None otherwise.
    """
    return _ffi_api.find_anchor_sblock(mod)  # type: ignore # pylint: disable=no-member
