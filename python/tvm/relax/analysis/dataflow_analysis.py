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
"""
Python bindings for the dataflow analysis framework
"""

from typing import List
import tvm
from tvm.ir.base import Node
from tvm.relax.expr import Expr, SeqExpr, Function, Var
from . import _ffi_api


@tvm._ffi.register_object("relax.analysis.BasicBlock")
class BasicBlock(Node):
    """Representation of a basic block on top of Relax's AST (SeqExprs)"""

    seq: SeqExpr
    args: List[Var]
    ret: Expr
    start_block_idx: int
    start_binding_idx: int
    end_block_idx: int
    end_binding_idx: int

    def __init__(
        self,
        seq: SeqExpr,
        args: List[Var],
        ret: Expr,
        start_block_idx: int,
        start_binding_idx: int,
        end_block_idx: int,
        end_binding_idx: int,
    ):
        """
        Create a basic block

        Parameters
        ----------
        seq: SeqExpr
            The SeqExpr that contains the basic block
            (in normal form, no basic block can span across SeqExprs)

        args: List[Var]
            The values passed into the block.
            The starting block of a function takes in the function args.
            Merge blocks (those after an If branch) take the variable
            the If expression is bound to.

        ret: Expr
            The expression corresponding to the final value produced by a block.
            For blocks ending in a branch, the final value is the branch condition.
            Otherwise, it is the `body` field of the SeqExpr.

        start_block_idx: int
            The index of the block in the SeqExpr's block list where the basic block starts

        start_binding_idx: int
            The index of the binding in the starting binding block where the basic block
            starts (convention: if the basic block is a merge point,
            use the index of the binding after the If node).
        """
        return self.__init_handle_by_constructor__(
            _ffi_api.BasicBlock,
            seq,
            args,
            ret,
            start_block_idx,
            start_binding_idx,
            end_block_idx,
            end_binding_idx,
        )  # type: ignore


@tvm._ffi.register_object("relax.analysis.ControlFlowGraph")
class ControlFlowGraph(Node):
    """Representation of a control flow graph, marking the successors
    and predecessors to all basic blocks"""

    def __init__(self, blocks: List[BasicBlock], preds: List[List[int]], succs: List[List[int]]):
        """
        Instantiate a control flow graph

        Parameters
        ----------
        blocks: List[BasicBlock]
            List of basic blocks in the graph

        preds: List[List[int]]
            The ith member is the list of predecessors to blocks[i] (given as indices in blocks)

        succs: List[List[int]]
            The ith member is the list of successors to blocks[i] (given as indices in blocks)
        """
        if len(blocks) != len(preds) or len(blocks) != len(succs):
            raise ValueError("The lengths of blocks, preds, and succs must all match.")

        return self.__init_handle_by_constructor__(
            _ffi_api.ControlFlowGraph, blocks, preds, succs
        )  # type: ignore


def ExtractCFG(func: Function) -> ControlFlowGraph:
    """
    Given a Relax function, produces the corresponding control flow graph.
    The function is expected to have been normalized.

    Parameters
    ----------
    func: Function
        A Relax function. Must be in normal form.

    Returns
    -------
    graph: ControlFlowGraph
        Control flow graph corresponding to the function.
    """
    return _ffi_api.ExtractCFG(func)  # type: ignore
