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

from typing import Any, Callable, List, Tuple
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


def DataflowAnalysis(
    cfg: ControlFlowGraph,
    init: Any,
    transfer_func: Callable[[BasicBlock, Any], Any],
    merge_func: Callable[[Any, Any], Any],
    forward: bool = True,
) -> Tuple[List[Any], List[Any]]:
    """
    Generic dataflow analysis framework, based on Adrian Sampson's course notes:
    https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/

    The analysis creates input and output maps (mapping basic block indices to a domain),
    sets the initial input and output for each basic block to the init value, and then
    performs a traversal of the CFG (BFS in this implementation, since unlike the general case,
    we do not have loops) and uses the transfer and merge function to update the inputs and
    outputs. The analysis can proceed forwards (from block 0 onwards) or backwards (from the last
    block back), flipping the roles of the input and output maps in the cases.

    Parameters
    ----------
    cfg: ControlFlowGraph
        The input control flow graph

    init: Any
        The initial value in the analysis domain to which all blocks should be initialized.

    transfer_func: Callable[[BasicBlock, Any], Any]
        Given a basic block and the input domain, compute the new output domain.

    merge_func: Callable[[Any, Any], Any]
        When two output domains are fed into a single block (i.e., after an If branch),
        the merge function is used to combine them into a single domain.

    forward: bool
        If true, the analysis proceeds forwards (starting from block 0 and going onwards).
        If false, the analysis proceeds backwards (starting from the last block and going back).
        The input and output maps play the opposite roles in forward and backward analyses.
        I.e., in a backward analysis, the "final output" is the input map entry for block 0
        and the initial input is the output map entry for the last block.

    Returns
    -------
    ret: Tuple[List[Any], List[Any]]
        A pair of the final input and output maps
    """
    return _ffi_api.DataflowAnalysis(forward, cfg, init, transfer_func, merge_func)  # type: ignore
