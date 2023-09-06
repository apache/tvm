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
from enum import Enum
from typing import Any, Callable, List, Tuple
import tvm
from tvm.ir.base import Node
from tvm.relax.expr import Expr, SeqExpr, Function, Var
from . import _ffi_api


class BindingNodeKind(Enum):
    kBinding = 0
    kIfCond = 1
    kIfMerge = 2
    kSeqBody = 3


@tvm._ffi.register_object("relax.analysis.GraphBinding")
class GraphBinding(Node):
    """Representation of a binding in a control flow graph"""

    seq: SeqExpr
    args: List[Var]
    block_idx: int
    binding_idx: int
    kind: BindingNodeKind

    def __init__(
        self,
        seq: SeqExpr,
        args: List[Var],
        block_idx: int,
        binding_idx: int,
        kind: BindingNodeKind,
    ):
        """
        Create a graph binding

        Parameters
        ----------
        seq: SeqExpr
            The SeqExpr that contains the binding

        args: List[Var]
            Arguments taken by the binding (only used for the entry binding:
            these will be the function arguments. Otherwise, this array should be empty.)

        block_idx: int
            The index of the block in the SeqExpr's block list where the binding resides
            (convention: for the SeqExpr body, we will use one past the final block)

        binding_idx: int
            The index of the binding in the binding block corresponding to this binding.

        kind: BindingNodeKind
            The kind of binding. We distinguish between ordinary bindings,
            If conditions, If merges (the var bound to the result of the If node),
            and the body of the SeqExpr.
        """
        return self.__init_handle_by_constructor__(
            _ffi_api.GraphBinding,
            seq,
            args,
            block_idx,
            binding_idx,
            kind,
        )  # type: ignore


@tvm._ffi.register_object("relax.analysis.ControlFlowGraph")
class ControlFlowGraph(Node):
    """Representation of a control flow graph, marking the successors
    and predecessors to all basic blocks"""

    def __init__(
        self, bindings: List[GraphBinding], preds: List[List[int]], succs: List[List[int]]
    ):
        """
        Instantiate a control flow graph

        Parameters
        ----------
        bindings: List[GraphBnding]
            List of bindings in the graph

        preds: List[List[int]]
            The ith member is the list of predecessors to bindings[i] (given as indices in bindings)

        succs: List[List[int]]
            The ith member is the list of successors to bindings[i] (given as indices in bindings)
        """
        if len(bindings) != len(preds) or len(bindings) != len(succs):
            raise ValueError("The lengths of blocks, preds, and succs must all match.")

        return self.__init_handle_by_constructor__(
            _ffi_api.ControlFlowGraph, bindings, preds, succs
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


def GetBindingIndex(
    cfg: ControlFlowGraph, seq: SeqExpr, block_idx: int, binding_idx: int, match_cond: bool = False
) -> int:
    """
    Helper function. Given a control flow graph and a seq expression with a block index and binding index,
    return the index of the corresponding GraphBinding in the CFG

    Parameters
    ----------
    cfg: ControlFlowGraph
        The control flow graph.

    seq: SeqExpr
        The target SeqExpr.

    block_idx: int
        The index of the target block in seq.
        Convention: If the target is `seq.body`, block_idx should be one past the last block
        (i.e., it should be equal to `len(seq.blocks)`).

    binding_idx: int
        The index of the target binding in the target block.

    match_cond: bool
        If true and the target binding in seq is an IfNode, then this function will return
        the binding index corresponding to the If condition.
        If false, then this function will return the binding index corresponding to the If merge.

    Returns
    -------
    idx: int
        The index of the corresponding GraphBindindg in `cfg.bindings`.
    """
    return _ffi_api.GetBindingIndex(cfg, seq, block_idx, binding_idx, match_cond)  # type: ignore


def DataflowAnalysis(
    cfg: ControlFlowGraph,
    init: Any,
    transfer_func: Callable[[GraphBinding, Any], Any],
    merge_func: Callable[[Any, Any], Any],
    forward: bool = True,
) -> Tuple[List[Any], List[Any]]:
    """
    Generic dataflow analysis framework, based on Adrian Sampson's course notes,
    except binding by binding instead of basic block by basic block:
    https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/

    The analysis creates input and output maps (mapping binding indices to a domain),
    sets the initial input and output for each binding to the init value, and then
    performs a traversal of the CFG (BFS in this implementation, since unlike the general case,
    we do not have loops) and uses the transfer and merge function to update the inputs and
    outputs. The analysis can proceed forwards (from binding 0 onwards) or backwards (from the last
    binding back), flipping the roles of the input and output maps in the cases.

    Parameters
    ----------
    cfg: ControlFlowGraph
        The input control flow graph

    init: Any
        The initial value in the analysis domain to which all blocks should be initialized.

    transfer_func: Callable[[GraphBinding, Any], Any]
        Given a binding and the input domain, compute the new output domain.

    merge_func: Callable[[Any, Any], Any]
        When two output domains are fed into a single block (i.e., after an If branch),
        the merge function is used to combine them into a single domain.

    forward: bool
        If true, the analysis proceeds forwards (starting from binding 0 and going onwards).
        If false, the analysis proceeds backwards (starting from the last binding and going back).
        The input and output maps play the opposite roles in forward and backward analyses.
        I.e., in a backward analysis, the "final output" is the input map entry for binding 0
        and the initial input is the output map entry for the last binding.

    Returns
    -------
    ret: Tuple[List[Any], List[Any]]
        A pair of the final input and output maps
    """
    return _ffi_api.DataflowAnalysis(cfg, init, transfer_func, merge_func, forward)  # type: ignore
