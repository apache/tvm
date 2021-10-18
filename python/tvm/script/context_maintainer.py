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
"""TVM Script Context Maintainer for TIR"""

from typing import List, Mapping, Union, Optional, Dict, Callable
import synr


import tvm
from tvm.ir import Span
from tvm.ir.expr import Range
from tvm.tir import Var, Buffer, PrimExpr, Stmt, MatchBufferRegion
from tvm.runtime import Object
from tvm.tir.expr import IterVar
from .tir.node import BufferSlice


class BlockInfo:
    """Information for block and block_realize signature

    Examples
    ----------
    .. code-block:: python

        @T.prim_func
        def example_func(a: T.handle, b: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, (16, 16), "float32")
            B = T.match_buffer(b, (16, 16), "float32")
            C = T.match_buffer(a, (16, 16), "float32")

            for i, j, k in T.grid(16, 16, 16):
                with T.block("matmul"):
                    vi = T.axis.S(16, i)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k)         # iter_bindings = {vj: i, vj: j, vk: k}

                    T.where(True)         # predicate of the block_realize

                    T.reads(A[0:16, 0:16], B[0: 16, 0: 16])      # reads region of the block
                    T.writes(C[0: 16, 0: 16])                    # writes region of the block
                    T.block_attr({"attr_key": "attr_value"})     # block annotations

                    # alloc_buffers inside the block
                    CC = T.alloc_buffer((1, 1), dtype="float32")

                    # match_buffers of the block,
                    # which bind a sub-region of source buffer into a new buffer
                    D = T.match_buffer(C[vi, vj], ())

                    # init part of the block, executed when all reduce axes are the beginning value
                    with T.init():
                        C[vi, vj] = T.float32(0)

                    # block body
                    CC[0, 0] = A[vi, vk] * B[vj, vk]
                    D[()] += CC[0, 0]         # The same as C[vi, vj] += CC[0, 0]
    """

    alloc_buffers: List[Buffer] = []
    """List[Buffer]: list of T.alloc_buffer statements in the block signature"""
    match_buffers: List[MatchBufferRegion] = []
    """List[MatchBufferRegion]: list of T.match_buffer statements in the block signature"""
    iter_values: List[PrimExpr] = []
    """List[PrimExpr]: list of binding values for iter vars"""
    iter_vars: List[IterVar] = []
    """List[PrimExpr]: list of iter vars in the block"""
    reads: Optional[List[BufferSlice]] = None
    """Optional[List[BufferSlice]]:
    list of T.reads statements in the block signature, None for not-visited"""
    writes: Optional[List[BufferSlice]] = None
    """Optional[List[BufferSlice]]:
    list of T.writes statements in the block signature, None for not-visited"""
    annotations: Optional[Mapping[str, Object]] = None
    """Optional[Mapping[str, Object]]:
    list of T.block_attr statements in the block signature, None for not-visited"""
    predicate: Optional[PrimExpr] = None
    """Optional[PrimExpr]: block realize predicate, None for not-visited"""
    init: Optional[Stmt] = None
    """Optional[Stmt]: init part of the block, None for not-visited"""

    def __init__(self):
        self.alloc_buffers = []
        self.match_buffers = []
        self.iter_values = []
        self.iter_vars = []
        self.reads = None
        self.writes = None
        self.annotations = None
        self.predicate = None
        self.init = None


class ContextMaintainer:
    """Maintain all the necessary context info
    Parameters
    ----------
    _report_error : Callable[[str, Union[Span, synr.ast.Span]], None]
        The report error function handle
    """

    # scope context
    node_stack: List[List[synr.ast.Node]] = []
    """List[List[synr.ast.Node]]: The ast nodes insides the current scope"""
    block_info_stack: List[BlockInfo] = []
    """List[BlockInfo]: The block info for the current block scope"""
    loop_stack: Dict[Var, Range] = {}
    """Dict[Var, Range]: The dict from loop var to its domain outside the block"""
    symbols: List[Dict[str, Union[Var, Buffer]]] = []
    """List[Dict[str, Union[Var, Buffer]]]: Symbol map from name to object for the current scope"""

    # function context
    func_params: List[Var] = []
    """List[Var]: The function parameters"""
    func_buffer_map: Mapping[Var, Buffer] = {}
    """Mapping[Var, Buffer]: The function buffer map"""
    func_dict_attr: Mapping[str, Object] = {}
    """Mapping[str, Object]: The function attrs"""
    func_var_env_dict: Mapping[Var, str] = {}
    """Mapping[Var, str]: The map from var to env thread"""

    # parser and analyzer
    analyzer: tvm.arith.Analyzer = tvm.arith.Analyzer()
    """tvm.arith.Analyzer: The analyzer for simplifying"""
    _report_error: Callable[[str, Union[Span, synr.ast.Span]], None]
    """Callable[[str, Union[Span, synr.ast.Span]], None]: The report error function handle"""

    def __init__(self, _report_error: Callable[[str, Union[Span, synr.ast.Span]], None]):
        # scope context
        self.node_stack = []
        self.block_info_stack = []
        self.loop_stack = {}
        self.symbols = []
        # function context
        self.func_params = []
        self.func_buffer_map = {}
        self.func_dict_attr = {}
        self.func_var_env_dict = {}
        # parser and analyzer
        self._report_error = _report_error
        self.analyzer = tvm.arith.Analyzer()

    def enter_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creates a new scope

        Note
        ----
        This function is used for normal scopes that do not involve
        a `with block` scope. Use `enter_block_scope`
        for block scope cases.

        Parameters
        ----------
        nodes : Optional[List[synr.ast.Node]]
            The synr AST nodes in new scope
        """
        if nodes is None:
            nodes = []
        self.node_stack.append(list(reversed(nodes)))
        self.symbols.append(dict())

    def enter_block_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creates a new block scope, the function will call `enter_scope` implicitly
        Besides the behaviors of `enter_scope`, it will update loop_stack and block_info_stack
        to maintain block info.

        Note
        ----
        This function should be used to handle a block scope,
        aka the blocks that involve a `with block` scope.

        Parameters
        ----------
        nodes : Optional[List[synr.ast.Node]]
            The synr AST nodes in new scope
        """
        self.enter_scope(nodes)
        # Create a new BlockInfo for the new block
        self.block_info_stack.append(BlockInfo())

    def exit_scope(self):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.node_stack.pop()

    def exit_block_scope(self):
        """Pop the inner most block scope, the function will call `exit_scope` implicitly"""
        self.exit_scope()
        # Pop block_info
        self.block_info_stack.pop()

    def update_symbol(self, name: str, symbol: Union[Buffer, Var], node: synr.ast.Node):
        """Append a symbol into current scope"""
        if isinstance(symbol, Buffer):
            if name in self.symbols[0]:
                self.report_error("Duplicate Buffer name: " + symbol.name, node.span)
            self.symbols[0][name] = symbol
        else:
            self.symbols[-1][name] = symbol

    def remove_symbol(self, name: str):
        """Remove a symbol"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                symbols.pop(name)
                return
        raise RuntimeError("Internal error of tvm script parser: no symbol named " + name)

    def lookup_symbol(self, name: str) -> Optional[Union[Buffer, Var]]:
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None

    def report_error(self, message: str, span: Union[Span, synr.ast.Span]):
        self._report_error(message, span)

    def current_block_scope(self) -> BlockInfo:
        return self.block_info_stack[-1]
