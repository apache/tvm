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
from tvm.tir import Var, Buffer, PrimExpr, Stmt, MatchBufferRegion
from tvm.runtime import Object
from .node import BufferSlice


class BlockInfo:
    """Information for block and block_realize signature"""

    alloc_buffers: List[Buffer]
    match_buffers: List[MatchBufferRegion]
    iter_bindings: Mapping[Var, PrimExpr]
    reads: Optional[List[BufferSlice]]
    writes: Optional[List[BufferSlice]]
    annotations: Optional[Mapping[str, Object]]
    predicate: Optional[PrimExpr]
    init: Optional[Stmt]

    def __init__(self):
        self.alloc_buffers = []
        self.match_buffers = []
        self.iter_bindings = {}
        self.reads = None
        self.writes = None
        self.annotations = None
        self.predicate = None
        self.init = None


class ContextMaintainer:
    """Maintain all the necessary context info"""

    # scope context
    # ast_node inside a scope
    node_stack: List[List[synr.ast.Node]]
    # loop stacks inside a block
    block_info_stack: List[BlockInfo]
    # loop stacks inside a block
    loop_stack: List[List[Var]]
    symbols: List[Dict[str, Union[Var, Buffer]]]

    # function context
    func_params: List[Var]
    func_buffer_map: Mapping[Var, Buffer]
    func_dict_attr: Mapping[str, Object]
    func_var_env_dict: Mapping[Var, str]

    # parser and analyzer
    _report_error: Callable[[str, Union[Span, synr.ast.Span]], None]
    analyzer: tvm.arith.Analyzer

    def __init__(self, _report_error: Callable[[str, Union[Span, synr.ast.Span]], None]):
        # scope context
        self.node_stack = []  # AST nodes of scopes
        self.block_info_stack = []  # Block info of scopes
        self.loop_stack = []  # stack of loop vars
        self.symbols = []  # symbols of scopes
        # function context
        self.func_params = []  # parameter list of function
        self.func_buffer_map = {}  # buffer_map of function
        self.func_dict_attr = {}  # func_attr of function
        self.func_var_env_dict = {}  # map from var to env_name
        # parser and analyzer
        self._report_error = _report_error
        self.analyzer = tvm.arith.Analyzer()

    def enter_scope(self, nodes: Optional[List[synr.ast.Node]] = None):
        """Creating a new scope

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
        """Creating a new block scope, the function will call `enter_scope` implicitly
        Besides behaviors of normal `enter_scope`, it will update loop_stack and block_info_stack
        for block info maintaining.

        Parameters
        ----------
        nodes : Optional[List[synr.ast.Node]]
            The synr AST nodes in new scope
        """
        self.enter_scope(nodes)
        # Create a new loop stack for the new block
        self.loop_stack.append([])
        # Create a new BlockInfo for the new block
        self.block_info_stack.append(BlockInfo())

    def exit_scope(self):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.node_stack.pop()

    def exit_block_scope(self):
        """Pop the inner most block scope, the function will call `exit_scope` implicitly"""
        self.exit_scope()
        # Pop loop stack
        self.loop_stack.pop()
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
