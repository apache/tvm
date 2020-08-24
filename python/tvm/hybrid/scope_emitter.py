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
"""Hybrid Script Scope Emitter for TIR"""

from tvm.te import schedule


class ScopeEmitter:
    """Maintain the nodes and symbols of scopes"""

    def __init__(self, parser):
        self.node_stack = [[]]  # AST nodes of scopes
        self.symbols = [dict()]  # Symbols of scopes
        self.parser = parser

    def pop_scope(self):
        """Pop the inner most scope"""
        self.symbols.pop()
        self.node_stack.pop()

    def new_scope(self):
        """ Creating a new scope """
        self.node_stack.append([])
        self.symbols.append(dict())

    def update_symbol(self, name, symbol):
        """Append a symbol into current scope"""
        if isinstance(symbol, schedule.Buffer):
            if name in self.symbols[0]:
                self.parser.report_error("Duplicate Buffer name")
            self.symbols[0][name] = symbol
        else:
            self.symbols[-1][name] = symbol

    def remove_symbol(self, name):
        """Remove a symbol"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                symbols.pop(name)
                return
        raise RuntimeError("Internal error of hybrid parser: no symbol named" + name)

    def lookup_symbol(self, name):
        """Look up symbol by name"""
        for symbols in reversed(self.symbols):
            if name in symbols:
                return symbols[name]
        return None
