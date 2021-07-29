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
"""Collect nodes that have more than a single child (branching) from a Relay graph."""
import tvm


class CollectBranchingNodes:
    """Collect nodes that have more than a single child (branching) from a Relay graph."""

    class _BranchingNodeCollector(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self._branching_nodes = set()

        def collect(self, expr):
            self.visit(expr)
            return self._branching_nodes

        def visit(self, expr):
            if (not isinstance(expr, tvm.ir.Op)) and (expr in self.memo_map):
                self._branching_nodes.add(expr)
            return super().visit(expr)

        def visit_function(self, fn):
            self.visit(fn.body)

    class _RelayTopologicalSorter(tvm.relay.ExprVisitor):
        def __init__(self, expr_root):
            super().__init__()
            self._expr_root = expr_root

        def sort(self, branching_nodes_set):
            self._branching_nodes_set = branching_nodes_set
            self._ret = []
            self.visit(self._expr_root)
            return self._ret

        def visit(self, expr):
            super().visit(expr)
            if expr in self._branching_nodes_set:
                self._ret.append(expr)
                self._branching_nodes_set.remove(expr)

    def collect(self, expr):
        """Collect nodes that have more than a single child (branching) from a Relay graph.

        Parameters
        ----------
        expr: tvm.relay.Expr
            The expression whose branching children are to be collected.

        Returns
        -------
        branching_nodes: list of tvm.relay.Expr
            The expressions where branching happens.
        """
        branching_nodes_set = self._BranchingNodeCollector().collect(expr)
        return self._RelayTopologicalSorter(expr).sort(branching_nodes_set)
