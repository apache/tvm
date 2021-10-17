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
"""Visualize Relay IR in AST text-form"""

from collections import deque
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)

import tvm
from tvm import relay

from .plotter import (
    Plotter,
    Graph,
)

from .node_edge_gen import (
    Node,
    Edge,
    NodeEdgeGenerator,
    DefaultNodeEdgeGenerator,
)


class TermNodeEdgeGenerator(NodeEdgeGenerator):
    """Terminal nodes and edges generator."""

    def __init__(self):
        self._default_ne_gen = DefaultNodeEdgeGenerator()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        """Generate node and edges consumed by TermGraph interfaces"""
        if isinstance(node, relay.Call):
            return self._call_node(node, node_to_id)

        if isinstance(node, relay.Let):
            return self._let_node(node, node_to_id)

        if isinstance(node, relay.GlobalVar):
            return self._global_var_node(node, node_to_id)

        if isinstance(node, relay.If):
            return self._if_node(node, node_to_id)

        if isinstance(node, tvm.ir.Op):
            return self._op_node(node, node_to_id)

        if isinstance(node, relay.Function):
            return self._function_node(node, node_to_id)

        # otherwise, delegate to the default implementation
        return self._default_ne_gen.get_node_edges(node, relay_param, node_to_id)

    def _call_node(self, node, node_to_id):
        node_id = node_to_id[node]
        node_info = Node(node_id, "Call", "")
        edge_info = [Edge(node_to_id[node.op], node_id)]
        for arg in node.args:
            arg_nid = node_to_id[arg]
            edge_info.append(Edge(arg_nid, node_id))
        return node_info, edge_info

    def _let_node(self, node, node_to_id):
        node_id = node_to_id[node]
        node_info = Node(node_id, "Let", "(var, val, body)")
        edge_info = [
            Edge(node_to_id[node.var], node_id),
            Edge(node_to_id[node.value], node_id),
            Edge(node_to_id[node.body], node_id),
        ]
        return node_info, edge_info

    def _global_var_node(self, node, node_to_id):
        node_id = node_to_id[node]
        node_info = Node(node_id, "GlobalVar", node.name_hint)
        edge_info = []
        return node_info, edge_info

    def _if_node(self, node, node_to_id):
        node_id = node_to_id[node]
        node_info = Node(node_id, "If", "(cond, true, false)")
        edge_info = [
            Edge(node_to_id[node.cond], node_id),
            Edge(node_to_id[node.true_branch], node_id),
            Edge(node_to_id[node.false_branch], node_id),
        ]
        return node_info, edge_info

    def _op_node(self, node, node_to_id):
        node_id = node_to_id[node]
        op_name = node.name
        node_info = Node(node_id, op_name, "")
        edge_info = []
        return node_info, edge_info

    def _function_node(self, node, node_to_id):
        node_id = node_to_id[node]
        node_info = Node(node_id, "Func", str(node.params))
        edge_info = [Edge(node_to_id[node.body], node_id)]
        return node_info, edge_info


class TermNode:
    def __init__(self, node_type, other_info):
        self.type = node_type
        self.other_info = other_info.replace("\n", ", ")


class TermGraph(Graph):
    """Terminal plot for a relay IR Module"""

    def __init__(self, name):
        # node_id: [ connected node_id]
        self._name = name
        self._graph = {}
        self._id_to_node = {}
        # reversed post order
        self._node_ids_rpo = deque()

    def node(self, node_id, node_type, node_detail):
        # actually we just need the last one.
        self._node_ids_rpo.appendleft(node_id)

        if node_id not in self._graph:
            self._graph[node_id] = []

        node = TermNode(node_type, node_detail)
        self._id_to_node[node_id] = node

    def edge(self, id_start, id_end):
        # want reserved post-order
        if id_end in self._graph:
            self._graph[id_end].append(id_start)
        else:
            self._graph[id_end] = [id_start]

    def render(self):
        """To draw a terminal graph"""
        lines = []
        seen_node = set()

        def gen_line(indent, n_id):
            if (indent, n_id) in seen_node:
                return
            seen_node.add((indent, n_id))

            conn_symbol = ["|--", "`--"]
            last = len(self._graph[n_id]) - 1
            for i, next_n_id in enumerate(self._graph[n_id]):
                node = self._id_to_node[next_n_id]
                lines.append(
                    f"{indent}{conn_symbol[1 if i==last else 0]}{node.type} {node.other_info}"
                )
                next_indent = indent
                next_indent += "   " if (i == last) else "|  "
                gen_line(next_indent, next_n_id)

        first_node_id = self._node_ids_rpo[0]
        node = self._id_to_node[first_node_id]
        lines.append(f"@{self._name}({node.other_info})")
        gen_line("", first_node_id)

        return "\n".join(lines)


class TermPlotter(Plotter):
    """Terminal plotter"""

    def __init__(self):
        self._name_to_graph = {}

    def create_graph(self, name):
        self._name_to_graph[name] = TermGraph(name)
        return self._name_to_graph[name]

    def render(self, filename):
        """If filename is None, print to stdio. Otherwise, write to the filename."""
        lines = []
        for name in self._name_to_graph:
            lines.append(self._name_to_graph[name].render())
        if filename is None:
            print("\n".join(lines))
        else:
            with open(filename, "w") as out_file:
                out_file.write("\n".join(lines))
