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

from pyparsing import line

from .plotter import (
    Plotter,
    Graph,
)
import functools

import tvm
from tvm import relay
from .render_callback import RenderCallback


class TermRenderCallback(RenderCallback):
    def __init__(self):
        super().__init__()

    def Call_node(self, node, relay_param, node_to_id):
        node_id = node_to_id[node]
        graph_info = [node_id, f"Call", ""]
        edge_info = [[node_to_id[node.op], node_id]]
        args = [node_to_id[arg] for arg in node.args]
        for arg in args:
            edge_info.append([arg, node_id])
        return graph_info, edge_info

    def Let_node(self, node, relay_param, node_to_id):
        node_id = node_to_id[node]
        graph_info = [node_id, "Let", "(var, val, body)"]
        edge_info = [[node_to_id[node.var], node_id]]
        edge_info.append([node_to_id[node.value], node_id])
        edge_info.append([node_to_id[node.body], node_id])
        return graph_info, edge_info

    def Global_var_node(self, node, relay_param, node_to_id):
        node_id = node_to_id[node]
        graph_info = [node_id, "GlobalVar", node.name_hint]
        edge_info = []
        return graph_info, edge_info

    def If_node(self, node, relay_param, node_to_id):
        node_id = node_to_id[node]
        graph_info = [node_id, "If", "(cond, true, false)"]
        edge_info = [[node_to_id[node.cond], node_id]]
        edge_info.append([node_to_id[node.true_branch], node_id])
        edge_info.append([node_to_id[node.false_branch], node_id])
        return graph_info, edge_info

    def Op_node(self, node, relay_param, node_to_id):
        node_id = node_to_id[node]
        op_name = node.name
        graph_info = [node_id, op_name, ""]
        edge_info = []
        return graph_info, edge_info


class Node:
    def __init__(self, node_type, other_info):
        self.type = node_type
        self.other_info = other_info.replace("\n", ", ")


class TermGraph(Graph):
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

        node = Node(node_type, node_detail)
        self._id_to_node[node_id] = node

    def edge(self, id_start, id_end):
        # want reserved post-order
        if id_end in self._graph:
            self._graph[id_end].append(id_start)
        else:
            self._graph[id_end] = [id_start]

    def render(self):
        lines = []

        @functools.lru_cache()
        def gen_line(indent, n_id):
            conn_symbol = ["|--", "`--"]
            last = len(self._graph[n_id]) - 1
            for i, next_n_id in enumerate(self._graph[n_id]):
                node = self._id_to_node[next_n_id]
                lines.append(f"{indent}{conn_symbol[i==last]}{node.type} {node.other_info}")
                next_indent = indent
                next_indent += "   " if (i == last) else "|  "
                gen_line(next_indent, next_n_id)

        first_node_id = self._node_ids_rpo[0]
        node = self._id_to_node[first_node_id]
        lines.append(f"@{self._name}({node.other_info})")
        gen_line("", first_node_id)

        return "\n".join(lines)


class TermPlotter(Plotter):
    def __init__(self):
        self._name_to_graph = {}

    def create_graph(self, name):
        self._name_to_graph[name] = TermGraph(name)
        return self._name_to_graph[name]

    def render(self, filename):
        # if filename  == "stdio", print to terminal.
        # Otherwise, print to the file?
        lines = []
        for name in self._name_to_graph:
            lines.append(self._name_to_graph[name].render())
        print("\n".join(lines))
