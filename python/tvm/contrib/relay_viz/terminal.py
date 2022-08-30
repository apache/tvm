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
"""Visualize Relay IR in AST text-form."""

from collections import deque
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from .interface import (
    DefaultVizParser,
    Plotter,
    VizEdge,
    VizGraph,
    VizNode,
    VizParser,
)


class TermVizParser(VizParser):
    """`TermVizParser` parse nodes and edges for `TermPlotter`."""

    def __init__(self):
        self._default_parser = DefaultVizParser()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Parse a node and edges from a relay.Expr."""
        if isinstance(node, relay.Call):
            return self._call(node, node_to_id)
        if isinstance(node, relay.Let):
            return self._let(node, node_to_id)
        if isinstance(node, relay.GlobalVar):
            return self._global_var(node, node_to_id)
        if isinstance(node, relay.If):
            return self._if(node, node_to_id)
        if isinstance(node, tvm.ir.Op):
            return self._op(node, node_to_id)
        if isinstance(node, relay.Function):
            return self._function(node, node_to_id)

        # Leverage logics from default parser.
        return self._default_parser.get_node_edges(node, relay_param, node_to_id)

    def _call(self, node, node_to_id):
        node_id = node_to_id[node]
        viz_node = VizNode(node_id, "Call", "")
        viz_edges = [VizEdge(node_to_id[node.op], node_id)]
        for arg in node.args:
            arg_id = node_to_id[arg]
            viz_edges.append(VizEdge(arg_id, node_id))
        return viz_node, viz_edges

    def _let(self, node, node_to_id):
        node_id = node_to_id[node]
        viz_node = VizNode(node_id, "Let", "(var, val, body)")
        viz_edges = [
            VizEdge(node_to_id[node.var], node_id),
            VizEdge(node_to_id[node.value], node_id),
            VizEdge(node_to_id[node.body], node_id),
        ]
        return viz_node, viz_edges

    def _global_var(self, node, node_to_id):
        node_id = node_to_id[node]
        viz_node = VizNode(node_id, "GlobalVar", node.name_hint)
        viz_edges = []
        return viz_node, viz_edges

    def _if(self, node, node_to_id):
        node_id = node_to_id[node]
        viz_node = VizNode(node_id, "If", "(cond, true, false)")
        viz_edges = [
            VizEdge(node_to_id[node.cond], node_id),
            VizEdge(node_to_id[node.true_branch], node_id),
            VizEdge(node_to_id[node.false_branch], node_id),
        ]
        return viz_node, viz_edges

    def _op(self, node, node_to_id):
        node_id = node_to_id[node]
        op_name = node.name
        viz_node = VizNode(node_id, op_name, "")
        viz_edges = []
        return viz_node, viz_edges

    def _function(self, node, node_to_id):
        node_id = node_to_id[node]
        viz_node = VizNode(node_id, "Func", str(node.params))
        viz_edges = [VizEdge(node_to_id[node.body], node_id)]
        return viz_node, viz_edges


class TermNode:
    """TermNode is aimed to generate text more suitable for terminal visualization."""

    def __init__(self, viz_node: VizNode):
        self.type = viz_node.type_name
        # We don't want too many lines in a terminal.
        self.other_info = viz_node.detail.replace("\n", ", ")


class TermGraph(VizGraph):
    """Terminal graph for a relay IR Module

    Parameters
    ----------
    name: str
        name of this graph.
    """

    def __init__(self, name: str):
        self._name = name
        # A graph in adjacency list form.
        # The key is source node, and the value is a list of destination nodes.
        self._graph = {}
        # a hash table for quick searching.
        self._id_to_term_node = {}
        # node_id in reversed post order
        # That mean, root is the first node.
        self._node_id_rpo = deque()

    def node(self, viz_node: VizNode) -> None:
        """Add a node to the underlying graph.
        Nodes in a Relay IR Module are expected to be added in the post-order.

        Parameters
        ----------
        viz_node : VizNode
            A `VizNode` instance.
        """

        self._node_id_rpo.appendleft(viz_node.identity)

        if viz_node.identity not in self._graph:
            # Add the node into the graph.
            self._graph[viz_node.identity] = []

        # Create TermNode from VizNode
        node = TermNode(viz_node)
        self._id_to_term_node[viz_node.identity] = node

    def edge(self, viz_edge: VizEdge) -> None:
        """Add an edge to the terminal graph.

        Parameters
        ----------
        viz_edge : VizEdge
            A `VizEdge` instance.
        """
        # Take CallNode as an example, instead of "arguments point to CallNode",
        # we want "CallNode points to arguments" in ast-dump form.
        #
        # The direction of edge is typically controlled by the implemented VizParser.
        # Reverse start/end here simply because we leverage default parser implementation.
        if viz_edge.end in self._graph:
            self._graph[viz_edge.end].append(viz_edge.start)
        else:
            self._graph[viz_edge.end] = [viz_edge.start]

    def render(self) -> str:
        """Draw a terminal graph

        Returns
        -------
        rv1: str
            text representing a graph.
        """
        lines = []
        seen_node = set()

        def gen_line(indent, n_id):
            if (indent, n_id) in seen_node:
                return
            seen_node.add((indent, n_id))

            conn_symbol = ["|--", "`--"]
            last = len(self._graph[n_id]) - 1
            for i, next_n_id in enumerate(self._graph[n_id]):
                node = self._id_to_term_node[next_n_id]
                lines.append(
                    f"{indent}{conn_symbol[1 if i==last else 0]}{node.type} {node.other_info}"
                )
                next_indent = indent
                # increase indent for the next level.
                next_indent += "   " if (i == last) else "|  "
                gen_line(next_indent, next_n_id)

        first_node_id = self._node_id_rpo[0]
        first_node = self._id_to_term_node[first_node_id]
        lines.append(f"@{self._name}({first_node.other_info})")
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
            text_graph = self._name_to_graph[name].render()
            lines.append(text_graph)
        if filename is None:
            print("\n".join(lines))
        else:
            with open(filename, "w") as out_file:
                out_file.write("\n".join(lines))
