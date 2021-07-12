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

import tvm
from tvm import relay

def render_cb(graph, node_to_id, relay_param):
    # Based on https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
    unknown_type = "unknown"
    for node, node_id in node_to_id.items():
        if isinstance(node, relay.Function):
            graph.node(node_id, f"Func", str(node.params))
            graph.edge(node_to_id[node.body], node_id)
        elif isinstance(node, relay.Var):
            name_hint = node.name_hint
            node_detail = ""
            node_type = "Var(Param)" if name_hint in relay_param else "Var(Input)"
            if node.type_annotation is not None:
                if hasattr(node.type_annotation, "shape"):
                    shape = tuple(map(int, node.type_annotation.shape))
                    dtype = node.type_annotation.dtype
                    node_detail = "name_hint: {}\nshape: {}\ndtype: {}".format(
                        name_hint, shape, dtype
                    )
                else:
                    node_detail = "name_hint: {}\ntype_annotation: {}".format(
                        name_hint, node.type_annotation
                    )
            graph.node(node_id, node_type, node_detail)
        elif isinstance(node, relay.GlobalVar):
            # Dont render this because GlobalVar is put to another graph.
            pass
        elif isinstance(node, relay.Tuple):
            graph.node(node_id, "Tuple", "")
            for field in node.fields:
                graph.edge(node_to_id[field], node_id)
        elif isinstance(node, relay.expr.Constant):
            node_detail = "shape: {}, dtype: {}".format(node.data.shape, node.data.dtype)
            graph.node(node_id, "Const", str(node))
        elif isinstance(node, relay.expr.Call):
            op_name = unknown_type
            node_details = []
            if isinstance(node.op, tvm.ir.Op):
                op_name = node.op.name
                if node.attrs:
                    node_details = [
                        "{}: {}".format(k, node.attrs.get_str(k)) for k in node.attrs.keys()
                    ]
            elif isinstance(node.op, relay.Function):
                func_attrs = node.op.attrs
                op_name = "Anonymous Func"
                if func_attrs:
                    node_details = [
                        "{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()
                    ]
                    # "Composite" might from relay.transform.MergeComposite
                    if "Composite" in func_attrs.keys():
                        op_name = func_attrs["Composite"]
            elif isinstance(node.op, relay.GlobalVar):
                op_name = "GlobalVar"
                node_details = [f"GlobalVar.name_hint: {node.op.name_hint}"]
            else:
                op_name = str(type(node.op)).split(".")[-1].split("'")[0]

            graph.node(node_id, f"Call {op_name}", "\n".join(node_details))
            args = [node_to_id[arg] for arg in node.args]
            for arg in args:
                graph.edge(arg, node_id)
        elif isinstance(node, relay.expr.TupleGetItem):
            graph.node(node_id, "TupleGetItem", "idx: {}".format(node.index))
            graph.edge(node_to_id[node.tuple_value], node_id)
        elif isinstance(node, tvm.ir.Op):
            pass
        elif isinstance(node, relay.Let):
            graph.node(node_id, "Let", "")
            graph.edge(node_to_id[node.value], node_id)
            graph.edge(node_id, node_to_id[node.var])
        else:
            unknown_info = "Unknown node: {}".format(type(node))
            graph.node(node_id, unknown_type, unknown_info)


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

        def gen_line(indent, n_id):
            conn_symbol = "|--"
            last_idx = len(lines) + len(self._graph[n_id]) - 1
            for next_n_id in self._graph[n_id]:
                node = self._id_to_node[next_n_id]
                lines.append(f"{indent}{conn_symbol}{node.type} {node.other_info}")
                gen_line(f"  {indent}", next_n_id)
            if len(self._graph[n_id]):
                lines[last_idx] = lines[last_idx].replace("|", "`")

        first_node_id = self._node_ids_rpo[0]
        node = self._id_to_node[first_node_id]
        lines.append(f"@{self._name}({node.other_info})")
        gen_line("  ", first_node_id)

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
