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
"""NodeEdgeGenerator interface"""
import abc
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay

UNKNOWN_TYPE = "unknown"


class Node:
    """Node carry information used by `plotter.Graph` interface."""

    def __init__(self, node_id: Union[int, str], node_type: str, node_detail: str):
        self._id = node_id
        self._type = node_type
        self._detail = node_detail

    @property
    def identity(self) -> Union[int, str]:
        return self._id

    @property
    def type_str(self) -> str:
        return self._type

    @property
    def detail(self) -> str:
        return self._detail


class Edge:
    """Edge for `plotter.Graph` interface."""

    def __init__(self, start_node: Union[int, str], end_node: Union[int, str]):
        self._start_node = start_node
        self._end_node = end_node

    @property
    def start(self) -> Union[int, str]:
        return self._start_node

    @property
    def end(self) -> Union[int, str]:
        return self._end_node


class NodeEdgeGenerator(abc.ABC):
    """An interface class to generate nodes and edges information for Graph interfaces."""

    @abc.abstractmethod
    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        """Generate node and edges consumed by Graph interfaces
        The returned tuple containing Node and a list of Edge instances.
        Tuple[None, list[]] for null results.
        """


class DefaultNodeEdgeGenerator(NodeEdgeGenerator):
    """NodeEdgeGenerator generate for nodes and edges consumed by Graph.
    This class is a default implementation for common relay types, heavily based on
    `visualize` function in https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
    """

    def __init__(self):
        self.render_rules = {}
        self.build_rules()

    def var_node(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        """Render rule for a relay var node"""
        node_id = node_to_id[node]
        name_hint = node.name_hint
        node_detail = ""
        node_type = "Var(Param)" if name_hint in relay_param else "Var(Input)"
        if node.type_annotation is not None:
            if hasattr(node.type_annotation, "shape"):
                shape = tuple(map(int, node.type_annotation.shape))
                dtype = node.type_annotation.dtype
                node_detail = "name_hint: {}\nshape: {}\ndtype: {}".format(name_hint, shape, dtype)
            else:
                node_detail = "name_hint: {}\ntype_annotation: {}".format(
                    name_hint, node.type_annotation
                )
        node_info = Node(node_id, node_type, node_detail)
        edge_info = []
        return node_info, edge_info

    def function_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        """Render rule for a relay function node"""
        node_details = []
        name = ""
        func_attrs = node.attrs
        if func_attrs:
            node_details = ["{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()]
            # "Composite" might from relay.transform.MergeComposite
            if "Composite" in func_attrs.keys():
                name = func_attrs["Composite"]
        node_id = node_to_id[node]
        node_info = Node(node_id, f"Func {name}", "\n".join(node_details))
        edge_info = [Edge(node_to_id[node.body], node_id)]
        return node_info, edge_info

    def call_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        """Render rule for a relay call node"""
        node_id = node_to_id[node]
        op_name = UNKNOWN_TYPE
        node_detail = []
        if isinstance(node.op, tvm.ir.Op):
            op_name = node.op.name
            if node.attrs:
                node_detail = ["{}: {}".format(k, node.attrs.get_str(k)) for k in node.attrs.keys()]
        elif isinstance(node.op, relay.Function):
            func_attrs = node.op.attrs
            op_name = "Anonymous Func"
            if func_attrs:
                node_detail = ["{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()]
                # "Composite" might from relay.transform.MergeComposite
                if "Composite" in func_attrs.keys():
                    op_name = func_attrs["Composite"]
        elif isinstance(node.op, relay.GlobalVar):
            op_name = "GlobalVar"
            node_detail = [f"GlobalVar.name_hint: {node.op.name_hint}"]
        else:
            op_name = str(type(node.op)).split(".")[-1].split("'")[0]

        node_info = Node(node_id, f"Call {op_name}", "\n".join(node_detail))
        args = [node_to_id[arg] for arg in node.args]
        edge_info = [Edge(arg, node_id) for arg in args]
        return node_info, edge_info

    def tuple_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        node_id = node_to_id[node]
        node_info = Node(node_id, "Tuple", "")
        edge_info = [Edge(node_to_id[field], node_id) for field in node.fields]
        return node_info, edge_info

    def tuple_get_item_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        node_id = node_to_id[node]
        node_info = Node(node_id, "TupleGetItem", "idx: {}".format(node.index))
        edge_info = [Edge(node_to_id[node.tuple_value], node_id)]
        return node_info, edge_info

    def constant_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        node_id = node_to_id[node]
        node_detail = "shape: {}, dtype: {}".format(node.data.shape, node.data.dtype)
        node_info = Node(node_id, "Const", node_detail)
        edge_info = []
        return node_info, edge_info

    def null(self, *_) -> Tuple[None, List[Edge]]:
        return None, []

    def build_rules(self):
        self.render_rules = {
            tvm.relay.Function: self.function_node,
            tvm.relay.expr.Call: self.call_node,
            tvm.relay.expr.Var: self.var_node,
            tvm.relay.expr.Tuple: self.tuple_node,
            tvm.relay.expr.TupleGetItem: self.tuple_get_item_node,
            tvm.relay.expr.Constant: self.constant_node,
            tvm.relay.expr.GlobalVar: self.null,
            tvm.ir.Op: self.null,
        }

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[Node, None], List[Edge]]:
        try:
            node_info, edge_info = self.render_rules[type(node)](node, relay_param, node_to_id)
        except KeyError:
            node_info = Node(
                node_to_id[node], UNKNOWN_TYPE, f"don't know how to parse {type(node)}"
            )
            edge_info = []
        return node_info, edge_info
