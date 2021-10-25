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
"""NodeEdgeGenerator interface for :py:class:`tvm.contrib.relay_viz.plotter.Graph`."""
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


class VizNode:
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


class VizEdge:
    """Edges for `plotter.Graph` interface."""

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
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Generate node and edges consumed by Graph interfaces.

        Parameters
        ----------
        node : relay.Expr
            relay.Expr which will be parsed and generate a node and edges.

        relay_param: Dict[str, tvm.runtime.NDArray]
            relay parameters dictionary.

        node_to_id : Dict[relay.Expr, Union[int, str]]
            a mapping from relay.Expr to node id which should be unique.

        Returns
        -------
        rv1 : Union[VizNode, None]
            VizNode represent the relay.Expr. If the relay.Expr is not intended to introduce a node
            to the graph, return None.

        rv2 : List[VizEdge]
            a list of VizEdge to describe the connectivity of the relay.Expr.
            Can be empty list to indicate no connectivity.
        """


class DefaultNodeEdgeGenerator(NodeEdgeGenerator):
    """NodeEdgeGenerator generate for nodes and edges consumed by Graph.
    This class is a default implementation for common relay types, heavily based on
    `visualize` function in https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
    """

    def __init__(self):
        self._render_rules = {}
        self._build_rules()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        try:
            node_info, edge_info = self._render_rules[type(node)](node, relay_param, node_to_id)
        except KeyError:
            node_info = VizNode(
                node_to_id[node], UNKNOWN_TYPE, f"don't know how to parse {type(node)}"
            )
            edge_info = []
        return node_info, edge_info

    def _var_node(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay var node"""
        node_id = node_to_id[node]
        name_hint = node.name_hint
        node_detail = f"name_hint: {name_hint}"
        node_type = "Var(Param)" if name_hint in relay_param else "Var(Input)"
        if node.type_annotation is not None:
            if hasattr(node.type_annotation, "shape"):
                shape = tuple(map(int, node.type_annotation.shape))
                dtype = node.type_annotation.dtype
                node_detail = f"name_hint: {name_hint}\nshape: {shape}\ndtype: {dtype}"
            else:
                node_detail = f"name_hint: {name_hint}\ntype_annotation: {node.type_annotation}"
        node_info = VizNode(node_id, node_type, node_detail)
        edge_info = []
        return node_info, edge_info

    def _function_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay function node"""
        node_details = []
        name = ""
        func_attrs = node.attrs
        if func_attrs:
            node_details = [f"{k}: {func_attrs.get_str(k)}" for k in func_attrs.keys()]
            # "Composite" might from relay.transform.MergeComposite
            if "Composite" in func_attrs.keys():
                name = func_attrs["Composite"]
        node_id = node_to_id[node]
        node_info = VizNode(node_id, f"Func {name}", "\n".join(node_details))
        edge_info = [VizEdge(node_to_id[node.body], node_id)]
        return node_info, edge_info

    def _call_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        """Render rule for a relay call node"""
        node_id = node_to_id[node]
        op_name = UNKNOWN_TYPE
        node_detail = []
        if isinstance(node.op, tvm.ir.Op):
            op_name = node.op.name
            if node.attrs:
                node_detail = [f"{k}: {node.attrs.get_str(k)}" for k in node.attrs.keys()]
        elif isinstance(node.op, relay.Function):
            func_attrs = node.op.attrs
            op_name = "Anonymous Func"
            if func_attrs:
                node_detail = [f"{k}: {func_attrs.get_str(k)}" for k in func_attrs.keys()]
                # "Composite" might from relay.transform.MergeComposite
                if "Composite" in func_attrs.keys():
                    op_name = func_attrs["Composite"]
        elif isinstance(node.op, relay.GlobalVar):
            op_name = "GlobalVar"
            node_detail = [f"GlobalVar.name_hint: {node.op.name_hint}"]
        else:
            op_name = str(type(node.op)).split(".")[-1].split("'")[0]

        node_info = VizNode(node_id, f"Call {op_name}", "\n".join(node_detail))
        args = [node_to_id[arg] for arg in node.args]
        edge_info = [VizEdge(arg, node_id) for arg in args]
        return node_info, edge_info

    def _tuple_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]
        node_info = VizNode(node_id, "Tuple", "")
        edge_info = [VizEdge(node_to_id[field], node_id) for field in node.fields]
        return node_info, edge_info

    def _tuple_get_item_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]
        node_info = VizNode(node_id, f"TupleGetItem", "idx: {node.index}")
        edge_info = [VizEdge(node_to_id[node.tuple_value], node_id)]
        return node_info, edge_info

    def _constant_node(
        self,
        node: relay.Expr,
        _: Dict[str, tvm.runtime.NDArray],  # relay_param
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:
        node_id = node_to_id[node]
        node_detail = f"shape: {node.data.shape}, dtype: {node.data.dtype}"
        node_info = VizNode(node_id, "Const", node_detail)
        edge_info = []
        return node_info, edge_info

    def _null(self, *_) -> Tuple[None, List[VizEdge]]:
        return None, []

    def _build_rules(self):
        self._render_rules = {
            tvm.relay.Function: self._function_node,
            tvm.relay.expr.Call: self._call_node,
            tvm.relay.expr.Var: self._var_node,
            tvm.relay.expr.Tuple: self._tuple_node,
            tvm.relay.expr.TupleGetItem: self._tuple_get_item_node,
            tvm.relay.expr.Constant: self._constant_node,
            tvm.relay.expr.GlobalVar: self._null,
            tvm.ir.Op: self._null,
        }
