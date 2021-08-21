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
"""Relay IR Visualizer"""
import logging
import copy
import tvm
from tvm import relay

_LOGGER = logging.getLogger(__name__)


def _dft_render_cb(graph, node_to_id, relay_param):
    """a callback to Add nodes and edges to the graph.

    Parameters
    ----------
    graph : class plotter.Graph

    node_to_id : Dict[relay.expr, int]

    relay_param : Dict[string, NDarray]
    """
    # Based on https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
    unknown_type = "unknown"
    for node, node_id in node_to_id.items():
        if isinstance(node, relay.Function):
            node_details = []
            name = ""
            func_attrs = node.attrs
            if func_attrs:
                node_details = [
                    "{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()
                ]
                # "Composite" might from relay.transform.MergeComposite
                if "Composite" in func_attrs.keys():
                    name = func_attrs["Composite"]

            graph.node(node_id, f"Func {name}", "\n".join(node_details))
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
            node_detail = "shape: {}, dtype: {}, str(node): {}".format(
                node.data.shape, node.data.dtype, str(node)
            )
            graph.node(node_id, "Const", node_detail)
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
                node_details = [f"name_hint: {node.op.name_hint}"]
            else:
                op_name = str(type(node.op)).split(".")[-1].split("'")[0]

            graph.node(node_id, op_name, "\n".join(node_details))
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
            _LOGGER.warning(unknown_info)
            graph.node(node_id, unknown_type, unknown_info)


class PlotterBackend:
    """Enumeration for available plotters."""

    BOKEH = "bokeh"


class RelayVisualizer:
    """Relay IR Visualizer"""

    def __init__(
        self, relay_mod, relay_param=None, plotter_be=PlotterBackend.BOKEH, render_cb=_dft_render_cb
    ):
        """Visualize Relay IR.

        Parameters
        ----------
        relay_mod : object
                        Relay IR module
        relay_param: dict
                        Relay parameter dictionary
        plotter_be: PlotterBackend.
                        The backend of plotting. Default "bokeh"
        render_cb: callable[plotter.Graph, Dict, Dict]
                        A callable accepting plotter.Graph, node_to_id, relay_param.
                        See _dft_render_cb(graph, node_to_id, relay_param) as
                        an example.
        """
        self._plotter = get_plotter(plotter_be)
        self._render_cb = render_cb
        self._relay_param = relay_param if relay_param is not None else {}
        # This field is used for book-keeping for each graph.
        self._node_to_id = {}

        global_vars = relay_mod.get_global_vars()
        graph_names = []
        # If we have main function, put it to the first.
        for gv_name in global_vars:
            if gv_name.name_hint == "main":
                graph_names.insert(0, gv_name.name_hint)
            else:
                graph_names.append(gv_name.name_hint)

        for name in graph_names:
            # clear previous graph
            self._node_to_id = {}
            relay.analysis.post_order_visit(
                relay_mod[name],
                lambda node: self._traverse_expr(node),  # pylint: disable=unnecessary-lambda
            )
            graph = self._plotter.create_graph(name)
            # shallow copy to prevent callback modify self._node_to_id
            self._render_cb(graph, copy.copy(self._node_to_id), self._relay_param)

    def _traverse_expr(self, node):
        # based on https://github.com/apache/tvm/pull/4370
        if node in self._node_to_id:
            return
        self._node_to_id[node] = len(self._node_to_id)

    def render(self, filename):
        return self._plotter.render(filename=filename)


def get_plotter(backend):
    if backend == PlotterBackend.BOKEH:
        from ._bokeh import BokehPlotter  # pylint: disable=import-outside-toplevel

        return BokehPlotter()

    raise ValueError("Unknown plotter backend {}".format(backend))
