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
from typing import (
    Dict,
    Tuple,
    Union,
)
from enum import Enum
import tvm
from tvm import relay
from .plotter import Plotter
from .node_edge_gen import NodeEdgeGenerator


class PlotterBackend(Enum):
    """Enumeration for available plotter backends."""

    BOKEH = "bokeh"
    TERMINAL = "terminal"


class RelayVisualizer:
    """Relay IR Visualizer

    Parameters
    ----------
    relay_mod : tvm.IRModule
        Relay IR module.
    relay_param: None | Dict[str, tvm.runtime.NDArray]
        Relay parameter dictionary. Default `None`.
    backend: PlotterBackend | Tuple[Plotter, NodeEdgeGenerator]
        The backend used to render graphs. It can be a tuple of an implemented Plotter instance and
        NodeEdgeGenerator instance to introduce customized parsing and visualization logics.
        Default ``PlotterBackend.TERMINAL``.
    """

    def __init__(
        self,
        relay_mod: tvm.IRModule,
        relay_param: Union[None, Dict[str, tvm.runtime.NDArray]] = None,
        backend: Union[PlotterBackend, Tuple[Plotter, NodeEdgeGenerator]] = PlotterBackend.TERMINAL,
    ):

        self._plotter, self._ne_generator = get_plotter_and_generator(backend)
        self._relay_param = relay_param if relay_param is not None else {}

        global_vars = relay_mod.get_global_vars()
        graph_names = []
        # If we have main function, put it to the first.
        # Then main function can be shown on the top.
        for gv_name in global_vars:
            if gv_name.name_hint == "main":
                graph_names.insert(0, gv_name.name_hint)
            else:
                graph_names.append(gv_name.name_hint)

        node_to_id = {}

        def traverse_expr(node):
            if node in node_to_id:
                return
            node_to_id[node] = len(node_to_id)

        for name in graph_names:
            node_to_id.clear()
            relay.analysis.post_order_visit(relay_mod[name], traverse_expr)
            graph = self._plotter.create_graph(name)
            self._add_nodes(graph, node_to_id, self._relay_param)

    def _add_nodes(self, graph, node_to_id, relay_param):
        """add nodes and to the graph.

        Parameters
        ----------
        graph : plotter.Graph

        node_to_id : Dict[relay.expr, str | int]

        relay_param : Dict[str, tvm.runtime.NDarray]
        """
        for node in node_to_id:
            node_info, edge_info = self._ne_generator.get_node_edges(node, relay_param, node_to_id)
            if node_info is not None:
                graph.node(node_info.identity, node_info.type_str, node_info.detail)
            for edge in edge_info:
                graph.edge(edge.start, edge.end)

    def render(self, filename: str = None) -> None:
        self._plotter.render(filename=filename)


def get_plotter_and_generator(backend):
    """Specify the Plottor and its NodeEdgeGenerator"""
    if isinstance(backend, (tuple, list)) and len(backend) == 2:
        if not isinstance(backend[0], Plotter):
            raise ValueError(f"First element should be an instance derived from {type(Plotter)}")

        if not isinstance(backend[1], NodeEdgeGenerator):
            raise ValueError(
                f"Second element should be an instance derived from {type(NodeEdgeGenerator)}"
            )

        return backend

    if backend not in PlotterBackend:
        raise ValueError(f"Unknown plotter backend {backend}")

    # Plotter modules are Lazy-imported to avoid they become a requirement of TVM.
    # Basically we want to keep them optional. Users can choose plotters they want to install.
    if backend == PlotterBackend.BOKEH:
        # pylint: disable=import-outside-toplevel
        from .bokeh import (
            BokehPlotter,
            BokehNodeEdgeGenerator,
        )

        plotter = BokehPlotter()
        ne_generator = BokehNodeEdgeGenerator()
    elif backend == PlotterBackend.TERMINAL:
        # pylint: disable=import-outside-toplevel
        from .terminal import (
            TermPlotter,
            TermNodeEdgeGenerator,
        )

        plotter = TermPlotter()
        ne_generator = TermNodeEdgeGenerator()
    return plotter, ne_generator
