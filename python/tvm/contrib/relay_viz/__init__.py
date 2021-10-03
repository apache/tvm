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
    """Enumeration for available plotters."""

    BOKEH = "bokeh"
    TERMINAL = "terminal"


class RelayVisualizer:
    """Relay IR Visualizer"""

    def __init__(
        self,
        relay_mod: tvm.IRModule,
        relay_param: Dict = None,
        backend: Union[PlotterBackend, Tuple[Plotter, NodeEdgeGenerator]] = PlotterBackend.TERMINAL,
    ):
        """Visualize Relay IR.

        Parameters
        ----------
        relay_mod : object
                        Relay IR module
        relay_param: dict
                        Relay parameter dictionary
        backend: PlotterBackend or a tuple
                        PlotterBackend: The backend of plotting. Default "terminal"
                        Tuple: A tuple with two arguments. First is user-defined Plotter,
                               the second is user-defined NodeEdgeGenerator
        """

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
        for name in graph_names:

            def traverse_expr(node):
                if node in node_to_id:
                    return
                node_to_id[node] = len(node_to_id)

            relay.analysis.post_order_visit(relay_mod[name], traverse_expr)
            graph = self._plotter.create_graph(name)
            self._render(graph, node_to_id, self._relay_param)
            node_to_id.clear()

    def _render(self, graph, node_to_id, relay_param):
        """render nodes and edges to the graph.

        Parameters
        ----------
        graph : class plotter.Graph

        node_to_id : Dict[relay.expr, int]

        relay_param : Dict[string, NDarray]
        """
        for node in node_to_id:
            graph_info, edge_info = self._ne_generator.get_node_edges(node, relay_param, node_to_id)
            if graph_info:
                graph.node(*graph_info)
            for edge in edge_info:
                graph.edge(*edge)

    def render(self, filename: str = None) -> None:
        self._plotter.render(filename=filename)


def get_plotter_and_generator(backend):
    """Specify the Plottor and its NodeEdgeGenerator"""
    if isinstance(backend, (tuple, list)) and len(backend) == 2:
        if not isinstance(backend[0], Plotter):
            raise ValueError(f"First element of backend should be derived from {type(Plotter)}")

        if not isinstance(backend[1], NodeEdgeGenerator):
            raise ValueError(
                f"Second element of backend should be derived from {type(NodeEdgeGenerator)}"
            )

        return backend

    if backend not in PlotterBackend:
        raise ValueError(f"Unknown plotter backend {backend}")

    # Plotter modules are Lazy-imported to avoid they become a requirement of TVM.
    # Basically we want to keep them as optional -- users can choose which plotter they want,
    # and just install libraries required by that plotter.
    if backend == PlotterBackend.BOKEH:
        # pylint: disable=import-outside-toplevel
        from ._bokeh import (
            BokehPlotter,
            BokehNodeEdgeGenerator,
        )

        plotter = BokehPlotter()
        ne_generator = BokehNodeEdgeGenerator()
    elif backend == PlotterBackend.TERMINAL:
        # pylint: disable=import-outside-toplevel
        from ._terminal import (
            TermPlotter,
            TermNodeEdgeGenerator,
        )

        plotter = TermPlotter()
        ne_generator = TermNodeEdgeGenerator()
    return plotter, ne_generator
