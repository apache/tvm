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
from typing import (
    Dict,
    Tuple,
    Union,
)
from enum import Enum
import tvm
from tvm import relay
from .plotter import Plotter
from .render_callback import (
    RenderCallbackInterface,
    RenderCallback,
)

_LOGGER = logging.getLogger(__name__)


class PlotterBackend(Enum):
    """Enumeration for available plotters."""

    BOKEH = "bokeh"
    TERMINAL = "terminal"


class RelayVisualizer:
    """Relay IR Visualizer"""

    def __init__(self,
                 relay_mod: tvm.IRModule,
                 relay_param: Dict = None,
                 backend: Union[PlotterBackend, Tuple[Plotter, RenderCallbackInterface]] = PlotterBackend.TERMINAL):
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
                               the second is user-defined RenderCallback
        """

        self._plotter, self._render_rules = get_plotter_and_render_rules(backend)
        self._relay_param = relay_param if relay_param is not None else {}

        global_vars = relay_mod.get_global_vars()
        graph_names = []
        # If we have main function, put it to the first.
        # Then main function can be shown at the top.
        for gv_name in global_vars:
            if gv_name.name_hint == "main":
                graph_names.insert(0, gv_name.name_hint)
            else:
                graph_names.append(gv_name.name_hint)

        for name in graph_names:
            node_to_id = {}
            def traverse_expr(node):
                if node in node_to_id:
                    return
                node_to_id[node] = len(node_to_id)

            relay.analysis.post_order_visit(relay_mod[name], traverse_expr)
            graph = self._plotter.create_graph(name)
            # shallow copy to prevent callback modify node_to_id
            self._render_cb(graph, node_to_id.copy(), self._relay_param)

    def _render_cb(self, graph, node_to_id, relay_param):
        """a callback to Add nodes and edges to the graph.

        Parameters
        ----------
        graph : class plotter.Graph

        node_to_id : Dict[relay.expr, int]

        relay_param : Dict[string, NDarray]
        """
        for node, node_id in node_to_id.items():
            try:
                graph_info, edge_info = self._render_rules[type(node)](
                    node, relay_param, node_to_id
                )
                if graph_info:
                    graph.node(*graph_info)
                for edge in edge_info:
                    graph.edge(*edge)
            except KeyError as excp:
                unknown_type = "unknown"
                unknown_info = f"Failed to render node: {type(node)}"
                _LOGGER.warning("When invoking render rule for %s, "
                                "KeyError with args=%s is raised.",
                                type(node),
                                excp.args,
                )
                graph.node(node_id, unknown_type, unknown_info)

    def render(self, filename: str) -> None:
        self._plotter.render(filename=filename)


def get_plotter_and_render_rules(backend):
    """Specify the Plottor and its render rules

    Parameters
        ----------
        backend: PlotterBackend or a tuple
                        PlotterBackend: The backend of plotting. Default "bokeh"
                        Tuple: A tuple with two arguments. First is user-defined Plotter, \
                               the second is user-defined RenderCallback
    """
    if isinstance(backend, (tuple, list)) and len(backend) == 2:
        if not isinstance(backend[0], Plotter):
            raise ValueError(f"First elemnet of backend argument should be derived from {type(Plotter)}")
        plotter = backend[0]
        if not isinstance(backend[1], RenderCallback):
            raise ValueError(f"Second elemnet of backend argument should be derived from {type(RenderCallbackInterface)}")
        render = backend[1]
        render_rules = render.get_rules()
        return plotter, render_rules

    if backend in PlotterBackend:
        # Plotter modules are Lazy-imported to avoid they become a requirement of TVM.
        # Basically we want to keep them as optional -- users can choose which plotter they want,
        # and just install libraries required by that plotter.
        if backend == PlotterBackend.BOKEH:
            # pylint: disable=import-outside-toplevel
            from ._bokeh import (
                BokehPlotter,
                BokehRenderCallback,
            )

            plotter = BokehPlotter()
            render = BokehRenderCallback()

        elif backend == PlotterBackend.TERMINAL:
            # pylint: disable=import-outside-toplevel
            from ._terminal import (
                TermPlotter,
                TermRenderCallback,
            )

            plotter = TermPlotter()
            render = TermRenderCallback()

        render_rules = render.get_rules()
        return plotter, render_rules

    raise ValueError(f"Unknown plotter backend {backend}")
