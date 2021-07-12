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
import copy
from tvm import relay


class PlotterBackend:
    """Enumeration for available plotters."""

    BOKEH = "bokeh"
    TERMINAL = "terminal"


class RelayVisualizer:
    """Relay IR Visualizer"""

    def __init__(
        self, relay_mod, relay_param=None, backend=PlotterBackend.BOKEH
    ):
        """Visualize Relay IR.

        Parameters
        ----------
        relay_mod : object
                        Relay IR module
        relay_param: dict
                        Relay parameter dictionary
        backend: PlotterBackend.
                        The backend of plotting. Default "bokeh"
        """

        self._plotter, self._render_cb = get_plotter_and_render_cb(backend)
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


def get_plotter_and_render_cb(backend):
    if backend == PlotterBackend.BOKEH:
        from ._bokeh import BokehPlotter, relay_render_cb  # pylint: disable=import-outside-toplevel

        return BokehPlotter(), relay_render_cb
    if backend == PlotterBackend.TERMINAL:
        from ._terminal import TermPlotter, render_cb

        return TermPlotter(), render_cb

    raise ValueError("Unknown plotter backend {}".format(backend))
