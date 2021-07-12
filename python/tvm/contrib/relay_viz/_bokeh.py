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
"""Bokeh backend for Relay IR Visualizer."""
import os
import html
import logging

import networkx as nx

from bokeh.io import output_file, save
from bokeh.plotting import from_networkx
from bokeh.layouts import (
    layout,
    Spacer,
)
from bokeh.models import (
    ColumnDataSource,
    Text,
    Rect,
    # NodesAndLinkedEdges,
    HoverTool,
    MultiLine,
    Legend,
    LegendItem,
    Toggle,
    Plot,
    TapTool,
    PanTool,
    ResetTool,
    WheelZoomTool,
    SaveTool,
)
from bokeh.palettes import (
    d3,
)

from .plotter import Plotter

_LOGGER = logging.getLogger(__name__)


class NodeDescriptor:
    """Descriptor used by Bokeh plotter."""

    def __init__(self, node_id, node_type, node_detail):
        self._node_id = node_id
        self._node_type = node_type
        self._node_detail = node_detail

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type(self):
        return self._node_type

    @property
    def detail(self):
        ret = html.escape(self._node_detail)
        ret = ret.replace("\n", "<br>")
        return ret


class GraphShaper:
    """Provide the bounding-box, and node location, height, width given by pygraphviz."""

    def __init__(self, nx_digraph, prog="neato", args=""):
        agraph = nx.nx_agraph.to_agraph(nx_digraph)
        agraph.layout(prog=prog, args=args)
        self._agraph = agraph

    def get_edge_path(self, start_node_id, end_node_id):
        """Get explicit path points for MultiLine."""
        edge = self._agraph.get_edge(start_node_id, end_node_id)
        pos_str = edge.attr["pos"]
        tokens = pos_str.split(" ")
        s_token = None
        e_token = None
        ret_x_pts = []
        ret_y_pts = []
        for token in tokens:
            if token.startswith("e,"):
                e_token = token
            elif token.startswith("s,"):
                s_token = token
            else:
                x_str, y_str = token.split(",")
                ret_x_pts.append(float(x_str))
                ret_y_pts.append(float(y_str))
        if s_token is not None:
            _, x_str, y_str = s_token.split(",")
            ret_x_pts.insert(0, float(x_str))
            ret_y_pts.insert(0, float(y_str))
        if e_token is not None:
            _, x_str, y_str = e_token.split(",")
            ret_x_pts.append(float(x_str))
            ret_y_pts.append(float(y_str))

        return ret_x_pts, ret_y_pts

    def get_node_pos(self, node_name):
        pos_str = self._get_node_attr(node_name, "pos", "0,0")
        return list(map(float, pos_str.split(",")))

    def get_node_height(self, node_name):
        height_str = self._get_node_attr(node_name, "height", "20")
        return float(height_str)

    def get_node_width(self, node_name):
        width_str = self._get_node_attr(node_name, "width", "20")
        return float(width_str)

    def _get_node_attr(self, node_name, attr_name, default_val):
        try:
            attr = self._agraph.get_node(node_name).attr
        except KeyError:
            _LOGGER.warning("%s does not exist in the graph.", node_name)
            return default_val

        try:
            val = attr[attr_name]
        except KeyError:
            _LOGGER.warning(
                "%s does not exist in node %s. Use default %s", attr_name, node_name, default_val
            )
        return val


class BokehPlotter(Plotter):
    """Use Bokeh library to plot Relay IR."""

    def __init__(self):
        self._digraph = nx.DiGraph()
        self._id_to_node = {}
        # for pending edge...
        self._pending_id_to_edges = {}

    def node(self, node_id, node_type, node_detail):
        if node_id in self._id_to_node:
            _LOGGER.warning("node_id %s already exists.", node_id)
            return

        self._digraph.add_node(node_id)
        self._id_to_node[node_id] = NodeDescriptor(node_id, node_type, node_detail)

        self._add_pending_edge(node_id)

    def edge(self, id_start, id_end):
        if id_start in self._id_to_node and id_end in self._id_to_node:
            self._edge(id_start, id_end)
            return

        if id_start not in self._id_to_node:
            try:
                self._pending_id_to_edges[id_start].add((id_start, id_end))
            except KeyError:
                self._pending_id_to_edges[id_start] = set([(id_start, id_end)])
        if id_end not in self._id_to_node:
            try:
                self._pending_id_to_edges[id_end].add((id_start, id_end))
            except KeyError:
                self._pending_id_to_edges[id_end] = set([(id_start, id_end)])

    def render(self, filename):

        if filename.endswith(".html"):
            graph_name = os.path.splitext(os.path.basename(filename))[0]
        else:
            graph_name = filename
            filename = "{}.html".format(filename)

        plot = Plot(
            title=graph_name,
            plot_width=1600,
            plot_height=900,
            align="center",
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 50),
        )

        layout_dom = self._create_layout_dom(plot)
        self._save_html(filename, layout_dom)

    def _edge(self, id_start, id_end):
        self._digraph.add_edge(id_start, id_end)

    def _add_pending_edge(self, node_id):
        added_edges = set()
        if node_id in self._pending_id_to_edges:
            for id_start, id_end in self._pending_id_to_edges[node_id]:
                if id_start in self._id_to_node and id_end in self._id_to_node:
                    self._edge(id_start, id_end)
                    added_edges.add((id_start, id_end))

        # fix pending_id_to_edges
        for id_start, id_end in added_edges:
            self._pending_id_to_edges[id_start].discard((id_start, id_end))
            self._pending_id_to_edges[id_end].discard((id_start, id_end))

    def _get_type_to_color_map(self):
        category20 = d3["Category20"][20]
        # FIXME: a problem is, for different network we have different color
        # for the same type.
        all_types = list({v.node_type for v in self._id_to_node.values()})
        all_types.sort()
        if len(all_types) > 20:
            _LOGGER.warning(
                "The number of types %d is larger than 20. "
                "Some colors are re-used for different types.",
                len(all_types),
            )
        type_to_color = {}
        for idx, t in enumerate(all_types):
            type_to_color[t] = category20[idx % 20]
        return type_to_color

    def _add_legend(self, plot, graph, label):
        legend_item = LegendItem(label=label, renderers=[graph.node_renderer])
        legend = Legend(items=[legend_item], title="Node Type")
        plot.add_layout(legend, "left")

    def _add_tooltip(self, plot):

        graph_name = self._get_graph_name(plot)
        tooltips = [
            ("node_type", "@label"),
            ("description", "@node_detail{safe}"),
        ]
        inspect_tool = WheelZoomTool()
        # only render graph_name
        hover_tool = HoverTool(tooltips=tooltips, names=[graph_name])
        plot.add_tools(PanTool(), TapTool(), inspect_tool, hover_tool, ResetTool(), SaveTool())
        plot.toolbar.active_scroll = inspect_tool

    def _create_node_type_toggler(self, plot, node_to_pos):

        source = ColumnDataSource(
            {
                "x": [pos[0] for pos in node_to_pos.values()],
                "y": [pos[1] for pos in node_to_pos.values()],
                "text": [self._id_to_node[n].node_type for n in node_to_pos],
            }
        )

        text_glyph = Text(
            x="x",
            y="y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_font_size={"value": "1.5em"},
        )
        node_annotation = plot.add_glyph(source, text_glyph, visible=False)

        # widgets
        toggler = Toggle(label="Toggle Type", align="end", default_size=100, button_type="primary")
        toggler.js_link("active", node_annotation, "visible")
        return toggler

    def _create_graph(self, plot, shaper, node_to_pos):

        graph = from_networkx(self._digraph, node_to_pos)
        graph.name = self._get_graph_name(plot)

        # TODO: I want to plot the network with lower-level bokeh APIs in the future,
        # which may not support NodesAndLinkedEdges() policy. So comment out here.
        # graph.selection_policy = NodesAndLinkedEdges()

        # edge
        edge_line_width = 3
        graph.edge_renderer.glyph = MultiLine(line_color="#888888", line_width=edge_line_width)
        x_path_list = []
        y_path_list = []
        for edge in self._digraph.edges():
            x_pts, y_pts = shaper.get_edge_path(edge[0], edge[1])
            x_path_list.append(x_pts)
            y_path_list.append(y_pts)
        graph.edge_renderer.data_source.data["xs"] = x_path_list
        graph.edge_renderer.data_source.data["ys"] = y_path_list

        # node
        graph.node_renderer.glyph = Rect(width="w", height="h", fill_color="fill_color")
        graph.node_renderer.hover_glyph = Rect(
            width="w", height="h", fill_color="fill_color", line_color="firebrick", line_width=3
        )
        graph.node_renderer.selection_glyph = Rect(
            width="w", height="h", fill_color="fill_color", line_color="firebrick", line_width=3
        )
        graph.node_renderer.nonselection_glyph = Rect(
            width="w", height="h", fill_color="fill_color"
        )

        # decide rect size
        px_per_inch = 72
        rect_w = [shaper.get_node_width(n) * px_per_inch for n in node_to_pos]
        rect_h = [shaper.get_node_height(n) * px_per_inch for n in node_to_pos]

        # get type-color map
        type_to_color = self._get_type_to_color_map()

        # add data source for nodes
        graph.node_renderer.data_source.data = dict(
            index=list(node_to_pos.keys()),
            w=rect_w,
            h=rect_h,
            label=[self._id_to_node[i].node_type for i in node_to_pos],
            fill_color=[type_to_color[self._id_to_node[i].node_type] for i in node_to_pos],
            node_detail=[self._id_to_node[i].detail for i in node_to_pos],
        )

        return graph

    def _create_layout_dom(self, plot):

        shaper = GraphShaper(self._digraph, prog="dot", args="-Grankdir=TB -Gsplines=ortho")

        node_to_pos = {}
        for node in self._digraph:
            node_to_pos[node] = shaper.get_node_pos(node)

        graph = self._create_graph(plot, shaper, node_to_pos)

        self._add_legend(plot, graph, {"field": "label"})

        self._add_tooltip(plot)

        # add graph
        plot.renderers.append(graph)

        node_type_toggler = self._create_node_type_toggler(plot, node_to_pos)

        layout_dom = layout(
            [
                [Spacer(sizing_mode="stretch_width"), node_type_toggler],
                [plot],
            ]
        )
        return layout_dom

    def _save_html(self, filename, layout_dom):

        output_file(filename, title=filename)

        template = """
        {% block postamble %}
        <style>
        .bk-root .bk {
            margin: 0 auto !important;
        }
        </style>
        {% endblock %}
        """

        save(layout_dom, filename=filename, title=filename, template=template)

    @staticmethod
    def _get_graph_name(plot):
        return plot.title
