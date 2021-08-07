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
import functools

import numpy as np
import pydot

from bokeh.io import output_file, save
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.models.renderers import GraphRenderer
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Text,
    Rect,
    # NodesAndLinkedEdges,
    HoverTool,
    MultiLine,
    Legend,
    LegendItem,
    Scatter,
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

    # defined by graphviz.
    _px_per_inch = 72

    def __init__(self, pydot_graph, prog="dot", args=None):
        if args is None:
            args = []
        # call the graphviz program to get layout
        pydot_graph_str = pydot_graph.create([prog] + args, format="dot").decode()
        # parse layout
        pydot_graph = pydot.graph_from_dot_data(pydot_graph_str)
        if len(pydot_graph) != 1:
            # should be unlikely.
            _LOGGER.warning(
                "Got %d pydot graphs. Only the first one will be used.", len(pydot_graph)
            )
        self._pydot_graph = pydot_graph[0]

    @functools.lru_cache()
    def get_edge_path(self, start_node_id, end_node_id):
        """Get explicit path points for MultiLine."""
        edge = self._pydot_graph.get_edge(str(start_node_id), str(end_node_id))
        if len(edge) != 1:
            _LOGGER.warning(
                "Got %d edges between %s and %s. Only the first one will be used.",
                len(edge),
                start_node_id,
                end_node_id
            )
        edge = edge[0]
        # filter out quotes and newline
        pos_str = edge.get_pos().strip("\"").replace("\\\n", "")
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
                ss = False
                try:
                    ret_x_pts.append(float(x_str))
                except ValueError:
                    print(token)
                    print(start_node_id, end_node_id)
                    ss = True
                if ss:
                    import pdb; pdb.set_trace()
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
        return float(height_str) * self._px_per_inch

    def get_node_width(self, node_name):
        width_str = self._get_node_attr(node_name, "width", "20")
        return float(width_str) * self._px_per_inch

    def _get_node_attr(self, node_name, attr_name, default_val):

        node = self._pydot_graph.get_node(str(node_name))
        if len(node) > 1:
            _LOGGER.error("There are %d nodes with the name %s. Randomly choose one.", len(node), node_name)
        if len(node) == 0:
            _LOGGER.warning("%s does not exist in the graph. Use default %s for attribute %s", node_name, default_val, attr_name)
            return default_val

        node = node[0]
        try:
            val = node.obj_dict["attributes"][attr_name].strip("\"")
        except KeyError:
            _LOGGER.warning("%s don't exist in node %s. Use default %s", attr_name, node_name, default_val)
            val = default_val
        return val


class BokehPlotter(Plotter):
    """Use Bokeh library to plot Relay IR."""

    def __init__(self):
        self._pydot_digraph = pydot.Dot(graph_type="digraph")
        self._id_to_node = {}

    def node(self, node_id, node_type, node_detail):
        # need string for pydot
        node_id = str(node_id)
        if node_id in self._id_to_node:
            _LOGGER.warning("node_id %s already exists.", node_id)
            return
        self._pydot_digraph.add_node(pydot.Node(node_id))
        self._id_to_node[node_id] = NodeDescriptor(node_id, node_type, node_detail)

    def edge(self, id_start, id_end):
        # need string to pydot
        id_start, id_end = str(id_start), str(id_end)
        self._pydot_digraph.add_edge(pydot.Edge(id_start, id_end))

    def render(self, filename):

        if filename.endswith(".html"):
            graph_name = os.path.splitext(os.path.basename(filename))[0]
        else:
            graph_name = filename
            filename = "{}.html".format(filename)

        plot = Plot(
            title=graph_name,
            width=1600,
            height=900,
            align="center",
            sizing_mode="stretch_both",
            margin=(0, 0, 0, 50),
        )

        layout_dom = self._create_layout_dom(plot)
        self._save_html(filename, layout_dom)

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

    def _add_scalable_glyph(self, plot, node_to_pos, shaper):

        text_source = ColumnDataSource(
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
            text_font_size={"value": "11px"},
        )
        node_annotation = plot.add_glyph(text_source, text_glyph)

        def get_scatter_loc(xs, xe, ys, ye, end_node):
            """return x, y, angle as a tuple"""
            node_x, node_y = node_to_pos[end_node]
            node_w = shaper.get_node_width(end_node)
            node_h = shaper.get_node_height(end_node)

            # only 4 direction
            if xe - xs > 0:
                return node_x - node_w / 2, ye, -np.pi / 2
            if xe - xs < 0:
                return node_x + node_w / 2, ye, np.pi / 2
            if ye - ys < 0:
                return xe, node_y + node_h / 2, np.pi
            return xe, node_y - node_h / 2, 0

        scatter_source = {"x": [], "y": [], "angle": []}
        for edge in self._pydot_digraph.get_edges():
            id_start = edge.get_source()
            id_end = edge.get_destination()
            x_pts, y_pts = shaper.get_edge_path(id_start, id_end)
            x, y, angle = get_scatter_loc(x_pts[-2], x_pts[-1], y_pts[-2], y_pts[-1], id_end)
            scatter_source["angle"].append(angle)
            scatter_source["x"].append(x)
            scatter_source["y"].append(y)

        scatter_glyph = Scatter(
            x="x",
            y="y",
            angle="angle",
            size=5,
            marker="triangle",
            fill_color="#AAAAAA",
            fill_alpha=0.8,
        )
        edge_end_arrow = plot.add_glyph(ColumnDataSource(scatter_source), scatter_glyph)

        plot.x_range.js_on_change(
            "start",
            CustomJS(
                args=dict(
                    plot=plot, node_annotation=node_annotation, edge_end_arrow=edge_end_arrow
                ),
                code="""
                 node_annotation.visible = plot.width/(this.end - this.start) >= 1.0
                 var ratio = Math.sqrt(plot.width/(this.end - this.start))

                 var new_text_size = Math.round(11 * ratio)
                 node_annotation.glyph.text_font_size = {value: `${new_text_size}px`}

                 var new_scatter_size = Math.round(5 * ratio)
                 edge_end_arrow.glyph.size = {value: new_scatter_size}
                 """,
            ),
        )

    def _create_graph(self, plot, shaper, node_to_pos):

        graph = GraphRenderer()
        graph.name = self._get_graph_name(plot)
        # FIXME: handle node attributes if necessary
        graph.node_renderer.data_source.data = {
            "index": [n.get_name() for n in self._pydot_digraph.get_nodes()]
        }

        edges = self._pydot_digraph.get_edges()
        graph.edge_renderer.data_source.data = {
            "start": [e.get_source() for e in edges],
            "end": [e.get_destination() for e in edges]   
        }

        graph.layout_provider = StaticLayoutProvider(graph_layout=node_to_pos)
        
        # TODO: I want to plot the network with lower-level bokeh APIs in the future,
        # which may not support NodesAndLinkedEdges() policy. So comment out here.
        # graph.selection_policy = NodesAndLinkedEdges()

        # edge
        edge_line_color = "#888888"
        edge_line_width = 3
        graph.edge_renderer.glyph = MultiLine(
            line_color=edge_line_color, line_width=edge_line_width
        )
        x_path_list = []
        y_path_list = []
        for edge in edges:
            id_start = edge.get_source()
            id_end = edge.get_destination()
            x_pts, y_pts = shaper.get_edge_path(id_start, id_end)
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
        rect_w = [shaper.get_node_width(n) for n in node_to_pos]
        rect_h = [shaper.get_node_height(n) for n in node_to_pos]

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

        shaper = GraphShaper(
            self._pydot_digraph, prog="dot", args=["-Grankdir=TB", "-Gsplines=ortho", "-Nordering=in"]
        )

        node_to_pos = {}
        for node in self._pydot_digraph.get_nodes():
            node_name = node.get_name()
            node_to_pos[node_name] = shaper.get_node_pos(node_name)

        graph = self._create_graph(plot, shaper, node_to_pos)

        self._add_legend(plot, graph, {"field": "label"})

        self._add_tooltip(plot)

        # add graph
        plot.renderers.append(graph)

        self._add_scalable_glyph(plot, node_to_pos, shaper)
        return plot

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
