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
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Text,
    Rect,
    HoverTool,
    MultiLine,
    Legend,
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
from bokeh.layouts import column

from .plotter import (
    Plotter,
    Graph,
)

import tvm
from tvm import relay

_LOGGER = logging.getLogger(__name__)


def relay_render_cb(graph, node_to_id, relay_param):
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
            func_attrs = node.attrs
            if func_attrs:
                node_details = [
                    "{}: {}".format(k, func_attrs.get_str(k)) for k in func_attrs.keys()
                ]

            graph.node(node_id, f"Func", "\n".join(node_details))
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
        return self._node_detail


class GraphShaper:
    """Provide the bounding-box, and node location, height, width given by pygraphviz."""

    # defined by graphviz.
    _px_per_inch = 72

    def __init__(self, pydot_graph, prog="dot", args=None):
        if args is None:
            args = []
        # call the graphviz program to get layout
        pydot_graph_str = pydot_graph.create([prog] + args, format="dot").decode()
        # remember original nodes
        self._nodes = [n.get_name() for n in pydot_graph.get_nodes()]
        # parse layout
        pydot_graph = pydot.graph_from_dot_data(pydot_graph_str)
        if len(pydot_graph) != 1:
            # should be unlikely.
            _LOGGER.warning(
                "Got %d pydot graphs. Only the first one will be used.", len(pydot_graph)
            )
        self._pydot_graph = pydot_graph[0]

    def get_nodes(self):
        return self._nodes

    @functools.lru_cache()
    def get_edge_path(self, start_node_id, end_node_id):
        """Get explicit path points for MultiLine."""
        edge = self._pydot_graph.get_edge(str(start_node_id), str(end_node_id))
        if len(edge) != 1:
            _LOGGER.warning(
                "Got %d edges between %s and %s. Only the first one will be used.",
                len(edge),
                start_node_id,
                end_node_id,
            )
        edge = edge[0]
        # filter out quotes and newline
        pos_str = edge.get_pos().strip('"').replace("\\\n", "")
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

    @functools.lru_cache()
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
            _LOGGER.error(
                "There are %d nodes with the name %s. Randomly choose one.", len(node), node_name
            )
        if len(node) == 0:
            _LOGGER.warning(
                "%s does not exist in the graph. Use default %s for attribute %s",
                node_name,
                default_val,
                attr_name,
            )
            return default_val

        node = node[0]
        try:
            val = node.obj_dict["attributes"][attr_name].strip('"')
        except KeyError:
            _LOGGER.warning(
                "%s don't exist in node %s. Use default %s", attr_name, node_name, default_val
            )
            val = default_val
        return val


class BokehGraph(Graph):
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
        self._pydot_digraph.add_node(pydot.Node(node_id, label=node_detail))
        self._id_to_node[node_id] = NodeDescriptor(node_id, node_type, node_detail)

    def edge(self, id_start, id_end):
        # need string to pydot
        id_start, id_end = str(id_start), str(id_end)
        self._pydot_digraph.add_edge(pydot.Edge(id_start, id_end))

    def render(self, plot):

        shaper = GraphShaper(
            self._pydot_digraph,
            prog="dot",
            args=["-Grankdir=TB", "-Gsplines=ortho", "-Gfontsize=14", "-Nordering=in"],
        )

        self._create_graph(plot, shaper)

        self._add_scalable_glyph(plot, shaper)
        return plot

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

    def _create_graph(self, plot, shaper):

        # Add edge first
        edges = self._pydot_digraph.get_edges()
        x_path_list = []
        y_path_list = []
        for edge in edges:
            id_start = edge.get_source()
            id_end = edge.get_destination()
            x_pts, y_pts = shaper.get_edge_path(id_start, id_end)
            x_path_list.append(x_pts)
            y_path_list.append(y_pts)

        multi_line_source = ColumnDataSource({"xs": x_path_list, "ys": y_path_list})
        edge_line_color = "#888888"
        edge_line_width = 3
        multi_line_glyph = MultiLine(line_color=edge_line_color, line_width=edge_line_width)
        plot.add_glyph(multi_line_source, multi_line_glyph)

        # Then add nodes
        type_to_color = self._get_type_to_color_map()

        def cnvt_to_html(s):
            return html.escape(s).replace("\n", "<br>")

        label_to_ids = {}
        for node_id in shaper.get_nodes():
            label = self._id_to_node[node_id].node_type
            if label not in label_to_ids:
                label_to_ids[label] = []
            label_to_ids[label].append(node_id)

        renderers = []
        legend_itmes = []
        for label, id_list in label_to_ids.items():
            source = ColumnDataSource(
                {
                    "x": [shaper.get_node_pos(n)[0] for n in id_list],
                    "y": [shaper.get_node_pos(n)[1] for n in id_list],
                    "width": [shaper.get_node_width(n) for n in id_list],
                    "height": [shaper.get_node_height(n) for n in id_list],
                    "node_detail": [cnvt_to_html(self._id_to_node[n].detail) for n in id_list],
                    "node_type": [label] * len(id_list),
                }
            )
            glyph = Rect(fill_color=type_to_color[label])
            renderer = plot.add_glyph(source, glyph)
            # set glyph for interactivity
            renderer.nonselection_glyph = Rect(fill_color=type_to_color[label])
            renderer.hover_glyph = Rect(
                fill_color=type_to_color[label], line_color="firebrick", line_width=3
            )
            renderer.selection_glyph = Rect(
                fill_color=type_to_color[label], line_color="firebrick", line_width=3
            )
            # Though it is called "muted_glyph", we actually use it
            # to emphasize nodes in this renderer.
            renderer.muted_glyph = Rect(
                fill_color=type_to_color[label], line_color="firebrick", line_width=3
            )
            name = f"{self._get_graph_name(plot)}_{label}"
            renderer.name = name
            renderers.append(renderer)
            legend_itmes.append((label, [renderer]))

        # add legend
        legend = Legend(
            items=legend_itmes,
            title="Click to highlight",
            inactive_fill_color="firebrick",
            inactive_fill_alpha=0.2,
        )
        legend.click_policy = "mute"
        legend.location = "top_right"
        plot.add_layout(legend)

        # add tooltips
        tooltips = [
            ("node_type", "@node_type"),
            ("description", "@node_detail{safe}"),
        ]
        inspect_tool = WheelZoomTool()
        # only render nodes
        hover_tool = HoverTool(tooltips=tooltips, renderers=renderers)
        plot.add_tools(PanTool(), TapTool(), inspect_tool, hover_tool, ResetTool(), SaveTool())
        plot.toolbar.active_scroll = inspect_tool

    def _add_scalable_glyph(self, plot, shaper):
        nodes = shaper.get_nodes()

        def populate_detail(n_type, n_detail):
            if n_detail:
                return f"{n_type}\n{n_detail}"
            return n_type

        text_source = ColumnDataSource(
            {
                "x": [shaper.get_node_pos(n)[0] for n in nodes],
                "y": [shaper.get_node_pos(n)[1] for n in nodes],
                "text": [self._id_to_node[n].node_type for n in nodes],
                "detail": [
                    populate_detail(self._id_to_node[n].node_type, self._id_to_node[n].detail)
                    for n in nodes
                ],
                "box_w": [shaper.get_node_width(n) for n in nodes],
                "box_h": [shaper.get_node_height(n) for n in nodes],
            }
        )

        text_glyph = Text(
            x="x",
            y="y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_font_size={"value": "14px"},
        )
        node_annotation = plot.add_glyph(text_source, text_glyph)

        def get_scatter_loc(x_start, x_end, y_start, y_end, end_node):
            """return x, y, angle as a tuple"""
            node_x, node_y = shaper.get_node_pos(end_node)
            node_w = shaper.get_node_width(end_node)
            node_h = shaper.get_node_height(end_node)

            # only 4 direction
            if x_end - x_start > 0:
                return node_x - node_w / 2, y_end, -np.pi / 2
            if x_end - x_start < 0:
                return node_x + node_w / 2, y_end, np.pi / 2
            if y_end - y_start < 0:
                return x_end, node_y + node_h / 2, np.pi
            return x_end, node_y - node_h / 2, 0

        scatter_source = {"x": [], "y": [], "angle": []}
        for edge in self._pydot_digraph.get_edges():
            id_start = edge.get_source()
            id_end = edge.get_destination()
            x_pts, y_pts = shaper.get_edge_path(id_start, id_end)
            x_loc, y_loc, angle = get_scatter_loc(x_pts[-2], x_pts[-1], y_pts[-2], y_pts[-1], id_end)
            scatter_source["angle"].append(angle)
            scatter_source["x"].append(x_loc)
            scatter_source["y"].append(y_loc)

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

        plot.y_range.js_on_change(
            "start",
            CustomJS(
                args=dict(
                    plot=plot,
                    node_annotation=node_annotation,
                    text_source=text_source,
                    edge_end_arrow=edge_end_arrow,
                ),
                code="""
                 // fontsize is in px
                 var fontsize = 14
                 // ratio = data_point/px
                 var ratio = (this.end - this.start)/plot.height
                 var text_list = text_source.data["text"]
                 var detail_list = text_source.data["detail"]
                 var box_h_list = text_source.data["box_h"]
                 for(var i = 0; i < text_list.length; i++) {
                     var line_num = Math.floor((box_h_list[i]/ratio) / (fontsize*1.5))
                     if(line_num <= 0) {
                         // relieve for the first line
                         if(Math.floor((box_h_list[i]/ratio) / (fontsize)) > 0) {
                            line_num = 1
                         }
                     }
                     var lines = detail_list[i].split("\\n")
                     lines = lines.slice(0, line_num)
                     text_list[i] = lines.join("\\n")
                 }
                 text_source.change.emit()

                 node_annotation.glyph.text_font_size = {value: `${fontsize}px`}

                 var new_scatter_size = Math.round(fontsize / ratio)
                 edge_end_arrow.glyph.size = {value: new_scatter_size}
                 """,
            ),
        )

    @staticmethod
    def _get_graph_name(plot):
        return plot.title


class BokehPlotter(Plotter):
    """Use Bokeh library to plot Relay IR."""

    def __init__(self):
        self._name_to_graph = {}

    def create_graph(self, name):
        if name in self._name_to_graph:
            _LOGGER.warning("Graph name %s exists. ")
        else:
            self._name_to_graph[name] = BokehGraph()
        return self._name_to_graph[name]

    def render(self, filename):

        if filename.endswith(".html"):
            graph_name = os.path.splitext(os.path.basename(filename))[0]
        else:
            graph_name = filename
            filename = "{}.html".format(filename)

        dom_list = []
        for name, graph in self._name_to_graph.items():
            plot = Plot(
                title=name,
                width=1600,
                height=900,
                align="center",
                margin=(0, 0, 0, 70),
            )

            dom = graph.render(plot)
            dom_list.append(dom)

        self._save_html(filename, column(*dom_list))

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

