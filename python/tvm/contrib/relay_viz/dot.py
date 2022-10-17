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
"""Visualize Relay IR by Graphviz DOT language."""

from typing import (
    Any,
    Callable,
    Dict,
)
from .interface import (
    DefaultVizParser,
    Plotter,
    VizEdge,
    VizGraph,
    VizNode,
)

try:
    import graphviz
except ImportError:
    # add "from None" to silence
    # "During handling of the above exception, another exception occurred"
    raise ImportError(
        "The graphviz package is required for DOT renderer. "
        "Please install it first. For example, pip3 install graphviz"
    ) from None

DotVizParser = DefaultVizParser


class DotGraph(VizGraph):
    """DOT graph for relay IR.

    See also :py:class:`tvm.contrib.relay_viz.dot.DotPlotter`

    Parameters
    ----------
    name: str
        name of this graph.
    graph_attr: Optional[Dict[str, str]]
        key-value pairs for the graph.
    node_attr: Optional[Dict[str, str]]
        key-value pairs for all nodes.
    edge_attr: Optional[Dict[str, str]]
        key-value pairs for all edges.
    get_node_attr: Optional[Callable[[VizNode], Dict[str, str]]]
        A callable returning attributes for the node.
    """

    def __init__(
        self,
        name: str,
        graph_attr: Dict[str, str] = None,
        node_attr: Dict[str, str] = None,
        edge_attr: Dict[str, str] = None,
        get_node_attr: Callable[[VizNode], Dict[str, str]] = None,
    ):
        self._name = name
        self._get_node_attr = self._default_get_node_attr
        if get_node_attr is not None:
            self._get_node_attr = get_node_attr

        # graphviz recognizes the subgraph as a cluster subgraph
        # by the name starting with "cluster" (all lowercase)
        self._digraph = graphviz.Digraph(
            name=f"cluster_{self._name}",
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )
        self._digraph.attr(label=self._name)

    def node(self, viz_node: VizNode) -> None:
        """Add a node to the underlying graph.
        Nodes in a Relay IR Module are expected to be added in the post-order.

        Parameters
        ----------
        viz_node : VizNode
            A `VizNode` instance.
        """
        self._digraph.node(
            viz_node.identity,
            f"{viz_node.type_name}\n{viz_node.detail}",
            **self._get_node_attr(viz_node),
        )

    def edge(self, viz_edge: VizEdge) -> None:
        """Add an edge to the underlying graph.

        Parameters
        ----------
        viz_edge : VizEdge
            A `VizEdge` instance.
        """
        self._digraph.edge(viz_edge.start, viz_edge.end)

    @property
    def digraph(self):
        return self._digraph

    @staticmethod
    def _default_get_node_attr(node: VizNode):
        if "Var" in node.type_name:
            return {"shape": "ellipse"}
        return {"shape": "box"}


class DotPlotter(Plotter):
    """DOT language graph plotter

    The plotter accepts various graphviz attributes for graphs, nodes, and edges.
    Please refer to https://graphviz.org/doc/info/attrs.html for available attributes.

    Parameters
    ----------
    graph_attr: Optional[Dict[str, str]]
        key-value pairs for all graphs.
    node_attr: Optional[Dict[str, str]]
        key-value pairs for all nodes.
    edge_attr: Optional[Dict[str, str]]
        key-value pairs for all edges.
    get_node_attr: Optional[Callable[[VizNode], Dict[str, str]]]
        A callable returning attributes for a specific node.
    render_kwargs: Optional[Dict[str, Any]]
        keyword arguments directly passed to `graphviz.Digraph.render()`.

    Examples
    --------

    .. code-block:: python

        from tvm.contrib import relay_viz
        from tvm.relay.testing import resnet

        mod, param = resnet.get_workload(num_layers=18)
        # graphviz attributes
        graph_attr = {"color": "red"}
        node_attr = {"color": "blue"}
        edge_attr = {"color": "black"}

        # VizNode is passed to the callback.
        # We want to color NCHW conv2d nodes. Also give Var a different shape.
        def get_node_attr(node):
            if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
                return {
                    "fillcolor": "green",
                    "style": "filled",
                    "shape": "box",
                }
            if "Var" in node.type_name:
                return {"shape": "ellipse"}
            return {"shape": "box"}

        # Create plotter and pass it to viz. Then render the graph.
        dot_plotter = relay_viz.DotPlotter(
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
            get_node_attr=get_node_attr)

        viz = relay_viz.RelayVisualizer(
            mod,
            relay_param=param,
            plotter=dot_plotter,
            parser=relay_viz.DotVizParser())
        viz.render("hello")
    """

    def __init__(
        self,
        graph_attr: Dict[str, str] = None,
        node_attr: Dict[str, str] = None,
        edge_attr: Dict[str, str] = None,
        get_node_attr: Callable[[VizNode], Dict[str, str]] = None,
        render_kwargs: Dict[str, Any] = None,
    ):
        self._name_to_graph = {}
        self._graph_attr = graph_attr
        self._node_attr = node_attr
        self._edge_attr = edge_attr
        self._get_node_attr = get_node_attr

        self._render_kwargs = {} if render_kwargs is None else render_kwargs

    def create_graph(self, name):
        self._name_to_graph[name] = DotGraph(
            name, self._graph_attr, self._node_attr, self._edge_attr, self._get_node_attr
        )
        return self._name_to_graph[name]

    def render(self, filename: str = None):
        """render the graph generated from the Relay IR module.

        This function is a thin wrapper of `graphviz.Digraph.render()`.
        """
        # Create or update the filename
        if filename is not None:
            self._render_kwargs["filename"] = filename
        # default cleanup
        if "cleanup" not in self._render_kwargs:
            self._render_kwargs["cleanup"] = True

        root_graph = graphviz.Digraph()
        for graph in self._name_to_graph.values():
            root_graph.subgraph(graph.digraph)
        root_graph.render(**self._render_kwargs)
