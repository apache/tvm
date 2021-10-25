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
# pylint: disable=line-too-long
"""
Use Relay Visualizer to Visualize Relay
============================================================
**Author**: `Chi-Wei Wang <https://github.com/chiwwang>`_

This is an introduction about using Relay Visualizer to visualize a Relay IR module.

Relay IR module can contain lots of operations.  Although individual
operations are usually easy to understand, they become complicated quickly
when you put them together. It could get even worse while optimiztion passes
come into play.

This utility abstracts an IR module as graphs containing nodes and edges.
It provides a default parser to interpret an IR modules with nodes and edges.
Two renderer backends are also implemented to visualize them.

Here we use a backend showing Relay IR module in the terminal for illustation.
It is a much more lightweight compared to another backend using `Bokeh <https://docs.bokeh.org/en/latest/>`_.
See ``<TVM_HOME>/python/tvm/contrib/relay_viz/README.md``.
Also we will introduce how to implement customized parsers and renderers through
some interfaces classes.
"""
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.node_edge_gen import (
    VizNode,
    VizEdge,
    NodeEdgeGenerator,
)
from tvm.contrib.relay_viz.terminal import (
    TermNodeEdgeGenerator,
    TermGraph,
    TermPlotter,
)

######################################################################
# Define a Relay IR Module with multiple GlobalVar
# ------------------------------------------------
# Let's build an example Relay IR Module containing multiple ``GlobalVar``.
# We define an ``add`` function and call it in the main function.
data = relay.var("data")
bias = relay.var("bias")
add_op = relay.add(data, bias)
add_func = relay.Function([data, bias], add_op)
add_gvar = relay.GlobalVar("AddFunc")

input0 = relay.var("input0")
input1 = relay.var("input1")
input2 = relay.var("input2")
add_01 = relay.Call(add_gvar, [input0, input1])
add_012 = relay.Call(add_gvar, [input2, add_01])
main_func = relay.Function([input0, input1, input2], add_012)
main_gvar = relay.GlobalVar("main")

mod = tvm.IRModule({main_gvar: main_func, add_gvar: add_func})

######################################################################
# Render the graph with Relay Visualizer on the terminal
# ------------------------------------------------------
# The terminal backend can show a Relay IR module as in a text-form
# similar to `clang ast-dump <https://clang.llvm.org/docs/IntroductionToTheClangAST.html#examining-the-ast>`_.
# We should see ``main`` and ``AddFunc`` function. ``AddFunc`` is called twice in the ``main`` function.
viz = relay_viz.RelayVisualizer(mod, {}, relay_viz.PlotterBackend.TERMINAL)
viz.render()

######################################################################
# Customize Parser for Interested Relay Types
# -------------------------------------------
# Sometimes the information shown by the default implementation is not suitable
# for a specific usage. It is possible to provide your own parser and renderer.
# Here demostrate how to customize parsers for ``relay.var``.
# We need to implement :py:class:`tvm.contrib.relay_viz.node_edge_gen.NodeEdgeGenerator` interface.
class YourAwesomeParser(NodeEdgeGenerator):
    def __init__(self):
        self._org_parser = TermNodeEdgeGenerator()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, Union[int, str]],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:

        if isinstance(node, relay.Var):
            node = VizNode(node_to_id[node], "AwesomeVar", f"name_hint {node.name_hint}")
            # no edge is introduced. So return an empty list.
            ret = (node, [])
            return ret

        # delegate other types to the original parser.
        return self._org_parser.get_node_edges(node, relay_param, node_to_id)


######################################################################
# Pass a tuple of :py:class:`tvm.contrib.relay_viz.plotter.Plotter` and
# :py:class:`tvm.contrib.relay_viz.node_edge_gen.NodeEdgeGenerator` instances
# to ``RelayVisualizer``. Here we re-use the Plotter interface implemented inside
# ``relay_viz.terminal`` module.
viz = relay_viz.RelayVisualizer(mod, {}, (TermPlotter(), YourAwesomeParser()))
viz.render()

######################################################################
# More Customization around Graph and Plotter
# -------------------------------------------
# All ``RelayVisualizer`` care about are interfaces defined in ``plotter.py`` and
# ``node_edge_generator.py``. We can override them to introduce custimized logics.
# For example, if we want the Graph to duplicate above ``AwesomeVar`` while it is added,
# we can override ``relay_viz.terminal.TermGraph.node``.
class AwesomeGraph(TermGraph):
    def node(self, node_id, node_type, node_detail):
        # add original node first
        super().node(node_id, node_type, node_detail)
        if node_type == "AwesomeVar":
            duplicated_id = f"duplciated_{node_id}"
            duplicated_type = "double AwesomeVar"
            super().node(duplicated_id, duplicated_type, "")
            # connect the duplicated var to the original one
            super().edge(duplicated_id, node_id)


# override TermPlotter to return `AwesomeGraph` instead
class AwesomePlotter(TermPlotter):
    def create_graph(self, name):
        self._name_to_graph[name] = AwesomeGraph(name)
        return self._name_to_graph[name]


viz = relay_viz.RelayVisualizer(mod, {}, (AwesomePlotter(), YourAwesomeParser()))
viz.render()

######################################################################
# Summary
# -------
# This tutorial demonstrates the usage of Relay Visualizer.
# The class :py:class:`tvm.contrib.relay_viz.RelayVisualizer` is composed of interfaces
# defined in ``plotter.py`` and ``node_edge_generator.py``. It provides a single entry point
# while keeping the possibility of implementing customized visualizer in various cases.
#
