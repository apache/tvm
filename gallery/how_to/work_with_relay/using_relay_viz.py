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

Relay IR module can contain lots of operations. Although an individual
operation is usually easy to understand, putting them together can cause
a complicated, hard-to-read graph. Things can get even worse with optimization-passes
coming into play.

This utility visualizes an IR module as nodes and edges. It defines a set of interfaces including
parser, plotter(renderer), graph, node, and edges.
A default parser is provided. Users can implement their own renderers to render the graph.

Here we use a renderer rendering graph in the text-form.
It is a lightweight, AST-like visualizer, inspired by `clang ast-dump <https://clang.llvm.org/docs/IntroductionToTheClangAST.html>`_.
We will introduce how to implement customized parsers and renderers through interface classes.
To install dependencies, run:

.. code-block:: bash

    %%shell
    pip install graphviz


For more details, please refer to :py:mod:`tvm.contrib.relay_viz`.
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
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
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
# The terminal can show a Relay IR module in text similar to clang AST-dump.
# We should see ``main`` and ``AddFunc`` function. ``AddFunc`` is called twice in the ``main`` function.
viz = relay_viz.RelayVisualizer(mod)
viz.render()

######################################################################
# Customize Parser for Interested Relay Types
# -------------------------------------------
# Sometimes we want to emphasize interested information, or parse things differently for a specific usage.
# It is possible to provide customized parsers as long as it obeys the interface.
# Here demonstrate how to customize parsers for ``relay.var``.
# We need to implement abstract interface :py:class:`tvm.contrib.relay_viz.interface.VizParser`.
class YourAwesomeParser(VizParser):
    def __init__(self):
        self._delegate = TermVizParser()

    def get_node_edges(
        self,
        node: relay.Expr,
        relay_param: Dict[str, tvm.runtime.NDArray],
        node_to_id: Dict[relay.Expr, str],
    ) -> Tuple[Union[VizNode, None], List[VizEdge]]:

        if isinstance(node, relay.Var):
            node = VizNode(node_to_id[node], "AwesomeVar", f"name_hint {node.name_hint}")
            # no edge is introduced. So return an empty list.
            return node, []

        # delegate other types to the other parser.
        return self._delegate.get_node_edges(node, relay_param, node_to_id)


######################################################################
# Pass the parser and an interested renderer to visualizer.
# Here we just the terminal renderer.
viz = relay_viz.RelayVisualizer(mod, {}, TermPlotter(), YourAwesomeParser())
viz.render()

######################################################################
# Customization around Graph and Plotter
# -------------------------------------------
# Besides parsers, we can also customize graph and renderers by implementing
# abstract class :py:class:`tvm.contrib.relay_viz.interface.VizGraph` and
# :py:class:`tvm.contrib.relay_viz.interface.Plotter`.
# Here we override the ``TermGraph`` defined in ``terminal.py`` for easier demo.
# We add a hook duplicating above ``AwesomeVar``, and make ``TermPlotter`` use the new class.
class AwesomeGraph(TermGraph):
    def node(self, viz_node):
        # add the node first
        super().node(viz_node)
        # if it's AwesomeVar, duplicate it.
        if viz_node.type_name == "AwesomeVar":
            duplicated_id = f"duplicated_{viz_node.identity}"
            duplicated_type = "double AwesomeVar"
            super().node(VizNode(duplicated_id, duplicated_type, ""))
            # connect the duplicated var to the original one
            super().edge(VizEdge(duplicated_id, viz_node.identity))


# override TermPlotter to use `AwesomeGraph` instead
class AwesomePlotter(TermPlotter):
    def create_graph(self, name):
        self._name_to_graph[name] = AwesomeGraph(name)
        return self._name_to_graph[name]


viz = relay_viz.RelayVisualizer(mod, {}, AwesomePlotter(), YourAwesomeParser())
viz.render()

######################################################################
# Summary
# -------
# This tutorial demonstrates the usage of Relay Visualizer and customization.
# The class :py:class:`tvm.contrib.relay_viz.RelayVisualizer` is composed of interfaces
# defined in ``interface.py``.
#
# It is aimed for quick look-then-fix iterations.
# The constructor arguments are intended to be simple, while the customization is still
# possible through a set of interface classes.
#
