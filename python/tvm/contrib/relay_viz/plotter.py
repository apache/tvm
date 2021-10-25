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
"""Abstract class used by :py:class:`tvm.contrib.relay_viz.RelayVisualizer`."""
import abc
from typing import Union


class Graph(abc.ABC):
    """Abstract class for graph, which is composed of nodes and edges."""

    @abc.abstractmethod
    def node(self, node_id: Union[int, str], node_type: str, node_detail: str) -> None:
        """Add a node to the underlying graph.

        Parameters
        ----------
        node_id : Union[int, str]
            Serve as the ID to the node.

        node_type : str
            the type of the node.

        node_detail : str
            the description of the node.
        """

    @abc.abstractmethod
    def edge(self, id_start: Union[int, str], id_end: Union[int, str]) -> None:
        """Add an edge to the underlying graph.

        Parameters
        ----------
        id_start : Union[int, str]
            the ID to the starting node.

        id_end : Union[int, str]
            the ID to the ending node.
        """


class Plotter(abc.ABC):
    """Abstract class for plotters, rendering a collection of Graph interface."""

    @abc.abstractmethod
    def create_graph(self, name: str) -> Graph:
        """Create a graph

        Parameters
        ----------
        name : str
            the name of the graph

        Return
        ------
        rv1: class Graph
        """

    @abc.abstractmethod
    def render(self, filename: str) -> None:
        """Render the graph as a file.

        Parameters
        ----------
        filename : str
            see the definition of implemented class.
        """
