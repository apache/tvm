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
"""Abstract class for plotters."""
import abc


class Graph(abc.ABC):
    """Abstract class for graph.

    Implement this interface for various graph libraries.
    """

    @abc.abstractmethod
    def node(self, node_id, node_type, node_detail):
        """Add a node to the underlying graph.

        Parameters
        ----------
        node_id : object
                        Serve as the ID to the node.

        node_type : string
                        the type of the node.

        node_detail : string
                        the description of the node.
        """

    @abc.abstractmethod
    def edge(self, id_start, id_end):
        """Add an edge to the underlying graph.

        Parameters
        ----------
        id_start : object
                        the ID to the starting node.

        id_start : object
                        the ID to the ending node.
        """


class Plotter(abc.ABC):
    """Abstract class for plotters.

    Implement this interface for various graph libraries.
    """

    @abc.abstractmethod
    def create_graph(self, name):
        """Create a graph

        Parameters
        ----------
        name : string, the name of the graph

        Return
        ------
        Graph instance.
        """

    @abc.abstractmethod
    def render(self, filename):
        """Render the graph as a file.

        Parameters
        ----------
        filename : string
        """
