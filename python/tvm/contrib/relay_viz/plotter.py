"""
"""

import abc

class Plotter(abc.ABC):
    @abc.abstractmethod
    def node(self, node_id, node_type, node_detail):
        """ Add a node to the underlying graph.

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
        """ Add an edge to the underlying graph.

        Parameters
        ----------
        id_start : object
                        the ID to the starting node.

        id_start : object
                        the ID to the ending node.
        """

    @abc.abstractclassmethod
    def render(self, filename):
        """ Render the graph as a file.

        Parameters
        ----------
        filename : string
        """
