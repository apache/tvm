"""TF: Tensorflow parser"""
from __future__ import absolute_import as _abs
from __future__ import print_function
from nnvm.frontend.protobuf import graph_pb2

class TFParser(object):
    """A Wrapper to handle tensorflow frozen model parsing
       Works w/o installing tensorflow,
       Protocol Buffer is needed
    ```
    parser = TfParser(pb_file)
    graph = parser.parse()
    ```
    Parameters
    ----------
    pb_file : tensorflow frozen pb file
        The pb file should include both operations and tensors
    """

    def __init__(self, pb_file):
        self._pb = pb_file
        self._graph = graph_pb2.GraphDef()

    def _load_model(self):
        """load frozen tensorflow model, return GraphDef """
        with open(self._pb, "rb") as f:
            self._graph.ParseFromString(f.read())

    def parse(self):
        self._load_model()
        return self._graph