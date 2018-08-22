# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""NNVM subgraph.

This is a developer API that is used to define configurations and util functions
for partitioning graphs.
"""

from __future__ import absolute_import as _abs
import ctypes
from ._base import _LIB
from ._base import c_array, c_str, nn_uint
from ._base import GraphHandle
from ._base import check_call
from .graph import Graph


_OP_WHITELIST_DICT = {"tensorrt": ['conv2d',
                                   'batch_norm',
                                   'relu',
                                   'sigmoid',
                                   'tanh',
                                   'elemwise_add',
                                   'max_pool2d',
                                   'avg_pool2d',
                                   'global_max_pool2d',
                                   'global_avg_pool2d',
                                   'dense',
                                   'softmax',
                                   'concatenate',
                                   'conv2d_transpose',
                                   'slice_like']}


def _partition(graph, subgraph_backend, op_names=None):
    """Internal function for partitioning the graph using
    the subgraph property of the subgraph_backend.

    Parameters
    ----------
    graph : Graph
        The graph to be partitioned.

    subgraph_backend : str
        The name of the external accelerator that serves as the backend of the subgraphs.

    op_names : list of strs
        The operator names supported by the external accelerator. By default,
        every subgraph backend has a whitelist of operators. This parameter
        is provided to override that whitelist for the purpose of testing.
    """
    if subgraph_backend not in _OP_WHITELIST_DICT:
        raise ValueError("Unsupported subgraph backend %s, valid candidates are %s"
                         % (subgraph_backend, _OP_WHITELIST_DICT.keys()))
    if op_names is None:
        op_names = _OP_WHITELIST_DICT.get(subgraph_backend)
    out = GraphHandle()
    check_call(_LIB.NNPartitionGraph(graph.handle,
                                     c_str(subgraph_backend),
                                     nn_uint(len(op_names)),
                                     c_array(ctypes.c_char_p, [c_str(s) for s in op_names]),
                                     ctypes.byref(out)))
    return Graph(out)
