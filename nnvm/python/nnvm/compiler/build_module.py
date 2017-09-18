# pylint: disable=invalid-name
"""Namespace for building operators."""
from __future__ import absolute_import as _abs

import tvm
from . import graph_attr
from .. import graph as _graph

@tvm.register_func("nnvm.compiler.lower")
def _lower(sch, inputs, func_name):
    f = tvm.lower(sch, inputs, name=func_name)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@tvm.register_func("nnvm.compiler.build_target")
def _build(funcs, target):
    return tvm.build(funcs, target=target)


_move_module = tvm.get_global_func("nnvm.compiler._move_module")


def optimize(graph):
    """Perform graph optimization

    Parameters
    ----------
    graph : Graph
        The graph to be used in lowering.

    Returns
    -------
    graph : Graph
        The optimized execution graph.
    """
    return graph


def build(graph, target, shape, dtype="float32"):
    """Build graph into runtime library.

    This is the final step of graph compilation.

    Parameters
    ----------
    graph : Graph
        The graph to be used in lowering

    target : str
        The build target

    shape : dict of str to tuple
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    graph : Graph
        The final execution graph.

    libmod : tvm.Module
        The modue that comes with the execution graph
    """
    if not isinstance(target, str):
        raise TypeError("require target to be str")
    if not isinstance(shape, dict):
        raise TypeError("require shape to be dict")

    graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
    graph = graph_attr.set_shape(graph, shape)
    graph = graph_attr.set_dtype(graph, dtype)
    graph._set_json_attr("target", target, "str")
    graph = graph.apply("InferShape").apply("InferType")
    graph = graph.apply("GraphFusePartition").apply("GraphFuse")
    libmod = _move_module(graph)
    return graph, libmod
