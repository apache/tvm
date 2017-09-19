# pylint: disable=invalid-name
"""Namespace of graph pass.

Principle:
- Graph in, graph out: always takes in graph as first argument and returns a graph
- Composable API: break graph transformation pass as segments of small transformations.
"""
from __future__ import absolute_import as _abs

from . import graph_attr


def infer_shape(graph, **shape):
    """Infer the shape given the shape of inputs.

    Parameters
    ----------
    graph : Graph
        The graph to perform shape inference from

    Returns
    -------
    in_shape : list of tuple
         Shape of inputs

    out_shape: list of tuple
         Shape of outputs
    """
    graph = graph_attr.set_shape_inputs(graph, shape)
    graph = graph.apply("InferShape")
    shape = graph.json_attr("shape")
    index = graph.index
    input_shape = [shape[index.entry_id(x)] for x in index.input_names]
    output_shape = [shape[index.entry_id(x)] for x in index.output_entries]
    return input_shape, output_shape


def infer_dtype(graph, **dtype):
    """Infer the type given the typeS of inputs.

    Parameters
    ----------
    graph : Graph
        The graph to perform type inference from

    Returns
    -------
    in_dtype : list of tuple
         Dtype of inputs

    out_dtype: list of tuple
         Dtype of outputs
    """
    graph = graph_attr.set_dtype_inputs(graph, dtype)
    graph = graph.apply("InferType")
    dtype = graph.json_attr("dtype")
    index = graph.index
    input_dtype = [graph_attr.TCODE_TO_DTYPE[dtype[index.entry_id(x)]]
                   for x in index.input_names]
    output_dtype = [graph_attr.TCODE_TO_DTYPE[dtype[index.entry_id(x)]]
                    for x in index.output_entries]
    return input_dtype, output_dtype
