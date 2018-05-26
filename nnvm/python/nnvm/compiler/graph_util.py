# pylint: disable=invalid-name
"""Utility function to get information from graph."""
from __future__ import absolute_import as _abs

import tvm
from . import graph_attr

def infer_shape(graph, **shape):
    """Infer the shape given the shape of inputs.

    Parameters
    ----------
    graph : Graph
        The graph to perform shape inference from

    shape : dict of str to tuple
        The specific input shape.

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

    dtype : dict of str to dtype
        The specific input data type.

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


_deep_compare = tvm.get_global_func("nnvm.graph.DeepCompare")

def check_graph_equal(grapha, graphb, compare_variable_attrs=False):
    """Check if two graphs have equal structure.

    Parameters
    ----------
    grapha : Graph
        The first graph

    graphb : Graph
        The second graph

    compare_variable_attrs : bool, optional
        Whether we want to compare attributes(names) on variables.
        Usually it is safe to skip it unless we want input name
        to exactly match

    Raises
    ------
    ValueError
        ValueError is raised with error message when graph not equal
    """
    err = _deep_compare(grapha, graphb, compare_variable_attrs)
    if err:
        raise ValueError("Graph compare error: " + err)
