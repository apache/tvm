# pylint: disable=invalid-name
"""Utilities to access graph attributes"""
from __future__ import absolute_import as _abs

import tvm

def set_shape_inputs(g, shape):
    """Set the shape of input graph nodes in the graph attribute.

    Parameters
    ----------
    g : Graph
        The input graph

    shape : dict of str to tuple
        The input shape

    Returns
    -------
    g : Graph
        The updated graph with updated shape.
    """
    list_shape = [
        shape.get(name, ()) for name in g.index.input_names]
    g._set_json_attr("shape_inputs", list_shape, 'list_shape')
    return g


DTYPE_TO_TCODE = {
    "default": -1,
    "float32": 0
}

TCODE_TO_DTYPE = {
    -1: None,
    0: "float32"
}

def set_dtype_inputs(g, dtype):
    """Set the dtype inputs of graph nodes

    Parameters
    ----------
    g : Graph
        The input graph

    dtype : dict of str to str or str
        The input dtype

    Returns
    -------
    g : Graph
        The updated graph with updated dtype.
    """
    if isinstance(dtype, dict):
        list_dtype = [
            DTYPE_TO_TCODE[str(dtype.get(name, "default"))]
            for name in g.index.input_names]
    else:
        list_dtype = [DTYPE_TO_TCODE[dtype]] * len(g.index.input_names)
    g._set_json_attr("dtype_inputs", list_dtype, "list_int")
    return g


def set_layout_inputs(g, layout):
    """Set the layout inputs of graph nodes

    Parameters
    ----------
    g : Graph
        The input graph

    layout : dict of str to str or str
        The input layout

    Returns
    -------
    g : Graph
        The updated graph with updated dtype.
    """
    list_shape = [
        layout.get(name, "default") for name in g.index.input_names]
    g._set_json_attr("layout_inputs", list_shape, 'list_str')
    return g

_move_out_module = tvm.get_global_func("nnvm.graph._move_module")
_move_out_graph = tvm.get_global_func("nnvm.graph._move_graph")
