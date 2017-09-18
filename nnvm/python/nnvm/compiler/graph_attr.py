"""Utilities to access graph attributes"""
from __future__ import absolute_import as _abs

def set_shape(g, shape):
    """Set the shape of graph nodes in the graph attribute.

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
    index = g.index
    list_shape = [[]] * index.num_node_entries
    for k, v in shape.items():
        list_shape[index.entry_id(k)] = v
    g._set_json_attr("shape", list_shape, 'list_shape')
    return g


DTYPE_DICT = {
    "float32": 0
}

def set_dtype(g, dtype):
    """Set the dtype of graph nodes

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
    index = g.index
    if isinstance(dtype, dict):
        list_dtype = [-1] * index.num_node_entries
        for k, v in dtype.items():
            list_dtype[index.entry_id(k)] = DTYPE_DICT[v]
    else:
        list_dtype = [DTYPE_DICT[dtype]] * index.num_node_entries
    g._set_json_attr("dtype", list_dtype, "list_int")
    return g
