# pylint: disable=invalid-name
"""Utility function to get information from graph."""
from __future__ import absolute_import as _abs

import tvm
from . import graph_attr

from .._base import GraphHandle, c_array, ctypes, c_str, check_call, _LIB, nn_uint
from ..graph import create, Graph
from ..symbol import Group, ones_like

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


def annotate_graph(graph, target, op_names=None):
    """ Annotate the nodes in a graph.
    The anntation indicates which device an operator will be scheduled to.

    Parameters
    ----------
    graph : Graph
        The input graph for annotation.

    target: str, tvm.target.Target, or dict of str to str or tvm.target.Target
        Device and compilation target pairs.

    op_names : list of str, optional
        The operators that want to annotated.

    Returns
    -------
    graph : Graph
        The Annotated graph.
    """
    if isinstance(target, str):
        graph._set_json_attr("target", target, "str")
    elif isinstance(target, tvm.target.Target):
        graph._set_json_attr("target", str(target), "str")
    elif isinstance(target, dict):
        if len(target) == 1:
            graph._set_json_attr("target", next(iter(d.values())), "str")
        else:
            for dev, tar in target.items():
                graph._set_json_attr("target" + dev, str(tar), "tar")
    else:
        raise ValueError(
            "target has to be a string, a tvm.target.Target, or a dict and cannot be none.")
    op_names = op_names if op_names else []
    names = c_array(ctypes.c_char_p, [c_str(name) for name in op_names])
    # Save the symbol that represents the updated graph with subgraphs
    out = GraphHandle()

    check_call(_LIB.NNAnnotateGraph(graph.handle, nn_uint(len(op_names)),
                                    names,
                                    ctypes.byref(out)))
    return Graph(out)


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

def get_gradient_graph(ys, xs, grad_ys=None):
    """Create gradient graph of ys with respect to xs.

    Parameters
    ----------
    ys : Symbol or list of Symbol
        Symbols from which the gradient is calculated.
    xs : Symbol or list of Symbol
        Symbols the gradient respect to.
        For group symbol, gradients for all outputs will be calculated.
    grad_ys : Symbol or list of Symbol
        Head gradients for ys.

    Returns
    -------
    ret : Graph
        Generated gradient graph.
    """
    if isinstance(ys, list):
        ys = Group(ys)
    g = create(ys)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    ny = len(ys.list_output_names())
    if grad_ys is None:
        grad_ys = [ones_like(ys[i]) for i in range(ny)]
    g._set_symbol_list_attr('grad_ys_out_grad', grad_ys)
    return g.apply('Gradient')

def gradients(ys, xs, grad_ys=None):
    """Create gradient symbol of ys respect to xs.

    Parameters
    ----------
    ys : Symbol or list of Symbol
        Symbols from which the gradient is calculated.
    xs : Symbol or list of Symbol
        Symbols the gradient respect to.
        For group symbol, gradients for all outputs will be calculated.
    grad_ys : Symbol or list of Symbol
        Head gradients for ys.

    Returns
    -------
    ret : list of Symbol
        Generated gradient symbol. For each xs,
        all gradients from ys are merged into a single symbol.
    """
    grad_g = get_gradient_graph(ys, xs, grad_ys)
    nx = len(Group(xs).list_output_names()) \
        if isinstance(xs, list) else len(xs.list_output_names())
    ret = [grad_g.symbol[i] for i in range(nx)]
    return ret
