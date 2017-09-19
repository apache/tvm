# pylint: disable=invalid-name
"""Namespace for building operators."""
from __future__ import absolute_import as _abs

import tvm
from . import graph_attr, graph_pass
from .. import graph as _graph
from .. import runtime

@tvm.register_func("nnvm.compiler.lower")
def _lower(sch, inputs, func_name):
    f = tvm.lower(sch, inputs, name=func_name)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@tvm.register_func("nnvm.compiler.build_target")
def _build(funcs, target):
    return tvm.build(funcs, target=target)


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
    graph = graph_attr.set_shape_inputs(graph, shape)
    graph = graph_attr.set_dtype_inputs(graph, dtype)
    graph._set_json_attr("target", target, "str")
    graph = graph.apply("InferShape").apply("InferType")
    graph = graph.apply("GraphFusePartition").apply("GraphFuse")
    libmod = graph_attr._move_out_module(graph, "module")
    return graph, libmod


def _run_graph(graph, params):
    """Helper utility to build and run and get outputs, only use cpu mode.

    Parameters
    ----------
    graph : Graph
        The graph to be executed.

    params: dict of str to ndarray
        The parameter dictionary.

    Returns
    -------
    out_dict: dict of str to tvm.NDArray
        The output dictionaries.
    """
    graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
    shape = {k : v.shape for k, v in params.items()}
    dtype = {k : v.dtype for k, v in params.items()}
    target = "llvm"
    ctx = tvm.cpu(0)
    _, oshape = graph_pass.infer_shape(graph, **shape)
    _, odtype = graph_pass.infer_dtype(graph, **dtype)
    graph, libmod = build(graph, target, shape, dtype)
    m = runtime.create(graph, libmod, ctx)
    set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
    for k, v in params.items():
        set_input(k, tvm.nd.array(v))
    run()
    out_data = []
    for i, kv in enumerate(zip(oshape, odtype)):
        shape, dtype = kv
        arr = tvm.nd.empty(shape, dtype, ctx)
        get_output(i, arr)
        out_data.append(arr)
    return out_data


def precompute_prune(graph, params):
    """Precompute the part of graph that can be pre-computed.

    This will create a new graph that only contains the ops
    that need to be computed depending on input as well as
    updated version of param dict that pre-computes some of
    intermediate results.

    Parameters
    ----------
    graph : Graph
        The input graph

    params : dict of str -> tvm.NDArray
        The parameter dictionary of the graph

    Returns
    -------
    pruned_graph : Graph
        The pruned graph

    new_params : dict of str-> tvm.NDArray
        The updated dictionary of parameters.
    """
    graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
    graph._set_json_attr("param_name_list", list(params.keys()), "list_str")
    graph = graph.apply("PrecomputePrune")
    pre_graph = graph_attr._move_out_graph(graph, "precompute_graph")
    if not pre_graph.symbol.list_output_names():
        return graph, params
    out_names = pre_graph.json_attr("output_names")
    out_arrs = _run_graph(pre_graph, params)
    return graph, dict(zip(out_names, out_arrs))
