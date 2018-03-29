# pylint: disable=invalid-name
"""Namespace for building operators."""
from __future__ import absolute_import as _abs

import logging
import tvm

from tvm.contrib import graph_runtime
from . import graph_attr, graph_util
from .. import graph as _graph
from .. import symbol as sym
from .._base import _all_var_init

OPT_PASS_LEVEL = {
    "SimplifyInference": 0,
    "PrecomputePrune": 2,
    "OpFusion": 1,
    "FoldScaleAxis": 3
}

# List of optimization pass and level when switch on
class BuildConfig(object):
    """Configuration scope to set a build config option.

    Parameters
    ----------
    kwargs
        Keyword arguments of configurations to set.
    """
    current = None
    defaults = {
        "opt_level": 2,
        "add_pass": None,
    }
    def __init__(self, **kwargs):
        self._old_scope = None
        for k, _ in kwargs.items():
            if k not in BuildConfig.defaults:
                raise ValueError(
                    "invalid argument %s, candidates are %s" % (k, BuildConfig.defaults.keys()))
        self._attr = kwargs

    def __getattr__(self, name):
        if name not in self._attr:
            return BuildConfig.defaults[name]
        return self._attr[name]

    def __enter__(self):
        # pylint: disable=protected-access
        self._old_scope = BuildConfig.current
        attr = BuildConfig.current._attr.copy()
        attr.update(self._attr)
        self._attr = attr
        BuildConfig.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        BuildConfig.current = self._old_scope

    def pass_enabled(self, pass_name):
        """Get whether pass is enabled.

        Parameters
        ----------
        pass_name : str
            The optimization pass name

        Returns
        -------
        enabled : bool
            Whether pass is enabled.
        """
        if self.add_pass and pass_name in self.add_pass:
            return True
        return self.opt_level >= OPT_PASS_LEVEL[pass_name]


BuildConfig.current = BuildConfig()

def build_config(**kwargs):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    opt_level: int, default=2
        Optimization level. See OPT_PASS_LEVEL for level of each pass.

    add_pass: set of str
        Optimization pass to be added regardless of optimization level.

    Returns
    -------
    config: BuildConfig
        The build configuration
    """
    return BuildConfig(**kwargs)


@tvm.register_func("nnvm.compiler.lower")
def _lower(sch, inputs, func_name, graph):
    import traceback
    # pylint: disable=broad-except
    try:
        f = tvm.lower(sch, inputs, name=func_name)
        logging.debug("lower function %s", func_name)
        logging.debug("%s", tvm.lower(sch, inputs, simple_mode=True))
    except Exception:
        msg = traceback.format_exc()
        msg += "Error during compile graph\n"
        msg += "--------------------------\n"
        msg += graph.ir(join_entry_attrs=["shape"])
        raise RuntimeError(msg)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@tvm.register_func("nnvm.compiler.build_target")
def _build(funcs, target, target_host):
    if target_host == "":
        target_host = None
    return tvm.build(funcs, target=target, target_host=target_host)


def _update_shape_dtype(shape, dtype, params):
    """Update shape dtype given params information"""
    if not params:
        return shape, dtype
    shape = shape.copy()
    shape.update({k : v.shape for k, v in params.items()})
    if isinstance(dtype, str):
        for k, v in params.items():
            if v.dtype != dtype:
                raise ValueError(
                    "%s: dtype not expected %s vs %s" % (k, dtype, v.dtype))
    else:
        dtype = dtype.copy()
        dtype.update({k : str(v.dtype) for k, v in params.items()})
    return shape, dtype


def optimize(graph, shape, dtype="float32"):
    """Perform target and parameter invariant graph optimization.

    This is an advanced function that usually do not need to be called.
    Call build instead.

    Parameters
    ----------
    graph : Graph
        The graph to be used in optimized.

    Returns
    -------
    graph : Graph
        The optimized graph.
    """
    # pylint: disable=unused-argument
    cfg = BuildConfig.current
    if cfg.pass_enabled("SimplifyInference"):
        graph = graph_attr.set_shape_inputs(graph, shape)
        graph = graph.apply(["InferShape", "SimplifyInference"])

    if cfg.pass_enabled("FoldScaleAxis"):
        graph = graph_attr.set_shape_inputs(graph, shape)
        graph = graph.apply(["InferShape", "FoldScaleAxis"])
    return graph


def build(graph, target=None, shape=None, dtype="float32", params=None, target_host=None):
    """Build graph into runtime library.

    The build function will optimize the graph and do the compilation.

    When params is provided, the compiler might split the graph to
    pre-compute certain values, so the final execution graph can
    be different from the original one.

    Parameters
    ----------
    graph : Graph
        The graph to be used in lowering

    target : str or :any:`tvm.target.Target`, optional
        The build target

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for pre-compute
        folding optimization.

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    initialize : bool, optional
        Whether to initialize variables in global dict _all_var_init.

    Returns
    -------
    graph : Graph
        The final execution graph.

    libmod : tvm.Module
        The module that comes with the execution graph

    params : dict of str to NDArray
        The updated parameters of graph if params is passed.
        This can be different from the params passed in.
    """
    target = target if target else tvm.target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")
    target = tvm.target.create(target)

    shape = shape if shape else {}
    if not isinstance(shape, dict):
        raise TypeError("require shape to be dict")
    cfg = BuildConfig.current
    graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
    shape, dtype = _update_shape_dtype(shape, dtype, params)
    # Initial pass do shape type inference
    ishape, _ = graph_util.infer_shape(graph, **shape)
    shape.update(zip(graph.index.input_names, ishape))
    if not isinstance(dtype, str):
        idtype, _ = graph_util.infer_dtype(graph, **dtype)
        dtype.update(zip(graph.index.input_names, idtype))
    # Initialize all variables specified in _all_var_init
    init_var = {}
    if _all_var_init:
        init_var = initialize_variables(shape, dtype)
    # Apply optimization
    graph = optimize(graph, shape, dtype)
    # Precompute prune
    if params and cfg.pass_enabled("PrecomputePrune"):
        graph, params = precompute_prune(graph, params)
        shape, dtype = _update_shape_dtype(shape, dtype, params)
    # Operator Fusion and generation
    graph = graph_attr.set_shape_inputs(graph, shape)
    graph = graph_attr.set_dtype_inputs(graph, dtype)
    graph._set_json_attr("target", str(target), "str")
    if target_host is not None:
        graph._set_json_attr("target_host", str(target_host), "str")
    if cfg.pass_enabled("OpFusion"):
        graph._set_json_attr("opt_level", 1, "int")
    else:
        graph._set_json_attr("opt_level", 0, "int")
    graph = graph.apply("InferShape").apply("InferType")
    with target:
        graph = graph.apply("GraphFusePartition").apply("GraphFuseCompile")
    libmod = graph_attr._move_out_module(graph, "module")
    # Write variable initial values into params
    if init_var:
        if params is None:
            params = {}
        params.update(init_var)
    return graph, libmod, params


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
    _, oshape = graph_util.infer_shape(graph, **shape)
    _, odtype = graph_util.infer_dtype(graph, **dtype)
    graph, libmod, _ = build(graph, target, shape, dtype)
    m = graph_runtime.create(graph, libmod, ctx)
    set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
    kset = set(graph.symbol.list_input_names())
    for k, v in params.items():
        if k in kset:
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
    if pre_graph is None:
        return graph, params
    out_names = pre_graph.json_attr("output_names")
    if not pre_graph.symbol.list_output_names():
        return graph, params
    with tvm.build_config(auto_unroll_max_step=0):
        out_arrs = _run_graph(pre_graph, params)
    return graph, dict(zip(out_names, out_arrs))


def initialize_variables(ishape, idtype):
    """ Initialize variables stored in _all_var_init dictionary.

    Parameters
    ----------
    ishape : dict of str to tuple of int
        The input shape to the graph

    idtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    init_var : dict of str to tvm.ndarray
    """
    symbol_init_dict = {}
    const_init_dict = {}
    init_var = {}
    for key, value in _all_var_init.items():
        if isinstance(value, sym.Symbol):
            symbol_init_dict[key] = value
        else:
            const_init_dict[key] = tvm.nd.array(value)
    # Make sure variables are initialized only once.
    _all_var_init.clear()
    if symbol_init_dict:
        # Create dummy params to run initialization graph
        params = {}
        for name, shape in ishape.items():
            dtype = idtype if isinstance(idtype, str) else idtype[name]
            params[name] = tvm.nd.empty(shape, dtype, ctx=tvm.cpu())
        init_group_sym = sym.Group(symbol_init_dict.values())
        graph = _graph.create(init_group_sym)
        with tvm.build_config(auto_unroll_max_step=0):
            init_values = _run_graph(graph, params)
        init_var.update(dict(zip(symbol_init_dict.keys(), init_values)))
    init_var.update(const_init_dict)
    for name, data in init_var.items():
        ishape[name] = data.shape
    return init_var
