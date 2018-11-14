"""
Construct the necessary state for the TVM graph runtime
from a Relay expression.
"""
from ..build_module import build as _tvm_build_module
from .. import nd as _nd, target as _target, autotvm
from ..contrib import graph_runtime as _graph_rt
from . import ir_pass
from . import expr
from .backend import interpreter as _interpreter
from .backend import graph_runtime_codegen as _graph_gen

# List of optimization pass and level when switch on
OPT_PASS_LEVEL = {
    "SimplifyInference": 0,
    "OpFusion": 1,
    "FoldConstant": 2,
    "FoldScaleAxis": 3,
}

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


def _bind_params_by_name(func, params):
    """Bind parameters of function by its name."""
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = expr.const(v)
    return expr.bind(func, bind_dict)


def optimize(func, params=None):
    """Perform target invariant optimizations.

    Parameters
    ----------
    func : tvm.relay.Function
        The input to optimization.

    params : Optional[Dict[str, tvm.nd.NDArray]]
        Input parameters to the graph that do not change
        during inference time. used for constant folding.

    Returns
    -------
    opt_func : tvm.relay.Function
        The optimized version of the function.
    """
    cfg = BuildConfig.current

    # bind expressions
    if params:
        func = _bind_params_by_name(func, params)

    if cfg.pass_enabled("SimplifyInference"):
        func = ir_pass.infer_type(func)
        func = ir_pass.simplify_inference(func)

    if cfg.pass_enabled("FoldScaleAxis"):
        func = ir_pass.infer_type(func)
        func = ir_pass.backward_fold_scale_axis(func)
        func = ir_pass.infer_type(func)
        func = ir_pass.forward_fold_scale_axis(func)

    if cfg.pass_enabled("FoldConstant"):
        func = ir_pass.fold_constant(func)

    return func


def build(func,
          target=None,
          target_host=None,
          params=None):
    """Build a function to run on TVM graph runtime.

    Parameters
    ----------
    func: relay.Function
        The function to build.

    target : str or :any:`tvm.target.Target`, optional
        The build target

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    graph_json : str
        The json string that can be accepted by graph runtime.

    mod : tvm.Module
        The module containing necessary libraries.

    params : dict
        The parameters of the final graph.
    """
    target = target if target else _target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")
    target = _target.create(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.util.EmptyContext()

    cfg = BuildConfig.current

    with tophub_context:
        func = optimize(func, params)
        # Fuse ops before running code gen
        func = ir_pass.infer_type(func)
        func = ir_pass.fuse_ops(func, cfg.opt_level)
        # Graph code generation
        func = ir_pass.infer_type(func)
        graph_gen = _graph_gen.GraphRuntimeCodegen(mod=None, target=target)
        graph_json, lowered_funcs, params = graph_gen.codegen(func)
        mod = _tvm_build_module(lowered_funcs, target=target, target_host=target_host)
    return graph_json, mod, params


class GraphExecutor(_interpreter.Executor):
    """Wrapper around Executor interface.

    This executor is used for debug and testing purpoes.

    Parameters
    ----------
    mod : tvm.relay.Module
        The module to support the execution.

    ctx : tvm.TVMContext
        The runtime context to run the code on.

    target : tvm.Target
        The target option to build the function.
    """
    def __init__(self, mod, ctx, target):
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, func):
        graph_json, mod, params = build(func, target=self.target)
        gmodule = _graph_rt.create(graph_json, mod, self.ctx)
        if params:
            gmodule.set_input(*params)
        def _graph_wrapper(*args):
            # Create map of inputs.
            for i, arg in enumerate(args):
                gmodule.set_input(i, arg)
            # Run the module, and fetch the output.
            gmodule.run()
            # make a copy so multiple invocation won't hurt perf.
            return gmodule.get_output(0).copyto(_nd.cpu(0))

        return _graph_wrapper


def create_executor(kind="debug",
                    mod=None,
                    ctx=None,
                    target="llvm"):
    """Factory function to create an executor.

    Parameters
    ----------
    kind : str
        The type of executor

    mod : relay.Mod
        The mod

    ctx : tvm.TVMContext
        The context to execute the code.

    target : tvm.Target
        The corresponding context
    """
    if ctx is not None:
        assert ctx.device_type == _nd.context(str(target), 0).device_type
    else:
        ctx = _nd.context(str(target), 0)

    if isinstance(target, str):
        target = _target.create(target)
    if kind == "debug":
        return _interpreter.Interpreter(mod, ctx, target)
    elif kind == "graph":
        return GraphExecutor(mod, ctx, target)
    else:
        raise RuntimeError("unknown mode {0}".format(mode))
