"""
Construct the necessary state for the TVM graph runtime
from a Relay expression.
"""
import warnings

from tvm._ffi.runtime_ctypes import TVMContext
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
    "CombineParallelConv2D": 3,
    "FoldScaleAxis": 3,
    "AlterOpLayout": 3,
    "CanonicalizeOps": 3,
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
        "fallback_device": None,
    }

    def __init__(self, **kwargs):
        self._old_scope = None
        for k, _ in kwargs.items():
            if k not in BuildConfig.defaults:
                raise ValueError("invalid argument %s, candidates are %s" %
                                 (k, BuildConfig.defaults.keys()))
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

    fallback_device : str or tvm.TVMContext
        The fallback device. It is also used as the default device for
        operators without specified device during heterogeneous execution.

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


def optimize(func, target=None, params=None):
    """Perform target invariant optimizations.

    Parameters
    ----------
    func : tvm.relay.Function
        The input to optimization.

    target : Optional[:any:`tvm.target.Target`, Dict[int, tvm.target.Target]]
        The optimization target. For heterogeneous compilation, it is a
        dictionary mapping device type to compilation target. For homogeneous
        compilation, it is a build target.

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

    if cfg.pass_enabled("CombineParallelConv2D"):
        func = ir_pass.infer_type(func)
        func = ir_pass.combine_parallel_conv2d(func)

    # The constant folding pass is necessary because FoldScaleAxis pass needs
    # to check the constantness and positiveness of scales.
    if cfg.pass_enabled("FoldConstant"):
        func = ir_pass.fold_constant(func)

    if cfg.pass_enabled("FoldScaleAxis"):
        func = ir_pass.infer_type(func)
        func = ir_pass.backward_fold_scale_axis(func)
        func = ir_pass.infer_type(func)
        func = ir_pass.forward_fold_scale_axis(func)
        func = ir_pass.fold_constant(func)

    if cfg.pass_enabled("CanonicalizeOps"):
        func = ir_pass.infer_type(func)
        func = ir_pass.canonicalize_ops(func)

    # FIXME(zhiics) Skip AlterOpLayout pass for heterogeneous compilation for
    # now. We probably need to pass target to this pass as well. Fix it in
    # a followup PR.
    if cfg.pass_enabled("AlterOpLayout"):
        if isinstance(target, _target.Target):
            func = ir_pass.infer_type(func)
            with target:
                func = ir_pass.alter_op_layout(func)
        elif isinstance(target, dict):
            warnings.warn("AlterOpLayout pass is not enabled for heterogeneous"
                          " execution yet.")

    if cfg.pass_enabled("FoldConstant"):
        func = ir_pass.fold_constant(func)

    return func


def build(func, target=None, target_host=None, params=None):
    """Build a function to run on TVM graph runtime.

    Parameters
    ----------
    func: relay.Function
        The function to build.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context
    name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    target_host : str or :any:`tvm.target.Target`, optional
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

    if isinstance(target, dict):
        target, fallback_device = _update_heterogeneous_inputs(target)
    elif isinstance(target, (str, _target.Target)):
        target = _target.create(target)
    else:
        raise ValueError("target must be the type of str, tvm.target.Target," +
                         "or dict of device name to target")

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        if isinstance(target, dict):
            tophub_context = autotvm.tophub.context(list(target.values()))
        else:
            tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.util.EmptyContext()

    cfg = BuildConfig.current

    with tophub_context:
        func = optimize(func, target, params)
        # Annotate the ops for heterogeneous execution.
        if isinstance(target, dict):
            func, target = _run_device_annotation_passes(func, target,
                                                         fallback_device)
        # Fuse ops before running code gen
        func = ir_pass.infer_type(func)
        func = ir_pass.fuse_ops(func, cfg.opt_level)
        # Graph code generation
        func = ir_pass.infer_type(func)
        graph_gen = _graph_gen.GraphRuntimeCodegen(mod=None, target=target)
        graph_json, lowered_funcs, params = graph_gen.codegen(func)
        mod = _tvm_build_module(
            lowered_funcs, target=target, target_host=target_host)
    return graph_json, mod, params


def _update_heterogeneous_inputs(target):
    """Update the target and fallback device required for heterogeneous
    compilation. CPU is used as the fallback device if it wasn't provided.
    Meanwhile, a CPU device type and "llvm" pair will be added to the target
    dictionary in this case.

    Parameters
    ----------
    target : dict of str(i.e. device/context name) to str/tvm.target.Target.
        A dict contains context to target pairs.

    Returns
    -------
    device_target : dict of int to tvm.target.Target.
        The updated device type to target dict.

    fallback_device : int
        The updated fallback device type.
    """
    if not isinstance(target, dict):
        raise ValueError("target must be dict of device name to target for " +
                         "heterogeneous execution, but received %s."
                         % type(target))

    fallback_device = BuildConfig.current.fallback_device
    if fallback_device is None:
        # cpu is used as the default fallback device when heterogeneous
        # execution is needed, but no fallback device is provided.
        fallback_device = _nd.cpu(0).device_type
        target[fallback_device] = str(_target.create("llvm"))
    elif isinstance(fallback_device, str):
        fallback_device = _nd.context(fallback_device).device_type
    elif isinstance(fallback_device, TVMContext):
        fallback_device = fallback_device.device_type
    else:
        raise ValueError("fallback_device expects the type of str or " +
                         "TVMContext, but received %s." % type(fallback_device))

    device_target = {}
    for dev, tgt in target.items():
        device_target[_nd.context(dev).device_type] = _target.create(tgt)

    if fallback_device not in device_target:
        raise ValueError("%s is used as the default device, but the target" +
                         "is not provided."
                         % _nd.context(fallback_device).device_name)
    return device_target, fallback_device


def _run_device_annotation_passes(func, target, fallback_device):
    """Execute the device annotation passes to update the input program and
    target information.

    Parameters
    ----------
    func: tvm.relay.Function
        The function where annotation passes will be execute at.

    target : Dict[int, tvm.target.Target]
        A dict contains device type to target pairs.

    fallback_device : int
        The fallback device type.

    Returns
    -------
    target : Dict[int, tvm.target.Target]
        The updated device type to target dict.

    func : tvm.relay.Function
        The updated func.
    """
    func = ir_pass.infer_type(func)
    func = ir_pass.rewrite_annotated_ops(func, fallback_device)
    device_map = ir_pass.collect_device_info(func)
    # The expression to device type map will be empty if all or none of
    # the expressions in the `func` are annotated because this map is
    # obtained by propagating the device information in the device copy
    # operator. None of the above cases needs device copy operator.
    if not device_map:
        annotation_map = ir_pass.collect_device_annotation_ops(func)
        # No annotation.
        if not annotation_map:
            target = {0: target[fallback_device]}
        else:
            dev_type = next(iter(annotation_map.values()))
            # All annotated with the same device type.
            if all(val == dev_type for val in annotation_map.values()):
                target = {0: target[dev_type]}
            else:
                raise RuntimeError("Expressions in the function are "
                                   "annotated with various device types,"
                                   "but not device copy operators "
                                   "found. Please check the "
                                   "RewriteAnnotation pass.")
    return func, target


class GraphExecutor(_interpreter.Executor):
    """Wrapper around Executor interface.

    This executor is used for debug and testing purpoes.

    Parameters
    ----------
    mod : :py:class:`~tvm.relay.module.Module`
        The module to support the execution.

    ctx : :py:class:`TVMContext`
        The runtime context to run the code on.

    target : :py:class:`Target`
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
            gmodule.set_input(**params)

        def _graph_wrapper(*args, **kwargs):
            args = self._convert_args(func, args, kwargs)
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

    mod : :py:class:`~tvm.relay.module.Module`
        The Relay module containing collection of functions

    ctx : :py:class:`tvm.TVMContext`
        The context to execute the code.

    target : :py:class:`tvm.Target`
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
    if kind == "graph":
        return GraphExecutor(mod, ctx, target)
    raise RuntimeError("unknown mode {0}".format(mode))
