# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Construct the necessary state for the TVM graph executor
from a Relay expression.
"""
import warnings
import numpy as np

from tvm.ir import IRModule

from tvm.ir.transform import PassContext
from tvm.tir import expr as tvm_expr
from tvm.target import Target
from .. import nd as _nd, autotvm, register_func
from ..target import Target
from ..contrib import graph_executor as _graph_rt
from . import _build_module
from . import ty as _ty
from . import expr as _expr
from . import function as _function
from .transform import InferType
from .backend.utils import mangle_module_name
from .backend import executor_factory as _executor_factory
from .backend import interpreter as _interpreter
from .backend.vm import VMExecutor


def build_target_by_device_type_map(target):
    """Build a map from DLDevice device_type to a Target used with that device.

    At runtime, TVM assigns target code to DLDevices by determining a device_type for each Target.
    This function handles this process at compile time and, as a side effect, validates that exactly
    one target maps to one device_type.

    Parameters
    ----------
    target : Target or str or dict
       If a Target or str: assumes that exactly one device type is present in the model.
       If a dict: keys are tvm.ndarray.device, values are the targets used for each device.

    Returns
    -------

    """
    target = target if target else Target.current()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")

    tgts = {}
    if isinstance(target, (str, Target)):
        dev_type = tvm_expr.IntImm("int32", _nd.device(str(target)).device_type)
        tgts[dev_type] = Target(target)
    elif isinstance(target, dict):
        for dev, tgt in target.items():
            dev_type = tvm_expr.IntImm("int32", _nd.device(dev).device_type)
            tgts[dev_type] = Target(tgt)
    else:
        raise TypeError(
            "target is expected to be str or "
            + "tvm.target.Target, but received "
            + "{}".format(type(target))
        )
    return tgts


def _convert_param_map(params):
    inputs = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = _nd.array(param)
        inputs[name] = _expr.const(param)
    return inputs


class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """

    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]
        self._get_function_metadata = self.mod["get_function_metadata"]

    def build(
        self, mod, target=None, target_host=None, params=None, executor="graph", mod_name=None
    ):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        target_host : str or :any:`tvm.target.Target`, optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm interpreter is used.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        executor: str[Optional]
            The type of executor to be used in order to run the model:
            - If "graph" is specified, then the graph_executor will be used
            - If "aot" is specified, then the aot_executor will be used

        mod_name: Optional[str]
            The module name we will build

        Returns
        -------
        graph_json : str
            The json string that can be accepted by graph executor.

        mod : tvm.Module
            The module containing necessary libraries.

        params : dict
            The parameters of the final graph.
        """
        if target_host is not None:
            warnings.warn(
                "target_host parameter is going to be deprecated. "
                "Please pass in tvm.target.Target(target, host=target_host) instead."
            )
        target = build_target_by_device_type_map(target)
        target, target_host = Target.check_and_update_host_consist(
            target, target_host, target_is_dict_key=False
        )

        # Setup the params.
        if params:
            self._set_params(params)

        # Build the IR module. If auto_scheduler is not enabled,
        # then use the TOPI-defined schedule.
        use_auto_scheduler = PassContext.current().config.get(
            "relay.backend.use_auto_scheduler", False
        )

        # Turn off AutoTVM config not found warnings if auto_scheduler is enabled.
        old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = use_auto_scheduler

        mod_name = mangle_module_name(mod_name)

        self._build(mod, target, target_host, executor, mod_name)
        autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

        # Get artifacts
        mod = self.get_module()
        params = self.get_params()
        executor_config = self.get_graph_json() if executor == "graph" else None

        return executor_config, mod, params

    def optimize(self, mod, target=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IR module to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        mod : :py:class:`~tvm.IRModule`
            The optimized relay module.

        params : dict
            The parameters of the final graph.
        """
        target = build_target_by_device_type_map(target)

        # Setup the params.
        if params:
            self._set_params(params)
        mod = self._optimize(mod, target)
        # Get artifacts
        params = self.get_params()

        return mod, params

    def _set_params(self, params):
        self._set_params_func(_convert_param_map(params))

    def get_graph_json(self):
        """Return the json file of the built program."""
        return self._get_graph_json()

    def get_module(self):
        """Return the built module."""
        return self._get_module()

    def get_function_metadata(self):
        """Return the compiled function metadata.
        Currently, the metadata contains workspace size required by
        each PrimFunc"""
        return self._get_function_metadata()

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret


@register_func("tvm.relay.module_export_library")
def _module_export(module, file_name):  # fcompile, addons, kwargs?
    return module.export_library(file_name)


@register_func("tvm.relay.build")
def _build_module_no_factory(mod, target=None, target_host=None, params=None, mod_name="default"):
    """A wrapper around build which discards the Python GraphFactoryRuntime.
    This wrapper is suitable to be used from other programming languages as
    the runtime::Module can be freely passed between language boundaries.
    """
    target, target_host = Target.check_and_update_host_consist(target, target_host)
    return build(mod, target, params=params, mod_name=mod_name).module


def get_executor_from_target(target, target_host):
    """Helper function to extract the executor parameter from the target

    Parameters
    ----------
    target : Dict of targets for heterogeneous compilation

    target_host :  Host compilation target

    Returns
    -------
    executor : str
    A string representing the executor type
    """

    # Default executor is graph
    executor = "graph"
    cpu_device_type = 1
    if target_host:
        executor = target_host.attrs.get("executor", "graph")
    else:
        for device_type in target:
            if device_type == cpu_device_type:
                executor = target[device_type].attrs.get("executor", "graph")
    return executor


def build(ir_mod, target=None, target_host=None, params=None, mod_name="default"):
    # fmt: off
    # pylint: disable=line-too-long
    """Helper function that builds a Relay function to run on TVM graph executor.

    Parameters
    ----------
    ir_mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    target_host : str or :any:`tvm.target.Target`, optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm interpreter is used.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    mod_name: Optional[str]
        The module name we will build

    Returns
    -------
    factory_module : tvm.relay.backend.executor_factory.ExecutorFactoryModule
            The runtime factory for the TVM graph executor.
    """
    # pylint: enable=line-too-long
    # fmt: on

    if not isinstance(ir_mod, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(ir_mod, _function.Function):
        if params:
            ir_mod = bind_params_by_name(ir_mod, params)
        ir_mod = IRModule.from_expr(ir_mod)
        warnings.warn(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter mod (tvm.relay.function.Function)",
            DeprecationWarning,
        )

    if target_host is not None:
        warnings.warn(
            "target_host parameter is going to be deprecated. "
            "Please pass in tvm.target.Target(target, host=target_host) instead."
        )

    target, target_host = Target.check_and_update_host_consist(
        target, target_host, target_is_dict_key=False
    )

    target = build_target_by_device_type_map(target)
    if isinstance(target_host, (str, Target)):
        target_host = Target(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " + "tvm.target.Target, or None")

    # Retrieve the executor from the target
    executor = get_executor_from_target(target, target_host)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        executor_config, runtime_mod, params = bld_mod.build(
            mod=ir_mod, target=target, params=params, executor=executor, mod_name=mod_name
        )
        func_metadata = bld_mod.get_function_metadata()

        if executor == "aot":
            executor_factory = _executor_factory.AOTExecutorFactoryModule(
                ir_mod, target, runtime_mod, mod_name, params, func_metadata
            )
        elif executor == "graph":
            executor_factory = _executor_factory.GraphExecutorFactoryModule(
                ir_mod, target, executor_config, runtime_mod, mod_name, params, func_metadata
            )
        else:
            assert False, "Executor " + executor + " not supported"

        return executor_factory


def optimize(mod, target=None, params=None):
    """Helper function that optimizes a Relay module.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to build. Using relay.Function is deprecated.

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context
    name) to str/tvm.target.Target, optional
        For heterogeneous compilation, it is a dictionary indicating context to
        target mapping. For homogeneous compilation, it is a build target.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    mod : :py:class:`~tvm.IRModule`
        The optimized relay module.

    params : dict
        The parameters of the final graph.
    """
    if not isinstance(mod, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(mod, _function.Function):
        if params:
            mod = bind_params_by_name(mod, params)
        mod = IRModule.from_expr(mod)
        warnings.warn(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter func (tvm.relay.function.Function)",
            DeprecationWarning,
        )

    target = build_target_by_device_type_map(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        mod, params = bld_mod.optimize(mod, target, params)
    return mod, params


def bind_params_by_name(func, params):
    """Bind params to function by name.
    This could be useful when assembling custom Relay optimization
    passes that involve constant folding.

    Parameters
    ----------
    func : relay.Function
        The function to bind parameters to.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    func : relay.Function
        The function with parameters bound
    """
    inputs = _convert_param_map(params)
    return _build_module.BindParamsByName(func, inputs)


class GraphExecutor(_interpreter.Executor):
    """Wrapper around Executor interface.

    This executor is used for debug and testing purposes.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to support the execution.

    device : :py:class:`Device`
        The runtime device to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """

    def __init__(self, mod, device, target):
        assert mod is not None
        self.mod = mod
        self.device = device
        self.target = target

    def _make_executor(self, expr=None):
        if expr:
            self.mod["main"] = expr
        self.mod = InferType()(self.mod)
        ret_type = self.mod["main"].checked_type.ret_type
        if _ty.is_dynamic(ret_type):
            raise ValueError(
                "Graph Executor only supports static graphs, got output type", ret_type
            )
        mod = build(self.mod, target=self.target)
        gmodule = _graph_rt.GraphModule(mod["default"](self.device))

        def _unflatten(flat_iter, cur_type):
            if isinstance(cur_type, _ty.TensorType):
                return next(flat_iter)
            if isinstance(cur_type, _ty.TupleType):
                fields = []
                for field_type in cur_type.fields:
                    field = _unflatten(flat_iter, field_type)
                    fields.append(field)
                return fields
            raise ValueError("Return type", ret_type, "contains unsupported type", cur_type)

        def _graph_wrapper(*args, **kwargs):
            args = self._convert_args(self.mod["main"], args, kwargs)
            # Create map of inputs.
            for i, arg in enumerate(args):
                gmodule.set_input(i, arg)
            # Run the module, and fetch the output.
            gmodule.run()
            flattened = []
            for i in range(gmodule.get_num_outputs()):
                flattened.append(gmodule.get_output(i).copyto(_nd.cpu(0)))
            unflattened = _unflatten(iter(flattened), ret_type)
            return unflattened

        return _graph_wrapper


# TODO(mbs): Collapse the create_executor/evaluate phases together since a) most callers don't
# reuse the executor for multiple expressions and b) any preparation necessary for the expression
# evaluation needs to (currently) be done along with preparation for the module.
def create_executor(kind="debug", mod=None, device=None, target="llvm", params=None):
    """Factory function to create an executor.

    Example
    -------
    .. code-block:: python

        import tvm.relay
        import numpy as np

        x = tvm.relay.var("x", tvm.relay.TensorType([1], dtype="float32"))
        expr = tvm.relay.add(x, tvm.relay.Constant(tvm.nd.array(np.array([1], dtype="float32"))))
        tvm.relay.create_executor(
            kind="vm", mod=tvm.IRModule.from_expr(tvm.relay.Function([x], expr))
        ).evaluate()(np.array([2], dtype="float32"))
        # returns `array([3.], dtype=float32)`

    Parameters
    ----------
    kind : str
        The type of executor. Avaliable options are `debug` for the
        interpreter, `graph` for the graph executor, and `vm` for the virtual
        machine.

    mod : :py:class:`~tvm.IRModule`
        The Relay module containing collection of functions

    device : :py:class:`Device`
        The device to execute the code.

    target : :py:class:`tvm.Target`
        The corresponding context

    params : dict of str to NDArray
         Input parameters to the graph that do not change
         during inference time.

    Returns
    -------
    executor : :py:class:`~tvm.relay.backend.interpreter.Executor`
    """
    if mod is None:
        mod = IRModule()
    if device is not None:
        assert device.device_type == _nd.device(str(target), 0).device_type
    else:
        device = _nd.device(str(target), 0)

    if params is not None:
        mod = IRModule.from_expr(bind_params_by_name(mod["main"], params))

    if isinstance(target, str):
        target = Target(target)
    if kind == "debug":
        return _interpreter.Interpreter(mod, device, target)
    if kind == "graph":
        return GraphExecutor(mod, device, target)
    if kind == "vm":
        return VMExecutor(mod, device, target)
    raise RuntimeError("unknown execution strategy: {0}".format(kind))
