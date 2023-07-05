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
from tvm.target import Target

from .. import autotvm
from .. import nd as _nd
from .. import register_func
from ..contrib import graph_executor as _graph_executor
from ..contrib import utils as contrib_utils
from ..runtime import load_module
from ..runtime.executor import aot_executor as _aot_executor
from ..target import Target
from . import _build_module
from . import expr as _expr
from . import function as _function
from . import ty as _ty
from .backend import Executor, Runtime
from .backend import executor_factory as _executor_factory
from .backend import interpreter as _interpreter
from .backend.utils import mangle_module_name
from .backend.vm import VMExecutor
from .transform import InferType


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
        self._get_executor_codegen_metadata = self.mod["get_executor_codegen_metadata"]
        self._get_devices = self.mod["get_devices"]
        self._get_irmodule = self.mod["get_irmodule"]

    def build(
        self,
        mod,
        target=None,
        target_host=None,
        executor=Executor("graph"),
        runtime=Runtime("cpp"),
        workspace_memory_pools=None,
        constant_memory_pools=None,
        params=None,
        mod_name=None,
    ):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : any multi-target like object, see Target.canon_multi_target
            For homogeneous compilation, the unique build target.
            For heterogeneous compilation, a dictionary or list of possible build targets.

        target_host : None, or any target-like object, see Target.canon_target
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm interpreter is used.

        executor : Optional[Executor]
            The executor configuration with which to build the model.
            Defaults to "graph" if no executor specified.

        runtime : Optional[Runtime]
            Runtime configuration to use when building the model.
            Defaults to "cpp" if no runtime specified.

        workspace_memory_pools : Optional[WorkspaceMemoryPools]
            The object that contains an Array of WorkspacePoolInfo objects
            that hold properties of read-write workspace pools that could be
            used by the inference.

        constant_memory_pools : Optional[ConstantMemoryPools]
            The object that contains an Array of ConstantPoolInfo objects
            that hold properties of read-only memory pools that could be
            used by the inference.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

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
        # pylint: disable=import-outside-toplevel
        from tvm.auto_scheduler import is_auto_scheduler_enabled
        from tvm.meta_schedule import is_meta_schedule_enabled

        # pylint: enable=import-outside-toplevel
        # Setup the params.
        if params:
            self._set_params(params)

        # Build the IR module. If auto_scheduler is not enabled,
        # then use the TOPI-defined schedule.

        # Turn off AutoTVM config not found warnings if auto_scheduler is enabled.
        old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = (
            is_auto_scheduler_enabled() or is_meta_schedule_enabled() or old_autotvm_silent
        )

        mod_name = mangle_module_name(mod_name)

        self._build(
            mod,
            target,
            target_host,
            executor,
            runtime,
            workspace_memory_pools,
            constant_memory_pools,
            mod_name,
        )
        autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

        # Get artifacts
        mod = self.get_module()
        params = self.get_params()
        executor_config = self.get_graph_json() if executor.name == "graph" else None

        return executor_config, mod, params

    def optimize(self, mod, target=None, target_host=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IR module to build.

        target : any multi-target like object, see Target.canon_multi_target.
            For homogeneous compilation, the unique build target.
            For heterogeneous compilation, a dictionary or list of possible build targets.

        target_host : None, or any target-like object, see Target.canon_target
            Host compilation target, if target is device.

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
        raw_targets = Target.canon_multi_target_and_host(target, target_host)

        # Setup the params.
        if params:
            self._set_params(params)
        mod = self._optimize(mod, raw_targets)
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

    def get_executor_codegen_metadata(self):
        """Return the metadata produced after executor
        codegen
        """
        return self._get_executor_codegen_metadata()

    def get_devices(self):
        """Returns a list of devices configured in this module"""
        return self._get_devices()

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret

    def get_irmodule(self):
        """Returns the TargetIRModule's post-lowering"""
        return self._get_irmodule()


@register_func("tvm.relay.module_export_library")
def _module_export(module, file_name):  # fcompile, addons, kwargs?
    return module.export_library(file_name)


@register_func("tvm.relay.build")
def _build_module_no_factory_impl(mod, target, target_host, params, mod_name):
    return build(
        mod, target=target, target_host=target_host, params=params, mod_name=mod_name
    ).module


def _build_module_no_factory(mod, target=None, target_host=None, params=None, mod_name="default"):
    """A wrapper around build which discards the Python GraphFactoryRuntime.
    This wrapper is suitable to be used from other programming languages as
    the runtime::Module can be freely passed between language boundaries.
    """
    return _build_module_no_factory_impl(mod, target, target_host, params, mod_name)


def build(
    ir_mod,
    target=None,
    target_host=None,
    executor=Executor("graph"),
    runtime=Runtime("cpp"),
    workspace_memory_pools=None,
    constant_memory_pools=None,
    params=None,
    mod_name="default",
):
    # fmt: off
    # pylint: disable=line-too-long
    """Helper function that builds a Relay function to run on TVM graph executor.

    Parameters
    ----------
    ir_mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

    target : None, or any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
        Defaults to the current target in the environment if None.

    target_host : None, or any target like object, see Target.canon_target
        Host compilation target, if target is device.

    executor : Optional[Executor]
        The executor configuration with which to build the model.
        Defaults to "graph" if no executor specified.

    runtime : Optional[Runtime]
        Runtime configuration to use when building the model.
        Defaults to "cpp" if no runtime specified.

    workspace_memory_pools : Optional[WorkspaceMemoryPools]
        The object that contains an Array of WorkspacePoolInfo objects
        that hold properties of read-write workspace pools that could be
        used by the inference.

    constant_memory_pools : Optional[ConstantMemoryPools]
        The object that contains an Array of ConstantPoolInfo objects
        that hold properties of read-only pools that could be
        used by the inference.

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

    raw_targets = Target.canon_multi_target_and_host(Target.target_or_current(target), target_host)
    assert len(raw_targets) > 0
    target_host = raw_targets[0].host

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(raw_targets))
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        graph_json, runtime_mod, params = bld_mod.build(
            mod=ir_mod,
            target=raw_targets,
            params=params,
            executor=executor,
            runtime=runtime,
            workspace_memory_pools=workspace_memory_pools,
            constant_memory_pools=constant_memory_pools,
            mod_name=mod_name,
        )
        func_metadata = bld_mod.get_function_metadata()
        devices = bld_mod.get_devices()
        lowered_ir_mods = bld_mod.get_irmodule()
        executor_codegen_metadata = bld_mod.get_executor_codegen_metadata()

        if executor.name == "aot":
            executor_factory = _executor_factory.AOTExecutorFactoryModule(
                ir_mod,
                lowered_ir_mods,
                raw_targets,
                executor,
                runtime,
                runtime_mod,
                mod_name,
                params,
                func_metadata,
                executor_codegen_metadata,
                devices,
            )
        elif executor.name == "graph":
            executor_factory = _executor_factory.GraphExecutorFactoryModule(
                ir_mod,
                raw_targets,
                executor,
                graph_json,
                runtime_mod,
                mod_name,
                params,
                func_metadata,
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

    target : None, or any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
        Defaults to the current target in the environment if None.

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

    raw_targets = Target.canon_multi_target_and_host(Target.target_or_current(target))

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(raw_targets)
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        mod, params = bld_mod.optimize(mod, target=raw_targets, params=params)
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

    target : any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
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
        gmodule = _graph_executor.GraphModule(mod["default"](self.device))

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


class AotExecutor(_interpreter.Executor):
    """Implements the Executor interface for AOT.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to support the execution.

    device : :py:class:`Device`
        The runtime device to run the code on.

    target : any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
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
            raise ValueError("AOT Executor only supports static graphs, got output type", ret_type)
        mod = build(self.mod, target=self.target, executor=Executor("aot"))

        # NOTE: Given AOT requires use of the "c" backend, must export/import to compile the
        # generated code.
        temp_so_dir = contrib_utils.TempDirectory()
        temp_so = temp_so_dir / "temp.so"
        mod.export_library(temp_so, cc="gcc", options=["-std=c11"])

        mod = load_module(temp_so)
        aot_mod = mod["default"](self.device)
        gmodule = _aot_executor.AotModule(aot_mod)

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

        def _aot_wrapper(*args, **kwargs):
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

        return _aot_wrapper


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
        The type of executor. Avaliable options are `debug` for the interpreter, `graph` for the
        graph executor, `aot` for the aot executor, and `vm` for the virtual machine.

    mod : :py:class:`~tvm.IRModule`
        The Relay module containing collection of functions

    device : :py:class:`Device`
        The device to execute the code.

    target : any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
        CAUTION: Though this API allows multiple targets, it does not allow multiple devices, so
        heterogenous compilation is not yet supported.

    params : dict of str to NDArray
         Input parameters to the graph that do not change
         during inference time.

    Returns
    -------
    executor : :py:class:`~tvm.relay.backend.interpreter.Executor`
    """
    raw_targets = Target.canon_multi_target(target)
    if mod is None:
        mod = IRModule()
    if device is not None:
        assert device.device_type == raw_targets[0].get_target_device_type()
    else:
        # Derive the default device from the first target.
        device = _nd.device(raw_targets[0].get_target_device_type(), 0)

    if params is not None:
        mod = IRModule.from_expr(bind_params_by_name(mod["main"], params))

    assert "executor" not in raw_targets[0].attrs or raw_targets[0].attrs["executor"] == kind

    if kind == "debug":
        assert len(raw_targets) == 1, "The interpreter currently only supports a single target"
        return _interpreter.Interpreter(mod, device, raw_targets[0])
    if kind == "graph":
        return GraphExecutor(mod, device, raw_targets)
    if kind == "vm":
        return VMExecutor(mod, device, raw_targets)
    if kind == "aot":
        return AotExecutor(mod, device, raw_targets)
    raise RuntimeError(f"unknown execution strategy: {kind}")
