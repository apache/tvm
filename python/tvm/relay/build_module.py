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
Construct the necessary state for the TVM graph runtime
from a Relay expression.
"""
import warnings
import numpy as np

from tvm.ir import IRModule

from tvm.tir import expr as tvm_expr
from .. import nd as _nd, autotvm
from ..target import Target
from ..contrib import graph_runtime as _graph_rt
from . import _build_module
from . import ty as _ty
from . import expr as _expr
from . import function as _function
from .backend import graph_runtime_factory as _graph_runtime_factory
from .backend import interpreter as _interpreter
from .backend.vm import VMExecutor


def _update_target(target):
    target = target if target else Target.current()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")

    tgts = {}
    if isinstance(target, (str, Target)):
        dev_type = tvm_expr.IntImm("int32", _nd.context(str(target)).device_type)
        tgts[dev_type] = Target(target)
    elif isinstance(target, dict):
        for dev, tgt in target.items():
            dev_type = tvm_expr.IntImm("int32", _nd.context(dev).device_type)
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
    """Build an IR module to run on TVM graph runtime. This class is used
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

    def build(self, mod, target=None, target_host=None, params=None):
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
        target = _update_target(target)

        # Setup the params.
        if params:
            self._set_params(params)
        # Build the IR module
        self._build(mod, target, target_host)
        # Get artifacts
        graph_json = self.get_json()
        mod = self.get_module()
        params = self.get_params()

        return graph_json, mod, params

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
        target = _update_target(target)

        # Setup the params.
        if params:
            self._set_params(params)
        mod = self._optimize(mod, target)
        # Get artifacts
        params = self.get_params()

        return mod, params

    def _set_params(self, params):
        self._set_params_func(_convert_param_map(params))

    def get_json(self):
        """Return the json file of the built program."""
        return self._get_graph_json()

    def get_module(self):
        """Return the built module."""
        return self._get_module()

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret


def build(mod, target=None, target_host=None, params=None, mod_name="default"):
    """Helper function that builds a Relay function to run on TVM graph
    runtime.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

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

    mod_name: Optional[str]
        The module name we will build

    Returns
    -------
    graph_json : str
        The json string that can be accepted by graph runtime.

    mod : tvm.Module
        The module containing necessary libraries.

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
            "instead of deprecated parameter mod (tvm.relay.function.Function)",
            DeprecationWarning,
        )

    target = _update_target(target)

    if isinstance(target_host, (str, Target)):
        target_host = Target(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " + "tvm.target.Target, or None")

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.util.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        graph_json, mod, params = bld_mod.build(mod, target, target_host, params)
        mod = _graph_runtime_factory.GraphRuntimeFactoryModule(graph_json, mod, mod_name, params)
        return mod


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

    target = _update_target(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.util.EmptyContext()

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

    This executor is used for debug and testing purpoes.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to support the execution.

    ctx : :py:class:`TVMContext`
        The runtime context to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """

    def __init__(self, mod, ctx, target):
        assert mod is not None
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, expr=None):
        if expr:
            self.mod["main"] = expr
        ret_type = self.mod["main"].checked_type.ret_type
        if _ty.is_dynamic(ret_type):
            raise ValueError("Graph Runtime only supports static graphs, got output type", ret_type)
        num_outputs = len(ret_type.fields) if isinstance(ret_type, _ty.TupleType) else 1
        mod = build(self.mod, target=self.target)
        gmodule = _graph_rt.GraphModule(mod["default"](self.ctx))

        def _graph_wrapper(*args, **kwargs):
            args = self._convert_args(self.mod["main"], args, kwargs)
            # Create map of inputs.
            for i, arg in enumerate(args):
                gmodule.set_input(i, arg)
            # Run the module, and fetch the output.
            gmodule.run()
            # make a copy so multiple invocation won't hurt perf.
            if num_outputs == 1:
                return gmodule.get_output(0).copyto(_nd.cpu(0))
            outputs = []
            for i in range(num_outputs):
                outputs.append(gmodule.get_output(i).copyto(_nd.cpu(0)))
            return outputs

        return _graph_wrapper


def create_executor(kind="debug", mod=None, ctx=None, target="llvm"):
    """Factory function to create an executor.

    Parameters
    ----------
    kind : str
        The type of executor

    mod : :py:class:`~tvm.IRModule`
        The Relay module containing collection of functions

    ctx : :py:class:`tvmContext`
        The context to execute the code.

    target : :py:class:`tvm.Target`
        The corresponding context
    """
    if mod is None:
        mod = IRModule()
    if ctx is not None:
        assert ctx.device_type == _nd.context(str(target), 0).device_type
    else:
        ctx = _nd.context(str(target), 0)

    if isinstance(target, str):
        target = Target(target)
    if kind == "debug":
        return _interpreter.Interpreter(mod, ctx, target)
    if kind == "graph":
        return GraphExecutor(mod, ctx, target)
    if kind == "vm":
        return VMExecutor(mod, ctx, target)
    raise RuntimeError("unknown execution strategy: {0}".format(kind))
