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
import numpy as np

from tvm._ffi.runtime_ctypes import TVMContext
from tvm import expr as tvm_expr
from .. import nd as _nd, target as _target, autotvm
from ..contrib import graph_runtime as _graph_rt
from . import _build_module
from . import ir_pass
from . import ty as _ty
from . import expr as _expr
from .backend import interpreter as _interpreter
from .backend.vm import VMExecutor

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
        "disable_pass": None,
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


BuildConfig.current = BuildConfig()


def build_config(**kwargs):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    opt_level: int, default=2
        Optimization level. See OPT_PASS_LEVEL for level of each pass.

    add_pass: set of str
        Optimization pass to be added regardless of optimization level.

    disable_pass: set of str
        Optimization pass to be disabled during optimization.

    fallback_device : str or tvm.TVMContext
        The fallback device. It is also used as the default device for
        operators without specified device during heterogeneous execution.

    Returns
    -------
    config: BuildConfig
        The build configuration
    """
    return BuildConfig(**kwargs)


def _update_target(target):
    target = target if target else _target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")

    tgts = {}
    if isinstance(target, (str, _target.Target)):
        dev_type = tvm_expr.IntImm("int32", _nd.context(str(target)).device_type)
        tgts[dev_type] = _target.create(target)
    elif isinstance(target, dict):
        for dev, tgt in target.items():
            dev_type = tvm_expr.IntImm("int32", _nd.context(dev).device_type)
            tgts[dev_type] = _target.create(tgt)
    else:
        raise TypeError("target is expected to be str or " +
                        "tvm.target.Target, but received " +
                        "{}".format(type(target)))
    return tgts


class BuildModule(object):
    """Build a Relay function to run on TVM graph runtime. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """
    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._add_pass = self.mod["add_pass"]
        self._disable_pass = self.mod["disable_pass"]
        self._set_opt_level = self.mod["set_opt_level"]
        self._set_fallback_device = self.mod["set_fallback_device"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]

    def build(self, func, target=None, target_host=None, params=None):
        """
        Parameters
        ----------
        func: relay.Function
            The function to build.

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

        # Setup the build configurations passed in through `with build_config`.
        self._setup_build_config(params)
        # Build the function
        self._build(func, target, target_host)
        # Get artifacts
        graph_json = self.get_json()
        mod = self.get_module()
        params = self.get_params()

        return graph_json, mod, params

    def _setup_build_config(self, params):
        cfg = BuildConfig.current

        # Set opt_level.
        self.set_opt_level(cfg.opt_level)

        # Set fallback device if it is available.
        if cfg.fallback_device:
            self.set_fallback_device(cfg.fallback_device)

        # Add required passes.
        if cfg.add_pass:
            passes = set()
            if isinstance(cfg.add_pass, (list, tuple, set)):
                passes = set(cfg.add_pass)
            else:
                raise TypeError("add_pass must be list, tuple, or set, but " +
                                "got {}".format(type(cfg.add_pass)))
            for pass_name in passes:
                self.add_pass(pass_name)

        # Add disabled passes.
        if cfg.disable_pass:
            passes = set()
            if isinstance(cfg.disable_pass, (list, tuple, set)):
                passes = set(cfg.disable_pass)
            else:
                raise TypeError("disable_pass must be list, tuple, or set, " +
                                "but got {}".format(type(cfg.disable_pass)))
            for pass_name in passes:
                self.disable_pass(pass_name)

        if params:
            self._set_params(params)

    def _set_params(self, params):
        inputs = {}
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                param = _nd.array(param)
            inputs[name] = _expr.const(param)
        self._set_params_func(inputs)

    def add_pass(self, pass_name):
        """Add a pass to the pass list.

        Parameters
        ----------
        pass_name : str
            The name of the pass that will be added to the list of passes used
            for optimizations.
        """
        self._add_pass(pass_name)

    def disable_pass(self, pass_name):
        """Add a pass to the disabled pass list.

        Parameters
        ----------
        pass_name : str
            The name of a pass. This pass will be added to the list of passes
            that are disabled during optimization.
        """
        self._disable_pass(pass_name)

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

    def set_opt_level(self, level):
        """Set the optimization level.

        Parameters
        ----------
        level : int
            The optimization level for build.
        """
        self._set_opt_level(level)

    def set_fallback_device(self, fallback_device):
        """Set the fallback device for heterogeneous execution.

        Parameters
        ----------
        fallback_device : str or tvm.TVMContext
            The fallback device used for heterogeneous execution.
        """
        if isinstance(fallback_device, str):
            fallback_device = _nd.context(fallback_device)
        if not isinstance(fallback_device, TVMContext):
            raise TypeError("fallback_device is expected to be str " +
                            "TVMContext, or dict of device name to target, " +
                            "but received: {}".format(type(fallback_device)))

        self._set_fallback_device(fallback_device.device_type)


def build(func, target=None, target_host=None, params=None):
    """Helper function that builds a Relay function to run on TVM graph
    runtime.

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
    target = _update_target(target)

    if isinstance(target_host, (str, _target.Target)):
        target_host = _target.create(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " +
                         "tvm.target.Target, or None")

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.util.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        graph_json, mod, params = bld_mod.build(func, target, target_host,
                                                params)
    return graph_json, mod, params


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
        ret_type = ir_pass.infer_type(func).ret_type
        num_outputs = len(ret_type.fields) if isinstance(ret_type, _ty.TupleType) else 1
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
            if num_outputs == 1:
                return gmodule.get_output(0).copyto(_nd.cpu(0))
            outputs = []
            for i in range(num_outputs):
                outputs.append(gmodule.get_output(i).copyto(_nd.cpu(0)))
            return outputs

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
    elif kind == "vm":
        return VMExecutor(mod, ctx, target)
    else:
        raise RuntimeError("unknown execution strategy: {0}".format(kind))
