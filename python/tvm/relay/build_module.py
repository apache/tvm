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

from tvm import expr as tvm_expr
from .. import nd as _nd, target as _target, autotvm
from ..contrib import graph_runtime as _graph_rt
from . import _build_module
from . import ty as _ty
from . import expr as _expr
from .module import Module as _Module
from .backend import interpreter as _interpreter
from .backend.vm import VMExecutor

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

        # Setup the params.
        if params:
            self._set_params(params)
        # Build the function
        self._build(func, target, target_host)
        # Get artifacts
        graph_json = self.get_json()
        mod = self.get_module()
        params = self.get_params()

        return graph_json, mod, params

    def _set_params(self, params):
        inputs = {}
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                param = _nd.array(param)
            inputs[name] = _expr.const(param)
        self._set_params_func(inputs)

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


def build(mod, target=None, target_host=None, params=None):
    """Helper function that builds a Relay function to run on TVM graph
    runtime.

    Parameters
    ----------
    mod : relay.Module
        The module to build. Using relay.Function is deprecated.

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
    if isinstance(mod, _Module):
        func = mod["main"]
    elif isinstance(mod, _expr.Function):
        func = mod
        warnings.warn(
            "Please use input parameter mod (tvm.relay.module.Module) "
            "instead of deprecated parameter func (tvm.relay.expr.Function)",
            DeprecationWarning)
    else:
        raise ValueError("Type of input parameter mod must be tvm.relay.module.Module")

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
        graph_json, mod, params = bld_mod.build(func, target, target_host, params)
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
        assert mod is not None
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, expr=None):
        if expr:
            self.mod["main"] = expr
        ret_type = self.mod["main"].checked_type.ret_type
        num_outputs = len(ret_type.fields) if isinstance(ret_type, _ty.TupleType) else 1
        graph_json, mod, params = build(self.mod, target=self.target)
        gmodule = _graph_rt.create(graph_json, mod, self.ctx)
        if params:
            gmodule.set_input(**params)

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
    if mod is None:
        mod = _Module()
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
