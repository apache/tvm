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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
The Relay Virtual Machine.

Implements a Python interface to compiling and executing on the Relay VM.
"""
import numpy as np

import tvm.runtime.ndarray as _nd
import tvm.runtime.vm as vm_rt
from tvm import autotvm
from tvm.relay import expr as _expr
from tvm.relay.backend.interpreter import Executor
from tvm.target import Target
from . import _vm


def compile(mod, target=None, target_host=None, params=None):
    """Compile the module to VM executable. A helper function for VMCompiler.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relay module to build.

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
        otherwise a stackvm intepreter is used.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    Returns
    -------
    exec : tvm.runtime.vm.Executable
        The VM executable that contains both library code and bytecode.
    """
    compiler = VMCompiler()
    if params:
        compiler.set_params(params)
    compiler.lower(mod, target, target_host)
    compiler.codegen()
    return compiler.get_exec()


class VMCompiler(object):
    """Compiler that compiles Relay module to VM executable."""

    def __init__(self):
        self.mod = _vm._VMCompiler()
        self._lower = self.mod["lower"]
        self._codegen = self.mod["codegen"]
        self._get_exec = self.mod["get_executable"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]
        self._optimize = self.mod["optimize"]

    def set_params(self, params):
        """Set constant parameters for the model.

        Parameters
        ----------
        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.
        """
        inputs = {}
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                param = _nd.array(param)
            inputs[name] = _expr.const(param)
        self._set_params_func(inputs)

    def get_params(self):
        """Return the updated weights."""
        params = self._get_params_func()
        ret = {}
        for key, value in params.items():
            ret[key] = value.data
        return ret

    def lower(self, mod, target=None, target_host=None):
        """Lower the module to VM bytecode.

        Parameters
        ----------
        mod : tvm.IRModule
            The Relay module to build.

        target : any multi-target like object, see Target.canon_multi_target
            For homogeneous compilation, the unique build target.
            For heterogeneous compilation, a dictionary or list of possible build targets.

        target_host : any target-like object, see Target.canon_target
            Host compilation target, if target is device.
        """
        raw_targets = Target.canon_multi_target_and_host(target, target_host)
        tophub_context = self._tophub_context(raw_targets)
        with tophub_context:
            self._lower(mod, raw_targets)

    def codegen(self):
        """Generate the kernel library."""
        self._codegen()

    def optimize(self, mod, target=None, target_host=None, params=None):
        """Helper method that optimizes a Relay module via VM.

        Parameters
        ----------
        mod : tvm.IRModule

        target : any multi-target like object, see Target.canon_multi_target
            For homogeneous compilation, the unique build target.
            For heterogeneous compilation, a dictionary or list of possible build targets.

        target_host : any target-like object, see Target.canon_target
            Host compilation target, if target is device.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        mod : tvm.IRModule
            The optimized relay module.

        params : dict
            The parameters of the final module.
        """
        raw_targets = Target.canon_multi_target_and_host(target, target_host)
        if params:
            self.set_params(params)
        return self._optimize(mod, raw_targets), self.get_params()

    def get_exec(self):
        """Get the VM executable.

        Returns
        -------
        exec : tvm.runtime.vm.Executable
            The VM executable that contains both library code and bytecode.
        """
        return vm_rt.Executable(self._get_exec())

    def _tophub_context(self, raw_targets):
        """Get the autotvm context."""
        # If current dispatch context is fallback context (the default root context),
        # then load pre-tuned parameters from TopHub
        if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
            tophub_context = autotvm.tophub.context(raw_targets)
        else:
            tophub_context = autotvm.utils.EmptyContext()
        return tophub_context


class VMExecutor(Executor):
    """
    An implementation of the executor interface for
    the Relay VM.

    Useful interface for experimentation and debugging
    the VM can also be used directly from the API.
    supported by `tvm.runtime.vm`.

    Parameters
    ----------
    mod : :py:class:`~tvm.IRModule`
        The module to support the execution.

    device : :py:class:`~tvm.runtime.Device`
        The runtime device to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """

    def __init__(self, mod, device, target):
        if mod is None:
            raise RuntimeError("Must provide module to get VM executor.")
        self.mod = mod
        self.device = device
        self.target = target
        self.executable = None
        self.vm = None

    def _make_executor(self, expr=None):
        if expr:
            self.mod["main"] = expr

        self.executable = compile(self.mod, self.target)
        self.vm = vm_rt.VirtualMachine(self.executable, self.device)

        def _vm_wrapper(*args, **kwargs):
            args = self._convert_args(self.mod["main"], args, kwargs)
            return self.vm.run(*args)

        return _vm_wrapper
