# License .to the Apache Software Foundation (ASF) under one
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
The Relay Virtual Machine profiler.

Provides extra APIs for profiling vm execution.
"""
from . import vm, _vm

def compile(mod, target=None, target_host=None, params=None):
    """
    Parameters
    ----------
    mod : relay.Module
        The Relay module to build.

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
    vm : VirtualMachineProfiler
        The profile VM runtime.
    """
    compiler = VMCompilerProfiler()
    target = compiler.update_target(target)
    target_host = compiler.update_target_host(target, target_host)
    if params:
        compiler.set_params(params)
    tophub_context = compiler.tophub_context(target)
    with tophub_context:
        compiler._compile(mod, target, target_host)
    return VirtualMachineProfiler(compiler._get_vm())

class VMCompilerProfiler(vm.VMCompiler):
    """Build Relay module to run on VM runtime."""
    def __init__(self):
        super().__init__()
        self.mod = _vm._VMCompilerProfiler()
        self._compile = self.mod["compile"]
        self._get_vm = self.mod["get_vm"]
        self._set_params_func = self.mod["set_params"]

class VirtualMachineProfiler(vm.VirtualMachine):
    """Relay profile VM runtime."""
    def __init__(self, mod):
        super().__init__(mod)
        self._get_stat = self.mod["get_stat"]

    def get_stat(self):
        return self._get_stat()
