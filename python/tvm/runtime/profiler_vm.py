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
The Relay Virtual Machine profiler.

Provides extra APIs for profiling vm execution.
"""
import warnings
from tvm.runtime import _ffi_api
from tvm.rpc import base as rpc_base
from . import vm
from .profiling import Report


def enabled():
    """Whether vm profiler is enabled."""
    return hasattr(_ffi_api, "_VirtualMachineDebug")


class VirtualMachineProfiler(vm.VirtualMachine):
    """Relay profile VM runtime."""

    def __init__(self, exe, device, memory_cfg=None):
        super(VirtualMachineProfiler, self).__init__(exe, device, memory_cfg)

        # Make sure the constructor of the VM module is on the proper device
        # Remote devices have device_type of their actual device_type + RPC_SESS_MASK
        if device.device_type >= rpc_base.RPC_SESS_MASK:
            self.module = device._rpc_sess.get_function("runtime._VirtualMachineDebug")(exe)
        else:
            self.module = _ffi_api._VirtualMachineDebug(exe.module)

        self._init = self.module["init"]
        self._invoke = self.module["invoke"]
        self._profile = self.module["profile"]
        self._profile_rpc = self.module["profile_rpc"]
        self._set_input = self.module["set_input"]
        self._setup_device(device, memory_cfg)

    def get_stat(self, sort_by_time=True):  # pylint: disable=unused-argument
        """Get the statistics of executed ops.

        REMOVED, use profile method instead.
        """
        warnings.warn("get_stat has been removed, use profile instead")
        return ""

    def profile(self, *args, func_name="main", collectors=None, **kwargs):
        """Profile a function call.

        Parameters
        ----------
        func_name : str
            The name of the function.

        collectors : Optional[Sequence[MetricCollector]]
            Extra metrics to collect. If profiling over RPC, collectors must be `None`.

        args : list[tvm.runtime.NDArray] or list[np.ndarray]
            The arguments to the function.

        kwargs: dict of str to tvm.runtime.NDArray or np.ndarray
            Named arguments to the function.

        Returns
        -------
        timing_results : str
            Overall and per-op timing results formatted in a table.
        """
        if args or kwargs:
            self.set_input(func_name, *args, **kwargs)
        if self.module.type_key == "rpc":
            # We cannot serialize MetricCollectors over RPC
            assert collectors is None, "Profiling with collectors is not supported over RPC"
            return Report.from_json(self._profile_rpc(func_name))
        return self._profile(func_name, collectors)
