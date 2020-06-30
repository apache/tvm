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
from tvm.runtime import _ffi_api
from . import vm


def enabled():
    """Whether vm profiler is enabled."""
    return hasattr(_ffi_api, "_VirtualMachineDebug")


class VirtualMachineProfiler(vm.VirtualMachine):
    """Relay profile VM runtime."""

    def __init__(self, mod):
        super(VirtualMachineProfiler, self).__init__(mod)
        m = mod.module if isinstance(mod, vm.Executable) else mod
        self.mod = _ffi_api._VirtualMachineDebug(m)
        self._init = self.mod["init"]
        self._invoke = self.mod["invoke"]
        self._get_stat = self.mod["get_stat"]
        self._set_input = self.mod["set_input"]
        self._reset = self.mod["reset"]

    def get_stat(self, sort_by_time=True):
        """Get the statistics of executed ops.

        Parameters
        ----------
        sort_by_time: Optional[Boolean]
           Set to indicate the returned results are sorted by execution time in
           the descending order. It is printed in the random order if this
           field is not set.

        Returns
        -------
            The execution statistics in string.
        """
        return self._get_stat(sort_by_time)

    def reset(self):
        self._reset()
