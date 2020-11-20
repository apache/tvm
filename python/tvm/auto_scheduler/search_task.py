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

""" The definiton of SearchTask """

import json

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api
from .workload_registry import register_workload_tensors


@tvm._ffi.register_object("auto_scheduler.SearchTask")
class SearchTask(Object):
    """The computation information and hardware parameters for a schedule search task.

    Parameters
    ----------
    dag : ComputeDAG
        The ComputeDAG for the corresponding compute declaration.
    workload_key : str
        The workload key for the corresponding compute declaration.
    target : tvm.target.Target
        The target device of this search task.
    target_host : Optional[tvm.target.Target]
        The target host device of this search task.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used in this search task.
    """

    def __init__(self, dag, workload_key, target, target_host=None, hardware_params=None):
        self.dag = dag
        self.workload_key = workload_key
        self.target = target
        self.target_host = target_host
        self.hardware_params = hardware_params
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask, dag, workload_key, target, target_host, hardware_params
        )

    def __getstate__(self):
        return {
            "dag": self.dag,
            "workload_key": self.workload_key,
            "target": self.target,
            "target_host": self.target_host,
            "hardware_params": self.hardware_params,
        }

    def __setstate__(self, state):
        self.dag = state["dag"]
        self.workload_key = state["workload_key"]

        # Register the workload if needed
        try:
            workload = json.loads(self.workload_key)
        except Exception:  # pylint: disable=broad-except
            raise RuntimeError("Invalid workload key %s" % self.workload_key)

        # The workload from a compute DAG does not have arguments and is not registered
        # by default so we register it here. If the workload has already been registered,
        # the later registration overrides the prvious one.
        if len(workload) == 1:
            register_workload_tensors(workload[0], self.dag.tensors)

        self.target = state["target"]
        self.target_host = state["target_host"]
        self.hardware_params = state["hardware_params"]
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            self.dag,
            self.workload_key,
            self.target,
            self.target_host,
            self.hardware_params,
        )
