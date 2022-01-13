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
"""Generic relay partitioner for NPUs"""

import tvm
from tvm import relay
from typing import Callable
from abc import abstractmethod


class GenericPartitioner(object):
    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the hardware target.

        Returns
        -------
        out : str
            The hardware target name.
        """

    @abstractmethod
    def _register_supported_ops(self) -> None:
        """Register a set of supported relay operations which are applied to the schedule.

        Example
        -------
        Here is an example of how two supported operations can be registered.

        .. code-block:: python

            def _register_supported_ops(self):
                self._register_supported_op(op_0)
                self._register_supported_op(op_1)
        """
        pass

    def _register_supported_op(self, op: str) -> Callable:
        @tvm.ir.register_op_attr(op, "target.{}".format(self.target_name))
        def _func_wrapper(_):
            return True

        return _func_wrapper

    def __call__(self, mod: tvm.IRModule) -> tvm.IRModule:
        """Partition the relay graph in by the NPU supported and unsupported parts.

        Parameters
        ----------
        mod : tvm.IRModule
            The relay module to be partitioned.

        Returns
        -------
        out : tvm.IRModule
            The partitioned relay module.

        """
        self._register_supported_ops()
        mod = relay.transform.AnnotateTarget(self.target_name)(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        return mod
