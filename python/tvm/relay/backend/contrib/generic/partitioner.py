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
        pattern = relay.op.contrib.get_pattern_table(self.target_name)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = relay.transform.AnnotateTarget(self.target_name)(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        mod = relay.transform.InferType()(mod)
        # Defunctionalize the partitioned functions to allow lowering
        for gv, func in mod.functions.items():
            mod.update_func(gv, relay.transform.Defunctionalization(func, mod))

        return mod
