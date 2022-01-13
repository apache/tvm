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
"""Relay partitioner for the UltraTrail accelerator"""

import tvm
from tvm import relay

custom_target_name = "ultra_trail"


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, f"target.{custom_target_name}")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


def partition_for_ultra_trail(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    _register_external_op_helper("nn.conv1d")
    mod = relay.transform.AnnotateTarget(custom_target_name)(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    return mod
