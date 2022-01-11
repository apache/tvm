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
"""Codegen for the RP_NPU"""

import tvm
from tvm import relay
from tvm.relay.backend.contrib.generic.codegen import GenericCodegen


class RBNPUCodegen(GenericCodegen):
    def __init__(self):
        super(RBNPUCodegen, self).__init__()

    def apply_schedules(self, schedule):
        return schedule


@tvm._ffi.register_func("relay.ext.generic.relay_to_tir_func_rb_npu")
def relay_to_tir_func_rb_npu(ext_func: relay.Function) -> tvm.tir.PrimFunc:
    """
    This is the hook for python-based lowering of relay function
    that gets offloaded to the RB NPU.

    Parameters
    ----------
    ext_func : relay.Function
        This is the partitioned relay function

    Returns
    -------
    prim_func : tir.PrimFunc
        This returns the scheduled PrimFunc
    """
    codegen = RBNPUCodegen()
    prim_func = codegen.relay_to_tir_func(ext_func)
    return prim_func
