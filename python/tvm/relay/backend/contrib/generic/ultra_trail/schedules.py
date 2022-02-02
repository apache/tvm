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
"""TIR schedule functions for the UltraTrail accelerator"""

from tvm.topi.utils import prod
from tvm import tir
from tvm.script import tir as T

# create one load buffer extern_call for each buffer_var (input/weights)
# - dont reset counter, only for first
# - packed buffers, correct layout, take care of missalignment at the end (software?,hardware?)
# create one load buffer for config
def insert_extern_calls(sch):
    def extern_calls():
        calls = []
        buffer_scopes = list(sch.mod["main"].attrs["relay_attrs"]["ut_buffer_scopes"])
        buffer_scopes.reverse() # for some reason TIR params are reversed to relay function
        for i, buffer_scope in enumerate(buffer_scopes):
            buffer = sch.mod["main"].buffer_map[sch.mod["main"].params[i]]
            size = prod(buffer.shape)
            var = buffer.data
            call = tir.call_extern("int32", f"load_{buffer_scope}", var, size)
            calls.append(tir.Evaluate(call))
        seq = tir.stmt_seq(*calls)
        return tir.Block([], [], [], "call_extern", seq)

    root_sref = sch.get_sref(sch.get_block("root"))
    sch.state.replace(root_sref, extern_calls())

    return sch
