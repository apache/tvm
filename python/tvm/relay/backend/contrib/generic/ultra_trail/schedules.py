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

# TODO
def insert_extern_calls(sch):
    return sch

def schedule_supported_ops(sch):
    block_rvs = sch.get_child_blocks(sch.get_block("root"))
    blocks = [sch.get_sref(block_rv).stmt for block_rv in block_rvs]

    sch.compute_inline(sch.get_block("pad_temp"))
    n, k, x, c, f = sch.get_loops(sch.get_block("conv1d_ncw"))
    sch.reorder(n, k, c, f, x)
    # sch.reverse_compute_at(sch.get_block("T_relu"), sch.get_loops(sch.get_block("conv1d_ncw"))[1])
    # k_o, k_i = sch.split(k, factors=[None, 8])
    # c_o, c_i = sch.split(c, factors=[None, 8])

    breakpoint()
    return sch
