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

import tvm
from tvm import te, tir
from tvm.ir.module import IRModule
from tvm.script import tir as T


def relu_te_compute(Input, out_shape, dtype):
    x      = tvm.tir.const(0, dtype)
    Output = te.compute(
               out_shape, 
                 lambda n, h, w, c: 
                   tvm.te.max(Input[n, h, w, c], x),
                     name="reluf16")
    return Output

def reluf16_te_sched(Output, Input, transform_crouton_activation):
    s = tvm.te.create_schedule(Output.op)
    s[Input].transform_layout(transform_crouton_activation)
    out_axes	= s[Output].transform_layout(transform_crouton_activation)
    fused 	= s[Output].fuse(out_axes[6], out_axes[7])
    s[Output].vectorize(fused)
    return s

def reluf16_stir_sched(func, transform_crouton_activation):
    sch   = tir.Schedule(func, debug_mask="all")
    block = sch.get_block("reluf16")
    n, i, j, k = sch.get_loops(block)
    i1, i2 = sch.split(i,  [None,  8])
    j1, j2 = sch.split(j,  [None,  4])
    k1, k2 = sch.split(k,  [None, 32])
    j3, j4 = sch.split(j2, [None,  2])
    sch.reorder(n, i1, j1, k1, i2, j3, k2, j4)
    sch.transform_layout  (block, 0, "read",  transform_crouton_activation)
    sch.set_axis_separator(block, 0, "read",  [4])
    sch.transform_layout  (block, 0, "write", transform_crouton_activation)
    sch.set_axis_separator(block, 0, "write", [4])
    fused = sch.fuse(k2, j4)
    sch.vectorize(fused)
    return sch
