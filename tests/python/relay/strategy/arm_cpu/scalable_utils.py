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
from tvm.tir.stmt_functor import post_order_visit, ir_transform


def calculate_extra_workspace_size_from_scalable_extents(func, known_vscale_value):
    """
    The AOT executor needs to know the size of the workspace ahead of time, but this
    isn't possible when some allocations are scalable (vscale is not known at compile-time).
    If we know the target hardware, we can reason about the value of vscale ahead of time.
    This function will calculate an upper-bound for the extra workspace bytes required by the
    AOT executor given TIR function and a known value for vscale.
    """
    extra_workspace_bytes = 0
    is_scalable_extent = False
    ana = tvm.arith.Analyzer()

    def replace_vscale_with_known_value(stmt):
        nonlocal is_scalable_extent
        if isinstance(stmt, tvm.tir.expr.Call) and stmt.op.name == "tir.vscale":
            is_scalable_extent = True
            return tvm.tir.IntImm(stmt.dtype, known_vscale_value)

    def calculate_workspace_bytes(stmt):
        nonlocal extra_workspace_bytes, is_scalable_extent
        if isinstance(stmt, tvm.tir.stmt.Allocate):
            for extent in stmt.extents:
                extent_stmt = tvm.tir.Evaluate(extent)
                is_scalable_extent = False
                mutated_extent = ir_transform(extent_stmt, replace_vscale_with_known_value, None)
                # Non scalable extents are already included in the calculation by AOT
                if is_scalable_extent:
                    alloc_bytes = ana.simplify(mutated_extent.value) * tvm.DataType(stmt.dtype).bits
                    extra_workspace_bytes += alloc_bytes

    post_order_visit(func.body, calculate_workspace_bytes)
    return extra_workspace_bytes
