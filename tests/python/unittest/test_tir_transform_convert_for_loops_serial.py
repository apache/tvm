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
import pytest

import tvm

from tvm.script import tir as T
from tvm.tir import stmt_functor

# fmt: off
@T.prim_func
def fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2(placeholder_30: T.handle, placeholder_31: T.handle, placeholder_32: T.handle, T_cast_8: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", "tir.noalias": True})
    placeholder_33 = T.match_buffer(placeholder_30, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_34 = T.match_buffer(placeholder_31, [1, 1, 192, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_35 = T.match_buffer(placeholder_32, [1, 1, 1, 16], dtype="int32", elem_offset=0, align=128, offset_factor=1)
    T_cast_9 = T.match_buffer(T_cast_8, [1, 28, 28, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    # body
    PaddedInput_3 = T.allocate([1, 28, 28, 192], "int16", "global")
    for i0_i1_fused_3 in T.parallel(0, 28):
        for i2_3, i3_3 in T.grid(28, 192):
            T.store(PaddedInput_3, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3), T.load("int16", placeholder_33.data, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3)), True)
    for ax0_ax1_fused_ax2_fused_3 in T.parallel(0, 784):
        for ax3_2 in T.serial(0, 16):
            Conv2dOutput_3 = T.allocate([1, 1, 1, 1], "int32", "global")
            T.store(Conv2dOutput_3, 0, 0, True)
            for rc_3 in T.serial(0, 192):
                T.store(Conv2dOutput_3, 0, (T.load("int32", Conv2dOutput_3, 0) + (T.cast(T.load("int16", PaddedInput_3, ((ax0_ax1_fused_ax2_fused_3*192) + rc_3)), "int32")*T.cast(T.load("int16", placeholder_34.data, ((rc_3*16) + ax3_2)), "int32"))), True)
            T.store(T_cast_9.data, ((ax0_ax1_fused_ax2_fused_3*16) + ax3_2), T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_3, 0) + T.load("int32", placeholder_35.data, ax3_2)), 1764006585, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16"), True)
# fmt: on


def test_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2():
    primfunc = fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2
    mod = tvm.IRModule.from_expr(primfunc)
    mod = tvm.tir.transform.ConvertForLoopsToSerial()(mod)

    def verify_serial_loops(stmt):
        if isinstance(stmt, tvm.tir.For):
            assert stmt.kind == tvm.tir.ForKind.SERIAL

    for _, primfunc in mod.functions.items():
        stmt_functor.post_order_visit(primfunc.body, verify_serial_loops)


if __name__ == "__main__":
    pytest.main([__file__])
