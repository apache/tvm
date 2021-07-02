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
from tvm import tir, script
from tvm.script import ty
from tvm.tir import stmt_functor

# fmt: off
@tvm.script.tir
def fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2(placeholder_30: ty.handle, placeholder_31: ty.handle, placeholder_32: ty.handle, T_cast_8: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", "tir.noalias": True})
    placeholder_33 = tir.match_buffer(placeholder_30, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_34 = tir.match_buffer(placeholder_31, [1, 1, 192, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_35 = tir.match_buffer(placeholder_32, [1, 1, 1, 16], dtype="int32", elem_offset=0, align=128, offset_factor=1)
    T_cast_9 = tir.match_buffer(T_cast_8, [1, 28, 28, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    # body
    PaddedInput_3 = tir.allocate([1, 28, 28, 192], "int16", "global")
    for i0_i1_fused_3 in tir.parallel(0, 28):
        for i2_3, i3_3 in tir.grid(28, 192):
            tir.store(PaddedInput_3, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3), tir.load("int16", placeholder_33.data, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3)), True)
    for ax0_ax1_fused_ax2_fused_3 in tir.parallel(0, 784):
        for ax3_2 in tir.serial(0, 16):
            Conv2dOutput_3 = tir.allocate([1, 1, 1, 1], "int32", "global")
            tir.store(Conv2dOutput_3, 0, 0, True)
            for rc_3 in tir.serial(0, 192):
                tir.store(Conv2dOutput_3, 0, (tir.load("int32", Conv2dOutput_3, 0) + (tir.cast(tir.load("int16", PaddedInput_3, ((ax0_ax1_fused_ax2_fused_3*192) + rc_3)), "int32")*tir.cast(tir.load("int16", placeholder_34.data, ((rc_3*16) + ax3_2)), "int32"))), True)
            tir.store(T_cast_9.data, ((ax0_ax1_fused_ax2_fused_3*16) + ax3_2), tir.cast(tir.cast(tir.max(tir.min(tir.q_multiply_shift((tir.load("int32", Conv2dOutput_3, 0) + tir.load("int32", placeholder_35.data, ax3_2)), 1764006585, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16"), True)
# fmt: on


def test_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2():
    primfunc = fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2
    primfunc = tvm.tir.usmp.transform.for_loop_serial_converter(primfunc)

    def verify_serial_loops(stmt):
        if isinstance(stmt, tvm.tir.For):
            assert stmt.kind == tvm.tir.ForKind.SERIAL

    stmt_functor.post_order_visit(primfunc.body, verify_serial_loops)


if __name__ == "__main__":
    pytest.main([__file__])
