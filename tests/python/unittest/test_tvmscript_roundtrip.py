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

import sys
import pytest

import tvm
from tvm import tir
from tvm.script import ty


@tvm.script.tir
class Module1:
    def mmult(A: ty.handle, B: ty.handle, C: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "mmult", "tir.noalias": True})
        # buffer definition
        C_global = tir.buffer_decl([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        packedB = tir.buffer_decl([32, 1024, 32], elem_offset=0, align=128, offset_factor=1)
        A_1 = tir.match_buffer(A, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        B_1 = tir.match_buffer(B, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        C_1 = tir.match_buffer(C, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        # body
        tir.realize(packedB[0:32, 0:1024, 0:32], "")
        for x in tir.parallel(0, 32):
            for y in tir.serial(0, 1024):
                for z in tir.vectorized(0, 32):
                    packedB[x, y, z] = B_1[y, ((x * 32) + z)]
        tir.realize(C_1[0:1024, 0:1024], "")
        for x_outer in tir.parallel(0, 32):
            for y_outer in tir.serial(0, 32):
                tir.realize(
                    C_global[
                        (x_outer * 32) : ((x_outer * 32) + 32),
                        (y_outer * 32) : ((y_outer * 32) + 32),
                    ],
                    "global",
                )
                for x_c_init in tir.serial(0, 32):
                    for y_c_init in tir.vectorized(0, 32):
                        C_global[
                            (x_c_init + (x_outer * 32)), (y_c_init + (y_outer * 32))
                        ] = tir.float32(0)
                for k_outer in tir.serial(0, 256):
                    for x_c in tir.serial(0, 32):
                        for k_inner in tir.unroll(0, 4):
                            for y_c in tir.vectorized(0, 32):
                                C_global[(x_c + (x_outer * 32)), (y_c + (y_outer * 32))] = C_global[
                                    (x_c + (x_outer * 32)), (y_c + (y_outer * 32))
                                ] + (
                                    A_1[(x_c + (x_outer * 32)), (k_inner + (k_outer * 4))]
                                    * packedB[
                                        tir.floordiv((y_c + (y_outer * 32)), 32),
                                        (k_inner + (k_outer * 4)),
                                        tir.floormod((y_c + (y_outer * 32)), 32),
                                    ]
                                )
                for x_inner in tir.serial(0, 32):
                    for y_inner in tir.serial(0, 32):
                        C_1[(x_inner + (x_outer * 32)), (y_inner + (y_outer * 32))] = C_global[
                            (x_inner + (x_outer * 32)), (y_inner + (y_outer * 32))
                        ]


def test_opt_gemm_normalize():
    mod = Module1()
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
class Module2:
    def mmult(A: ty.handle, B: ty.handle, C: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "mmult", "tir.noalias": True})
        A_1 = tir.match_buffer(A, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        B_1 = tir.match_buffer(B, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        C_1 = tir.match_buffer(C, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        # body
        packedB = tir.allocate([32768], "float32x32", "global")
        for x in tir.parallel(0, 32):
            for y in tir.serial(0, 1024):
                tir.store(
                    packedB,
                    tir.ramp(((x * 32768) + (y * 32)), 1, 32),
                    tir.load(
                        "float32x32",
                        B_1.data,
                        tir.ramp(((y * 1024) + (x * 32)), 1, 32),
                        tir.broadcast(True, 32),
                    ),
                    tir.broadcast(True, 32),
                )
        for x_outer in tir.parallel(0, 32):
            C_global = tir.allocate([1024], "float32", "global")
            for y_outer in tir.serial(0, 32):
                for x_c_init in tir.serial(0, 32):
                    tir.store(
                        C_global,
                        tir.ramp((x_c_init * 32), 1, 32),
                        tir.broadcast(tir.float32(0), 32),
                        tir.broadcast(True, 32),
                    )
                for k_outer in tir.serial(0, 256):
                    for x_c in tir.serial(0, 32):
                        tir.store(
                            C_global,
                            tir.ramp((x_c * 32), 1, 32),
                            (
                                tir.load(
                                    "float32x32",
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.broadcast(True, 32),
                                )
                                + (
                                    tir.broadcast(
                                        tir.load(
                                            "float32",
                                            A_1.data,
                                            (((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4)),
                                        ),
                                        32,
                                    )
                                    * tir.load(
                                        "float32x32",
                                        packedB,
                                        tir.ramp(((y_outer * 32768) + (k_outer * 128)), 1, 32),
                                        tir.broadcast(True, 32),
                                    )
                                )
                            ),
                            tir.broadcast(True, 32),
                        )
                        tir.store(
                            C_global,
                            tir.ramp((x_c * 32), 1, 32),
                            (
                                tir.load(
                                    "float32x32",
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.broadcast(True, 32),
                                )
                                + (
                                    tir.broadcast(
                                        tir.load(
                                            "float32",
                                            A_1.data,
                                            (
                                                (((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4))
                                                + 1
                                            ),
                                        ),
                                        32,
                                    )
                                    * tir.load(
                                        "float32x32",
                                        packedB,
                                        tir.ramp(
                                            (((y_outer * 32768) + (k_outer * 128)) + 32), 1, 32
                                        ),
                                        tir.broadcast(True, 32),
                                    )
                                )
                            ),
                            tir.broadcast(True, 32),
                        )
                        tir.store(
                            C_global,
                            tir.ramp((x_c * 32), 1, 32),
                            (
                                tir.load(
                                    "float32x32",
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.broadcast(True, 32),
                                )
                                + (
                                    tir.broadcast(
                                        tir.load(
                                            "float32",
                                            A_1.data,
                                            (
                                                (((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4))
                                                + 2
                                            ),
                                        ),
                                        32,
                                    )
                                    * tir.load(
                                        "float32x32",
                                        packedB,
                                        tir.ramp(
                                            (((y_outer * 32768) + (k_outer * 128)) + 64), 1, 32
                                        ),
                                        tir.broadcast(True, 32),
                                    )
                                )
                            ),
                            tir.broadcast(True, 32),
                        )
                        tir.store(
                            C_global,
                            tir.ramp((x_c * 32), 1, 32),
                            (
                                tir.load(
                                    "float32x32",
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.broadcast(True, 32),
                                )
                                + (
                                    tir.broadcast(
                                        tir.load(
                                            "float32",
                                            A_1.data,
                                            (
                                                (((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4))
                                                + 3
                                            ),
                                        ),
                                        32,
                                    )
                                    * tir.load(
                                        "float32x32",
                                        packedB,
                                        tir.ramp(
                                            (((y_outer * 32768) + (k_outer * 128)) + 96), 1, 32
                                        ),
                                        tir.broadcast(True, 32),
                                    )
                                )
                            ),
                            tir.broadcast(True, 32),
                        )
                for x_inner in tir.serial(0, 32):
                    for y_inner in tir.serial(0, 32):
                        C_1.data[
                            ((((x_outer * 32768) + (x_inner * 1024)) + (y_outer * 32)) + y_inner)
                        ] = tir.load("float32", C_global, ((x_inner * 32) + y_inner))


def test_opt_gemm_lower():
    mod = Module2()
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
class Module3:
    def mmult(
        args: ty.handle,
        arg_type_ids: ty.handle,
        num_args: ty.int32,
        out_ret_value: ty.handle,
        out_ret_tcode: ty.handle,
    ) -> ty.int32:
        # function attr dict
        tir.func_attr(
            {
                "tir.noalias": True,
                "global_symbol": "mmult",
                "tir.is_entry_func": True,
                "calling_conv": 1,
            }
        )
        # var definition
        C_global = tir.buffer_var("float32", "global")
        packedB = tir.buffer_var("float32", "global")
        # body
        assert num_args == 3, "mmult: num_args should be 3"
        arg0: ty.handle = tir.tvm_struct_get(args, 0, 12, dtype="handle")
        arg0_code: ty.int32 = tir.load("int32", arg_type_ids, 0)
        arg1: ty.handle = tir.tvm_struct_get(args, 1, 12, dtype="handle")
        arg1_code: ty.int32 = tir.load("int32", arg_type_ids, 1)
        arg2: ty.handle = tir.tvm_struct_get(args, 2, 12, dtype="handle")
        arg2_code: ty.int32 = tir.load("int32", arg_type_ids, 2)
        A: ty.handle = tir.tvm_struct_get(arg0, 0, 1, dtype="handle")
        tir.attr(A, "storage_alignment", 128)
        arg0_shape: ty.handle = tir.tvm_struct_get(arg0, 0, 2, dtype="handle")
        arg0_strides: ty.handle = tir.tvm_struct_get(arg0, 0, 3, dtype="handle")
        dev_id: ty.int32 = tir.tvm_struct_get(arg0, 0, 9, dtype="int32")
        B: ty.handle = tir.tvm_struct_get(arg1, 0, 1, dtype="handle")
        tir.attr(B, "storage_alignment", 128)
        arg1_shape: ty.handle = tir.tvm_struct_get(arg1, 0, 2, dtype="handle")
        arg1_strides: ty.handle = tir.tvm_struct_get(arg1, 0, 3, dtype="handle")
        C: ty.handle = tir.tvm_struct_get(arg2, 0, 1, dtype="handle")
        tir.attr(C, "storage_alignment", 128)
        arg2_shape: ty.handle = tir.tvm_struct_get(arg2, 0, 2, dtype="handle")
        arg2_strides: ty.handle = tir.tvm_struct_get(arg2, 0, 3, dtype="handle")
        assert (((arg0_code == 3) or (arg0_code == 13)) or (arg0_code == 7)) or (
            arg0_code == 4
        ), "mmult: Expect arg[0] to be pointer"
        assert (((arg1_code == 3) or (arg1_code == 13)) or (arg1_code == 7)) or (
            arg1_code == 4
        ), "mmult: Expect arg[1] to be pointer"
        assert (((arg2_code == 3) or (arg2_code == 13)) or (arg2_code == 7)) or (
            arg2_code == 4
        ), "mmult: Expect arg[2] to be pointer"
        assert 2 == tir.tvm_struct_get(
            arg0, 0, 4, dtype="int32"
        ), "arg0.ndim is expected to equal 2"
        assert 2 == tir.tvm_struct_get(
            arg0, 0, 4, dtype="int32"
        ), "arg0.ndim is expected to equal 2"
        assert (
            (tir.tvm_struct_get(arg0, 0, 5, dtype="uint8") == tir.uint8(2))
            and (tir.tvm_struct_get(arg0, 0, 6, dtype="uint8") == tir.uint8(32))
        ) and (
            tir.tvm_struct_get(arg0, 0, 7, dtype="uint16") == tir.uint16(1)
        ), "arg0.dtype is expected to be float32"
        assert 1024 == tir.cast(
            tir.load("int64", arg0_shape, 0), "int32"
        ), "Argument arg0.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            tir.load("int64", arg0_shape, 1), "int32"
        ), "Argument arg0.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg0_strides, dtype="bool")):
            assert (1 == tir.cast(tir.load("int64", arg0_strides, 1), "int32")) and (
                1024 == tir.cast(tir.load("int64", arg0_strides, 0), "int32")
            ), "arg0.strides: expected to be compact array"
            tir.evaluate(0)
        assert tir.uint64(0) == tir.tvm_struct_get(
            arg0, 0, 8, dtype="uint64"
        ), "Argument arg0.byte_offset has an unsatisfied constraint"
        assert 1 == tir.tvm_struct_get(
            arg0, 0, 10, dtype="int32"
        ), "Argument arg0.device_type has an unsatisfied constraint"
        assert 2 == tir.tvm_struct_get(
            arg1, 0, 4, dtype="int32"
        ), "arg1.ndim is expected to equal 2"
        assert 2 == tir.tvm_struct_get(
            arg1, 0, 4, dtype="int32"
        ), "arg1.ndim is expected to equal 2"
        assert (
            (tir.tvm_struct_get(arg1, 0, 5, dtype="uint8") == tir.uint8(2))
            and (tir.tvm_struct_get(arg1, 0, 6, dtype="uint8") == tir.uint8(32))
        ) and (
            tir.tvm_struct_get(arg1, 0, 7, dtype="uint16") == tir.uint16(1)
        ), "arg1.dtype is expected to be float32"
        assert 1024 == tir.cast(
            tir.load("int64", arg1_shape, 0), "int32"
        ), "Argument arg1.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            tir.load("int64", arg1_shape, 1), "int32"
        ), "Argument arg1.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg1_strides, dtype="bool")):
            assert (1 == tir.cast(tir.load("int64", arg1_strides, 1), "int32")) and (
                1024 == tir.cast(tir.load("int64", arg1_strides, 0), "int32")
            ), "arg1.strides: expected to be compact array"
            tir.evaluate(0)
        assert tir.uint64(0) == tir.tvm_struct_get(
            arg1, 0, 8, dtype="uint64"
        ), "Argument arg1.byte_offset has an unsatisfied constraint"
        assert 1 == tir.tvm_struct_get(
            arg1, 0, 10, dtype="int32"
        ), "Argument arg1.device_type has an unsatisfied constraint"
        assert dev_id == tir.tvm_struct_get(
            arg1, 0, 9, dtype="int32"
        ), "Argument arg1.device_id has an unsatisfied constraint"
        assert 2 == tir.tvm_struct_get(
            arg2, 0, 4, dtype="int32"
        ), "arg2.ndim is expected to equal 2"
        assert 2 == tir.tvm_struct_get(
            arg2, 0, 4, dtype="int32"
        ), "arg2.ndim is expected to equal 2"
        assert (
            (tir.tvm_struct_get(arg2, 0, 5, dtype="uint8") == tir.uint8(2))
            and (tir.tvm_struct_get(arg2, 0, 6, dtype="uint8") == tir.uint8(32))
        ) and (
            tir.tvm_struct_get(arg2, 0, 7, dtype="uint16") == tir.uint16(1)
        ), "arg2.dtype is expected to be float32"
        assert 1024 == tir.cast(
            tir.load("int64", arg2_shape, 0), "int32"
        ), "Argument arg2.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            tir.load("int64", arg2_shape, 1), "int32"
        ), "Argument arg2.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg2_strides, dtype="bool")):
            assert (1 == tir.cast(tir.load("int64", arg2_strides, 1), "int32")) and (
                1024 == tir.cast(tir.load("int64", arg2_strides, 0), "int32")
            ), "arg2.strides: expected to be compact array"
            tir.evaluate(0)
        assert tir.uint64(0) == tir.tvm_struct_get(
            arg2, 0, 8, dtype="uint64"
        ), "Argument arg2.byte_offset has an unsatisfied constraint"
        assert 1 == tir.tvm_struct_get(
            arg2, 0, 10, dtype="int32"
        ), "Argument arg2.device_type has an unsatisfied constraint"
        assert dev_id == tir.tvm_struct_get(
            arg2, 0, 9, dtype="int32"
        ), "Argument arg2.device_id has an unsatisfied constraint"
        tir.attr(0, "compute_scope", "mmult_compute_")
        tir.attr(packedB, "storage_scope", "global")
        tir.attr(packedB, "storage_alignment", 128)
        with tir.let(
            packedB,
            tir.TVMBackendAllocWorkspace(1, dev_id, tir.uint64(4194304), 2, 32, dtype="handle"),
        ):
            if tir.isnullptr(packedB, dtype="bool"):
                tir.evaluate(tir.tvm_throw_last_error(dtype="int32"))
            for x in tir.parallel(0, 32):
                for y in tir.serial(0, 1024):
                    tir.store(
                        packedB,
                        tir.ramp(((x * 32768) + (y * 32)), 1, 32),
                        tir.load(
                            "float32x32",
                            B,
                            tir.ramp(((y * 1024) + (x * 32)), 1, 32),
                            tir.broadcast(True, 32),
                        ),
                        tir.broadcast(True, 32),
                    )
            for x_outer in tir.parallel(0, 32):
                tir.attr(C_global, "storage_scope", "global")
                tir.attr(C_global, "storage_alignment", 128)
                with tir.let(
                    C_global,
                    tir.TVMBackendAllocWorkspace(
                        1, dev_id, tir.uint64(4096), 2, 32, dtype="handle"
                    ),
                ):
                    if tir.isnullptr(C_global, dtype="bool"):
                        tir.evaluate(tir.tvm_throw_last_error(dtype="int32"))
                    for y_outer in tir.serial(0, 32):
                        for x_c_init in tir.serial(0, 32):
                            tir.store(
                                C_global,
                                tir.ramp((x_c_init * 32), 1, 32),
                                tir.broadcast(tir.float32(0), 32),
                                tir.broadcast(True, 32),
                            )
                        for k_outer in tir.serial(0, 256):
                            for x_c in tir.serial(0, 32):
                                tir.store(
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.call_llvm_pure_intrin(
                                        tir.uint32(97),
                                        tir.uint32(3),
                                        tir.broadcast(
                                            tir.load(
                                                "float32",
                                                A,
                                                (
                                                    ((x_outer * 32768) + (x_c * 1024))
                                                    + (k_outer * 4)
                                                ),
                                            ),
                                            32,
                                        ),
                                        tir.load(
                                            "float32x32",
                                            packedB,
                                            tir.ramp(((y_outer * 32768) + (k_outer * 128)), 1, 32),
                                            tir.broadcast(True, 32),
                                        ),
                                        tir.load(
                                            "float32x32",
                                            C_global,
                                            tir.ramp((x_c * 32), 1, 32),
                                            tir.broadcast(True, 32),
                                        ),
                                        dtype="float32x32",
                                    ),
                                    tir.broadcast(True, 32),
                                )
                                tir.store(
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.call_llvm_pure_intrin(
                                        tir.uint32(97),
                                        tir.uint32(3),
                                        tir.broadcast(
                                            tir.load(
                                                "float32",
                                                A,
                                                (
                                                    (
                                                        ((x_outer * 32768) + (x_c * 1024))
                                                        + (k_outer * 4)
                                                    )
                                                    + 1
                                                ),
                                            ),
                                            32,
                                        ),
                                        tir.load(
                                            "float32x32",
                                            packedB,
                                            tir.ramp(
                                                (((y_outer * 32768) + (k_outer * 128)) + 32), 1, 32
                                            ),
                                            tir.broadcast(True, 32),
                                        ),
                                        tir.load(
                                            "float32x32",
                                            C_global,
                                            tir.ramp((x_c * 32), 1, 32),
                                            tir.broadcast(True, 32),
                                        ),
                                        dtype="float32x32",
                                    ),
                                    tir.broadcast(True, 32),
                                )
                                tir.store(
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.call_llvm_pure_intrin(
                                        tir.uint32(97),
                                        tir.uint32(3),
                                        tir.broadcast(
                                            tir.load(
                                                "float32",
                                                A,
                                                (
                                                    (
                                                        ((x_outer * 32768) + (x_c * 1024))
                                                        + (k_outer * 4)
                                                    )
                                                    + 2
                                                ),
                                            ),
                                            32,
                                        ),
                                        tir.load(
                                            "float32x32",
                                            packedB,
                                            tir.ramp(
                                                (((y_outer * 32768) + (k_outer * 128)) + 64), 1, 32
                                            ),
                                            tir.broadcast(True, 32),
                                        ),
                                        tir.load(
                                            "float32x32",
                                            C_global,
                                            tir.ramp((x_c * 32), 1, 32),
                                            tir.broadcast(True, 32),
                                        ),
                                        dtype="float32x32",
                                    ),
                                    tir.broadcast(True, 32),
                                )
                                tir.store(
                                    C_global,
                                    tir.ramp((x_c * 32), 1, 32),
                                    tir.call_llvm_pure_intrin(
                                        tir.uint32(97),
                                        tir.uint32(3),
                                        tir.broadcast(
                                            tir.load(
                                                "float32",
                                                A,
                                                (
                                                    (
                                                        ((x_outer * 32768) + (x_c * 1024))
                                                        + (k_outer * 4)
                                                    )
                                                    + 3
                                                ),
                                            ),
                                            32,
                                        ),
                                        tir.load(
                                            "float32x32",
                                            packedB,
                                            tir.ramp(
                                                (((y_outer * 32768) + (k_outer * 128)) + 96), 1, 32
                                            ),
                                            tir.broadcast(True, 32),
                                        ),
                                        tir.load(
                                            "float32x32",
                                            C_global,
                                            tir.ramp((x_c * 32), 1, 32),
                                            tir.broadcast(True, 32),
                                        ),
                                        dtype="float32x32",
                                    ),
                                    tir.broadcast(True, 32),
                                )
                        for x_inner in tir.serial(0, 32):
                            for y_inner in tir.serial(0, 32):
                                C[
                                    (
                                        (((x_outer * 32768) + (x_inner * 1024)) + (y_outer * 32))
                                        + y_inner
                                    )
                                ] = tir.load("float32", C_global, ((x_inner * 32) + y_inner))
                if tir.TVMBackendFreeWorkspace(1, dev_id, C_global, dtype="int32") != 0:
                    tir.evaluate(tir.tvm_throw_last_error(dtype="int32"))
        if tir.TVMBackendFreeWorkspace(1, dev_id, packedB, dtype="int32") != 0:
            tir.evaluate(tir.tvm_throw_last_error(dtype="int32"))


def test_opt_gemm_mod_host():
    mod = Module3()
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
def opt_conv_tensorcore_normalize(A: ty.handle, W: ty.handle, Conv: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # var definition
    bx = tir.env_thread("blockIdx.x")
    by = tir.env_thread("blockIdx.y")
    bz = tir.env_thread("blockIdx.z")
    tx = tir.env_thread("threadIdx.x")
    ty = tir.env_thread("threadIdx.y")
    tz = tir.env_thread("threadIdx.z")
    # buffer definition
    Apad_shared = tir.buffer_decl(
        [16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Apad_shared_wmma_matrix_a = tir.buffer_decl(
        [16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    BA = tir.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256
    )
    BB = tir.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256
    )
    BC = tir.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
    Conv_wmma_accumulator = tir.buffer_decl(
        [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    W_shared = tir.buffer_decl(
        [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_shared_wmma_matrix_b = tir.buffer_decl(
        [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    buffer = tir.buffer_decl([16, 16], dtype="float16", scope="shared", align=32, offset_factor=256)
    buffer_1 = tir.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256
    )
    buffer_2 = tir.buffer_decl(
        [16, 16], dtype="float16", scope="shared", align=32, offset_factor=256
    )
    buffer_3 = tir.buffer_decl(
        [16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256
    )
    buffer_4 = tir.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
    buffer_5 = tir.buffer_decl([16, 16], align=32, offset_factor=256)
    A_1 = tir.match_buffer(
        A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_1 = tir.match_buffer(
        W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Conv_1 = tir.match_buffer(
        Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    # body
    tir.realize(Conv_1[0:16, 0:14, 0:14, 0:32, 0:16, 0:16], "")
    tir.launch_thread(bz, 196)
    tir.launch_thread(bx, 2)
    tir.launch_thread(by, 4)
    tir.launch_thread(ty, 4)
    tir.launch_thread(tz, 2)
    tir.realize(
        Conv_wmma_accumulator[
            ((bx * 8) + (ty * 2)) : (((bx * 8) + (ty * 2)) + 2),
            tir.floordiv(bz, 14) : (tir.floordiv(bz, 14) + 1),
            tir.floormod(bz, 14) : (tir.floormod(bz, 14) + 1),
            ((by * 8) + (tz * 4)) : (((by * 8) + (tz * 4)) + 4),
            0:16,
            0:16,
        ],
        "wmma.accumulator",
    )
    for n_c_init in tir.serial(0, 2):
        for o_c_init in tir.serial(0, 4):
            tir.attr(
                [BC, Conv_wmma_accumulator],
                "buffer_bind_scope",
                tir.tvm_tuple(
                    (n_c_init + ((bx * 8) + (ty * 2))),
                    1,
                    tir.floordiv(bz, 14),
                    1,
                    tir.floormod(bz, 14),
                    1,
                    (o_c_init + ((by * 8) + (tz * 4))),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            tir.evaluate(
                tir.tvm_fill_fragment(
                    BC.data,
                    16,
                    16,
                    16,
                    tir.floordiv(BC.elem_offset, 256),
                    tir.float32(0),
                    dtype="handle",
                )
            )
    for ic_outer in tir.serial(0, 8):
        for kh in tir.serial(0, 3):
            tir.realize(
                Apad_shared[
                    (bx * 8) : ((bx * 8) + 8),
                    (tir.floordiv(bz, 14) + kh) : ((tir.floordiv(bz, 14) + kh) + 1),
                    tir.floormod(bz, 14) : (tir.floormod(bz, 14) + 3),
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    0:16,
                    0:16,
                ],
                "shared",
            )
            for ax2 in tir.serial(0, 3):
                for ax3 in tir.serial(0, 2):
                    for ax4_ax5_fused_outer in tir.serial(0, 8):
                        tir.launch_thread(tx, 32)
                        Apad_shared[
                            ((tz + (ty * 2)) + (bx * 8)),
                            (tir.floordiv(bz, 14) + kh),
                            (ax2 + tir.floormod(bz, 14)),
                            (ax3 + (ic_outer * 2)),
                            tir.floordiv((tx + (ax4_ax5_fused_outer * 32)), 16),
                            tir.floormod((tx + (ax4_ax5_fused_outer * 32)), 16),
                        ] = tir.if_then_else(
                            (
                                (
                                    (
                                        ((tir.floordiv(bz, 14) + kh) >= 1)
                                        and (((tir.floordiv(bz, 14) + kh) - 1) < 14)
                                    )
                                    and ((ax2 + tir.floormod(bz, 14)) >= 1)
                                )
                                and (((ax2 + tir.floormod(bz, 14)) - 1) < 14)
                            ),
                            A_1[
                                ((tz + (ty * 2)) + (bx * 8)),
                                ((tir.floordiv(bz, 14) + kh) - 1),
                                ((ax2 + tir.floormod(bz, 14)) - 1),
                                (ax3 + (ic_outer * 2)),
                                tir.floordiv((tx + (ax4_ax5_fused_outer * 32)), 16),
                                tir.floormod((tx + (ax4_ax5_fused_outer * 32)), 16),
                            ],
                            tir.float16(0),
                            dtype="float16",
                        )
            tir.realize(
                W_shared[
                    kh : (kh + 1),
                    0:3,
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    (by * 8) : ((by * 8) + 8),
                    0:16,
                    0:16,
                ],
                "shared",
            )
            for ax1 in tir.serial(0, 3):
                for ax2_1 in tir.serial(0, 2):
                    tir.launch_thread(tx, 32)
                    for ax4_ax5_fused_inner in tir.vectorized(0, 8):
                        W_shared[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((tz + (ty * 2)) + (by * 8)),
                            tir.floordiv((ax4_ax5_fused_inner + (tx * 8)), 16),
                            tir.floormod((ax4_ax5_fused_inner + (tx * 8)), 16),
                        ] = W_1[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((tz + (ty * 2)) + (by * 8)),
                            tir.floordiv((ax4_ax5_fused_inner + (tx * 8)), 16),
                            tir.floormod((ax4_ax5_fused_inner + (tx * 8)), 16),
                        ]
            for ic_inner in tir.serial(0, 2):
                for kw in tir.serial(0, 3):
                    tir.realize(
                        Apad_shared_wmma_matrix_a[
                            ((bx * 8) + (ty * 2)) : (((bx * 8) + (ty * 2)) + 2),
                            (tir.floordiv(bz, 14) + kh) : ((tir.floordiv(bz, 14) + kh) + 1),
                            (kw + tir.floormod(bz, 14)) : ((kw + tir.floormod(bz, 14)) + 1),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            0:16,
                            0:16,
                        ],
                        "wmma.matrix_a",
                    )
                    for ax0 in tir.serial(0, 2):
                        tir.attr(
                            [buffer, Apad_shared],
                            "buffer_bind_scope",
                            tir.tvm_tuple(
                                (ax0 + ((bx * 8) + (ty * 2))),
                                1,
                                (tir.floordiv(bz, 14) + kh),
                                1,
                                (kw + tir.floormod(bz, 14)),
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        tir.attr(
                            [buffer_1, Apad_shared_wmma_matrix_a],
                            "buffer_bind_scope",
                            tir.tvm_tuple(
                                (ax0 + ((bx * 8) + (ty * 2))),
                                1,
                                (tir.floordiv(bz, 14) + kh),
                                1,
                                (kw + tir.floormod(bz, 14)),
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        tir.evaluate(
                            tir.tvm_load_matrix_sync(
                                buffer_1.data,
                                16,
                                16,
                                16,
                                tir.floordiv(buffer_1.elem_offset, 256),
                                tir.tvm_access_ptr(
                                    tir.type_annotation(dtype="float16"),
                                    buffer.data,
                                    buffer.elem_offset,
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                    tir.realize(
                        W_shared_wmma_matrix_b[
                            kh : (kh + 1),
                            kw : (kw + 1),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            ((by * 8) + (tz * 4)) : (((by * 8) + (tz * 4)) + 4),
                            0:16,
                            0:16,
                        ],
                        "wmma.matrix_b",
                    )
                    for ax3_1 in tir.serial(0, 4):
                        tir.attr(
                            [buffer_2, W_shared],
                            "buffer_bind_scope",
                            tir.tvm_tuple(
                                kh,
                                1,
                                kw,
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                (ax3_1 + ((by * 8) + (tz * 4))),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        tir.attr(
                            [buffer_3, W_shared_wmma_matrix_b],
                            "buffer_bind_scope",
                            tir.tvm_tuple(
                                kh,
                                1,
                                kw,
                                1,
                                ((ic_outer * 2) + ic_inner),
                                1,
                                (ax3_1 + ((by * 8) + (tz * 4))),
                                1,
                                0,
                                16,
                                0,
                                16,
                                dtype="handle",
                            ),
                        )
                        tir.evaluate(
                            tir.tvm_load_matrix_sync(
                                buffer_3.data,
                                16,
                                16,
                                16,
                                tir.floordiv(buffer_3.elem_offset, 256),
                                tir.tvm_access_ptr(
                                    tir.type_annotation(dtype="float16"),
                                    buffer_2.data,
                                    buffer_2.elem_offset,
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                    for n_c in tir.serial(0, 2):
                        for o_c in tir.serial(0, 4):
                            tir.attr(
                                [BA, Apad_shared_wmma_matrix_a],
                                "buffer_bind_scope",
                                tir.tvm_tuple(
                                    (n_c + ((bx * 8) + (ty * 2))),
                                    1,
                                    (tir.floordiv(bz, 14) + kh),
                                    1,
                                    (tir.floormod(bz, 14) + kw),
                                    1,
                                    ((ic_outer * 2) + ic_inner),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            tir.attr(
                                [BB, W_shared_wmma_matrix_b],
                                "buffer_bind_scope",
                                tir.tvm_tuple(
                                    kh,
                                    1,
                                    kw,
                                    1,
                                    ((ic_outer * 2) + ic_inner),
                                    1,
                                    (o_c + ((by * 8) + (tz * 4))),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            tir.attr(
                                [BC, Conv_wmma_accumulator],
                                "buffer_bind_scope",
                                tir.tvm_tuple(
                                    (n_c + ((bx * 8) + (ty * 2))),
                                    1,
                                    tir.floordiv(bz, 14),
                                    1,
                                    tir.floormod(bz, 14),
                                    1,
                                    (o_c + ((by * 8) + (tz * 4))),
                                    1,
                                    0,
                                    16,
                                    0,
                                    16,
                                    dtype="handle",
                                ),
                            )
                            tir.evaluate(
                                tir.tvm_mma_sync(
                                    BC.data,
                                    tir.floordiv(BC.elem_offset, 256),
                                    BA.data,
                                    tir.floordiv(BA.elem_offset, 256),
                                    BB.data,
                                    tir.floordiv(BB.elem_offset, 256),
                                    BC.data,
                                    tir.floordiv(BC.elem_offset, 256),
                                    dtype="handle",
                                )
                            )
    for n_inner in tir.serial(0, 2):
        for o_inner in tir.serial(0, 4):
            tir.attr(
                [buffer_4, Conv_wmma_accumulator],
                "buffer_bind_scope",
                tir.tvm_tuple(
                    ((((bx * 4) + ty) * 2) + n_inner),
                    1,
                    tir.floordiv(bz, 14),
                    1,
                    tir.floormod(bz, 14),
                    1,
                    ((((by * 2) + tz) * 4) + o_inner),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            tir.attr(
                [buffer_5, Conv_1],
                "buffer_bind_scope",
                tir.tvm_tuple(
                    ((((bx * 4) + ty) * 2) + n_inner),
                    1,
                    tir.floordiv(bz, 14),
                    1,
                    tir.floormod(bz, 14),
                    1,
                    ((((by * 2) + tz) * 4) + o_inner),
                    1,
                    0,
                    16,
                    0,
                    16,
                    dtype="handle",
                ),
            )
            tir.evaluate(
                tir.tvm_store_matrix_sync(
                    buffer_4.data,
                    16,
                    16,
                    16,
                    tir.floordiv(buffer_4.elem_offset, 256),
                    tir.tvm_access_ptr(
                        tir.type_annotation(dtype="float32"),
                        buffer_5.data,
                        buffer_5.elem_offset,
                        256,
                        2,
                        dtype="handle",
                    ),
                    16,
                    "row_major",
                    dtype="handle",
                )
            )


def test_opt_conv_tensorcore_normalize():
    mod = opt_conv_tensorcore_normalize
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
def opt_conv_tensorcore_lower(A: ty.handle, W: ty.handle, Conv: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # body
    A_1 = tir.match_buffer(
        A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_1 = tir.match_buffer(
        W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Conv_1 = tir.match_buffer(
        Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    bx = tir.env_thread("blockIdx.x")
    by = tir.env_thread("blockIdx.y")
    bz = tir.env_thread("blockIdx.z")
    tx = tir.env_thread("threadIdx.x")
    ty = tir.env_thread("threadIdx.y")
    tz = tir.env_thread("threadIdx.z")
    tir.launch_thread(bz, 196)
    Conv_wmma_accumulator = tir.allocate([2048], "float32", "wmma.accumulator")
    Apad_shared = tir.allocate([12288], "float16", "shared")
    W_shared = tir.allocate([12288], "float16", "shared")
    Apad_shared_wmma_matrix_a = tir.allocate([512], "float16", "wmma.matrix_a")
    W_shared_wmma_matrix_b = tir.allocate([1024], "float16", "wmma.matrix_b")
    tir.launch_thread(bx, 2)
    tir.launch_thread(by, 4)
    tir.launch_thread(ty, 4)
    tir.launch_thread(tz, 2)
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 0, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 1, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 2, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 3, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 4, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 5, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 6, tir.float32(0), dtype="handle")
    )
    tir.evaluate(
        tir.tvm_fill_fragment(Conv_wmma_accumulator, 16, 16, 16, 7, tir.float32(0), dtype="handle")
    )
    for ic_outer in tir.serial(0, 8):
        for kh in tir.serial(0, 3):
            for ax2 in tir.serial(0, 3):
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        ((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61440
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 32)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61408
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 64)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61376
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 96)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61344
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 128)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61312
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 160)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61280
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 192)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61248
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 224)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61216
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 256)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61184
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 288)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61152
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 320)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61120
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 352)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61088
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 384)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61056
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 416)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 61024
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.launch_thread(tx, 32):
                    Apad_shared[
                        (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 448)
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(bz, 14) + kh))
                                    and ((tir.floordiv(bz, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(bz, 14)))
                            )
                            and ((ax2 + tir.floormod(bz, 14)) < 15)
                        ),
                        tir.load(
                            "float16",
                            A_1.data,
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((bx * 6422528) + (ty * 1605632))
                                                        + (tz * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (bz * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + tx
                                )
                                - 60992
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                tir.launch_thread(tx, 32)
                Apad_shared[
                    (((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 480)
                ] = tir.if_then_else(
                    (
                        (
                            (
                                (1 <= (tir.floordiv(bz, 14) + kh))
                                and ((tir.floordiv(bz, 14) + kh) < 15)
                            )
                            and (1 <= (ax2 + tir.floormod(bz, 14)))
                        )
                        and ((ax2 + tir.floormod(bz, 14)) < 15)
                    ),
                    tir.load(
                        "float16",
                        A_1.data,
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (((bx * 6422528) + (ty * 1605632)) + (tz * 802816))
                                                + (kh * 57344)
                                            )
                                            + (bz * 4096)
                                        )
                                        + (ax2 * 4096)
                                    )
                                    + (ic_outer * 512)
                                )
                                + tx
                            )
                            - 60960
                        ),
                    ),
                    tir.float16(0),
                    dtype="float16",
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp((((ty * 512) + (tz * 256)) + (tx * 8)), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                        + (ty * 512)
                                    )
                                    + (tz * 256)
                                )
                                + (tx * 8)
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 2048), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 8192
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 4096), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 131072
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 6144), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 139264
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 8192), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 262144
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.launch_thread(tx, 32):
                tir.store(
                    W_shared,
                    tir.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 10240), 1, 8),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 270336
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            for ic_inner in tir.serial(0, 2):
                for kw in tir.serial(0, 3):
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            Apad_shared_wmma_matrix_a,
                            16,
                            16,
                            16,
                            0,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                Apad_shared,
                                (((ty * 3072) + (kw * 512)) + (ic_inner * 256)),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            Apad_shared_wmma_matrix_a,
                            16,
                            16,
                            16,
                            1,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                Apad_shared,
                                ((((ty * 3072) + (kw * 512)) + (ic_inner * 256)) + 1536),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            W_shared_wmma_matrix_b,
                            16,
                            16,
                            16,
                            0,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                W_shared,
                                (((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            W_shared_wmma_matrix_b,
                            16,
                            16,
                            16,
                            1,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                W_shared,
                                ((((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) + 256),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            W_shared_wmma_matrix_b,
                            16,
                            16,
                            16,
                            2,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                W_shared,
                                ((((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) + 512),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_load_matrix_sync(
                            W_shared_wmma_matrix_b,
                            16,
                            16,
                            16,
                            3,
                            tir.tvm_access_ptr(
                                tir.type_annotation(dtype="float16"),
                                W_shared,
                                ((((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) + 768),
                                256,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            0,
                            Apad_shared_wmma_matrix_a,
                            0,
                            W_shared_wmma_matrix_b,
                            0,
                            Conv_wmma_accumulator,
                            0,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            1,
                            Apad_shared_wmma_matrix_a,
                            0,
                            W_shared_wmma_matrix_b,
                            1,
                            Conv_wmma_accumulator,
                            1,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            2,
                            Apad_shared_wmma_matrix_a,
                            0,
                            W_shared_wmma_matrix_b,
                            2,
                            Conv_wmma_accumulator,
                            2,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            3,
                            Apad_shared_wmma_matrix_a,
                            0,
                            W_shared_wmma_matrix_b,
                            3,
                            Conv_wmma_accumulator,
                            3,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            4,
                            Apad_shared_wmma_matrix_a,
                            1,
                            W_shared_wmma_matrix_b,
                            0,
                            Conv_wmma_accumulator,
                            4,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            5,
                            Apad_shared_wmma_matrix_a,
                            1,
                            W_shared_wmma_matrix_b,
                            1,
                            Conv_wmma_accumulator,
                            5,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            6,
                            Apad_shared_wmma_matrix_a,
                            1,
                            W_shared_wmma_matrix_b,
                            2,
                            Conv_wmma_accumulator,
                            6,
                            dtype="handle",
                        )
                    )
                    tir.evaluate(
                        tir.tvm_mma_sync(
                            Conv_wmma_accumulator,
                            7,
                            Apad_shared_wmma_matrix_a,
                            1,
                            W_shared_wmma_matrix_b,
                            3,
                            Conv_wmma_accumulator,
                            7,
                            dtype="handle",
                        )
                    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            0,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048)) + (tz * 1024)),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            1,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 256
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            2,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 512
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            3,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 768
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            4,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 1605632
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            5,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 1605888
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            6,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 1606144
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )
    tir.evaluate(
        tir.tvm_store_matrix_sync(
            Conv_wmma_accumulator,
            16,
            16,
            16,
            7,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    + 1606400
                ),
                256,
                2,
                dtype="handle",
            ),
            16,
            "row_major",
            dtype="handle",
        )
    )


def test_opt_conv_tensorcore_lower():
    mod = opt_conv_tensorcore_lower
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
def opt_conv_tensorcore_mod_host(
    args: ty.handle,
    arg_type_ids: ty.handle,
    num_args: ty.int32,
    out_ret_value: ty.handle,
    out_ret_tcode: ty.handle,
    resource_handle: ty.handle,
) -> ty.int32:
    # function attr dict
    tir.func_attr(
        {
            "tir.noalias": True,
            "global_symbol": "default_function",
            "tir.is_entry_func": True,
            "calling_conv": 1,
        }
    )
    # body
    stack_tcode: ty.handle = tir.tvm_stack_alloca("arg_tcode", 10, dtype="handle")
    stack_value: ty.handle = tir.tvm_stack_alloca("arg_value", 10, dtype="handle")
    assert num_args == 3, "default_function: num_args should be 3"
    arg0: ty.handle = tir.tvm_struct_get(args, 0, 12, dtype="handle")
    arg0_code: ty.int32 = tir.load("int32", arg_type_ids, 0)
    arg1: ty.handle = tir.tvm_struct_get(args, 1, 12, dtype="handle")
    arg1_code: ty.int32 = tir.load("int32", arg_type_ids, 1)
    arg2: ty.handle = tir.tvm_struct_get(args, 2, 12, dtype="handle")
    arg2_code: ty.int32 = tir.load("int32", arg_type_ids, 2)
    A: ty.handle = tir.tvm_struct_get(arg0, 0, 1, dtype="handle")
    tir.attr(A, "storage_alignment", 128)
    arg0_shape: ty.handle = tir.tvm_struct_get(arg0, 0, 2, dtype="handle")
    arg0_strides: ty.handle = tir.tvm_struct_get(arg0, 0, 3, dtype="handle")
    dev_id: ty.int32 = tir.tvm_struct_get(arg0, 0, 9, dtype="int32")
    W: ty.handle = tir.tvm_struct_get(arg1, 0, 1, dtype="handle")
    tir.attr(W, "storage_alignment", 128)
    arg1_shape: ty.handle = tir.tvm_struct_get(arg1, 0, 2, dtype="handle")
    arg1_strides: ty.handle = tir.tvm_struct_get(arg1, 0, 3, dtype="handle")
    Conv: ty.handle = tir.tvm_struct_get(arg2, 0, 1, dtype="handle")
    tir.attr(Conv, "storage_alignment", 128)
    arg2_shape: ty.handle = tir.tvm_struct_get(arg2, 0, 2, dtype="handle")
    arg2_strides: ty.handle = tir.tvm_struct_get(arg2, 0, 3, dtype="handle")
    assert (((arg0_code == 3) or (arg0_code == 13)) or (arg0_code == 7)) or (
        arg0_code == 4
    ), "default_function: Expect arg[0] to be pointer"
    assert (((arg1_code == 3) or (arg1_code == 13)) or (arg1_code == 7)) or (
        arg1_code == 4
    ), "default_function: Expect arg[1] to be pointer"
    assert (((arg2_code == 3) or (arg2_code == 13)) or (arg2_code == 7)) or (
        arg2_code == 4
    ), "default_function: Expect arg[2] to be pointer"
    assert 6 == tir.tvm_struct_get(arg0, 0, 4, dtype="int32"), "arg0.ndim is expected to equal 6"
    assert 6 == tir.tvm_struct_get(arg0, 0, 4, dtype="int32"), "arg0.ndim is expected to equal 6"
    assert (
        (tir.tvm_struct_get(arg0, 0, 5, dtype="uint8") == tir.uint8(2))
        and (tir.tvm_struct_get(arg0, 0, 6, dtype="uint8") == tir.uint8(16))
    ) and (
        tir.tvm_struct_get(arg0, 0, 7, dtype="uint16") == tir.uint16(1)
    ), "arg0.dtype is expected to be float16"
    assert 16 == tir.cast(
        tir.load("int64", arg0_shape, 0), "int32"
    ), "Argument arg0.shape[0] has an unsatisfied constraint"
    assert 14 == tir.cast(
        tir.load("int64", arg0_shape, 1), "int32"
    ), "Argument arg0.shape[1] has an unsatisfied constraint"
    assert 14 == tir.cast(
        tir.load("int64", arg0_shape, 2), "int32"
    ), "Argument arg0.shape[2] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg0_shape, 3), "int32"
    ), "Argument arg0.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg0_shape, 4), "int32"
    ), "Argument arg0.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg0_shape, 5), "int32"
    ), "Argument arg0.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg0_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast(tir.load("int64", arg0_strides, 5), "int32"))
                        and (16 == tir.cast(tir.load("int64", arg0_strides, 4), "int32"))
                    )
                    and (256 == tir.cast(tir.load("int64", arg0_strides, 3), "int32"))
                )
                and (4096 == tir.cast(tir.load("int64", arg0_strides, 2), "int32"))
            )
            and (57344 == tir.cast(tir.load("int64", arg0_strides, 1), "int32"))
        ) and (
            802816 == tir.cast(tir.load("int64", arg0_strides, 0), "int32")
        ), "arg0.strides: expected to be compact array"
        tir.evaluate(0)
    assert tir.uint64(0) == tir.tvm_struct_get(
        arg0, 0, 8, dtype="uint64"
    ), "Argument arg0.byte_offset has an unsatisfied constraint"
    assert 2 == tir.tvm_struct_get(
        arg0, 0, 10, dtype="int32"
    ), "Argument arg0.device_type has an unsatisfied constraint"
    assert 6 == tir.tvm_struct_get(arg1, 0, 4, dtype="int32"), "arg1.ndim is expected to equal 6"
    assert 6 == tir.tvm_struct_get(arg1, 0, 4, dtype="int32"), "arg1.ndim is expected to equal 6"
    assert (
        (tir.tvm_struct_get(arg1, 0, 5, dtype="uint8") == tir.uint8(2))
        and (tir.tvm_struct_get(arg1, 0, 6, dtype="uint8") == tir.uint8(16))
    ) and (
        tir.tvm_struct_get(arg1, 0, 7, dtype="uint16") == tir.uint16(1)
    ), "arg1.dtype is expected to be float16"
    assert 3 == tir.cast(
        tir.load("int64", arg1_shape, 0), "int32"
    ), "Argument arg1.shape[0] has an unsatisfied constraint"
    assert 3 == tir.cast(
        tir.load("int64", arg1_shape, 1), "int32"
    ), "Argument arg1.shape[1] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg1_shape, 2), "int32"
    ), "Argument arg1.shape[2] has an unsatisfied constraint"
    assert 32 == tir.cast(
        tir.load("int64", arg1_shape, 3), "int32"
    ), "Argument arg1.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg1_shape, 4), "int32"
    ), "Argument arg1.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg1_shape, 5), "int32"
    ), "Argument arg1.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg1_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast(tir.load("int64", arg1_strides, 5), "int32"))
                        and (16 == tir.cast(tir.load("int64", arg1_strides, 4), "int32"))
                    )
                    and (256 == tir.cast(tir.load("int64", arg1_strides, 3), "int32"))
                )
                and (8192 == tir.cast(tir.load("int64", arg1_strides, 2), "int32"))
            )
            and (131072 == tir.cast(tir.load("int64", arg1_strides, 1), "int32"))
        ) and (
            393216 == tir.cast(tir.load("int64", arg1_strides, 0), "int32")
        ), "arg1.strides: expected to be compact array"
        tir.evaluate(0)
    assert tir.uint64(0) == tir.tvm_struct_get(
        arg1, 0, 8, dtype="uint64"
    ), "Argument arg1.byte_offset has an unsatisfied constraint"
    assert 2 == tir.tvm_struct_get(
        arg1, 0, 10, dtype="int32"
    ), "Argument arg1.device_type has an unsatisfied constraint"
    assert dev_id == tir.tvm_struct_get(
        arg1, 0, 9, dtype="int32"
    ), "Argument arg1.device_id has an unsatisfied constraint"
    assert 6 == tir.tvm_struct_get(arg2, 0, 4, dtype="int32"), "arg2.ndim is expected to equal 6"
    assert 6 == tir.tvm_struct_get(arg2, 0, 4, dtype="int32"), "arg2.ndim is expected to equal 6"
    assert (
        (tir.tvm_struct_get(arg2, 0, 5, dtype="uint8") == tir.uint8(2))
        and (tir.tvm_struct_get(arg2, 0, 6, dtype="uint8") == tir.uint8(32))
    ) and (
        tir.tvm_struct_get(arg2, 0, 7, dtype="uint16") == tir.uint16(1)
    ), "arg2.dtype is expected to be float32"
    assert 16 == tir.cast(
        tir.load("int64", arg2_shape, 0), "int32"
    ), "Argument arg2.shape[0] has an unsatisfied constraint"
    assert 14 == tir.cast(
        tir.load("int64", arg2_shape, 1), "int32"
    ), "Argument arg2.shape[1] has an unsatisfied constraint"
    assert 14 == tir.cast(
        tir.load("int64", arg2_shape, 2), "int32"
    ), "Argument arg2.shape[2] has an unsatisfied constraint"
    assert 32 == tir.cast(
        tir.load("int64", arg2_shape, 3), "int32"
    ), "Argument arg2.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg2_shape, 4), "int32"
    ), "Argument arg2.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        tir.load("int64", arg2_shape, 5), "int32"
    ), "Argument arg2.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg2_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast(tir.load("int64", arg2_strides, 5), "int32"))
                        and (16 == tir.cast(tir.load("int64", arg2_strides, 4), "int32"))
                    )
                    and (256 == tir.cast(tir.load("int64", arg2_strides, 3), "int32"))
                )
                and (8192 == tir.cast(tir.load("int64", arg2_strides, 2), "int32"))
            )
            and (114688 == tir.cast(tir.load("int64", arg2_strides, 1), "int32"))
        ) and (
            1605632 == tir.cast(tir.load("int64", arg2_strides, 0), "int32")
        ), "arg2.strides: expected to be compact array"
        tir.evaluate(0)
    assert tir.uint64(0) == tir.tvm_struct_get(
        arg2, 0, 8, dtype="uint64"
    ), "Argument arg2.byte_offset has an unsatisfied constraint"
    assert 2 == tir.tvm_struct_get(
        arg2, 0, 10, dtype="int32"
    ), "Argument arg2.device_type has an unsatisfied constraint"
    assert dev_id == tir.tvm_struct_get(
        arg2, 0, 9, dtype="int32"
    ), "Argument arg2.device_id has an unsatisfied constraint"
    tir.evaluate(tir.tvm_struct_set(stack_value, 0, 12, tir.cast(2, "int64"), dtype="int32"))
    stack_tcode[0] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 1, 12, tir.cast(dev_id, "int64"), dtype="int32"))
    stack_tcode[1] = 0
    tir.evaluate(
        tir.tvm_call_packed_lowered(
            "__tvm_set_device", stack_value, stack_tcode, 0, 2, dtype="int32"
        )
    )
    tir.attr(0, "compute_scope", "default_function_compute_")
    tir.evaluate(tir.tvm_struct_set(stack_value, 0, 12, A, dtype="int32"))
    stack_tcode[0] = 3
    tir.evaluate(tir.tvm_struct_set(stack_value, 1, 12, W, dtype="int32"))
    stack_tcode[1] = 3
    tir.evaluate(tir.tvm_struct_set(stack_value, 2, 12, Conv, dtype="int32"))
    stack_tcode[2] = 3
    tir.evaluate(tir.tvm_struct_set(stack_value, 3, 12, tir.cast(196, "int64"), dtype="int32"))
    stack_tcode[3] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 4, 12, tir.cast(2, "int64"), dtype="int32"))
    stack_tcode[4] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 5, 12, tir.cast(4, "int64"), dtype="int32"))
    stack_tcode[5] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 6, 12, tir.cast(4, "int64"), dtype="int32"))
    stack_tcode[6] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 7, 12, tir.cast(2, "int64"), dtype="int32"))
    stack_tcode[7] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 8, 12, tir.cast(32, "int64"), dtype="int32"))
    stack_tcode[8] = 0
    tir.evaluate(
        tir.tvm_call_packed_lowered(
            "default_function_kernel0", stack_value, stack_tcode, 0, 9, dtype="int32"
        )
    )


def test_opt_conv_tensorcore_mod_host():
    mod = opt_conv_tensorcore_mod_host
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.script.tir
def vthread_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")

    i0 = tir.env_thread("blockIdx.x")
    i1 = tir.env_thread("threadIdx.x")
    i2 = tir.env_thread("vthread")

    tir.launch_thread(i0, 4)
    tir.launch_thread(i1, 2)
    tir.launch_thread(i2, 2)
    B = tir.allocate([16], "float32", "local")
    for j in range(16):
        B[j] = tir.load("float32", A.data, i0 * 64 + i1 * 32 + i2 * 16 + j) + tir.float32(1)
    for j in range(16):
        C.data[i0 * 64 + i1 * 32 + i2 * 16 + j] = tir.load("float32", B, j) * tir.float32(2)


def test_vthread():
    func = vthread_func
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func, True)


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = tir.float32(0)
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def matmul_original(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)

        for k in range(128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")

    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * tir.float32(2)

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def predicate(b: ty.handle, c: ty.handle) -> None:
    B = tir.match_buffer(b, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")

    for i, jo, ji in tir.grid(16, 4, 5):
        with tir.block([16, 16], "update") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, jo * 4 + ji)
            tir.where(jo * 4 + ji < 16)
            C[vi, vj] = B[vi, vj] + tir.float32(1)


def test_module_define():
    func1 = tvm.script.create_module({"matmul": matmul})["matmul"]
    func2 = tvm.script.create_module({"element_wise": element_wise})["element_wise"]
    func3 = tvm.script.create_module({"predicate": predicate})["predicate"]
    mod1 = tvm.script.create_module({"func1": func1, "func2": func2, "func3": func3})
    mod2 = tvm.script.create_module({"func1": matmul, "func2": element_wise, "func3": predicate})
    tvm.ir.assert_structural_equal(mod1, mod2)


def test_matmul():
    func = matmul
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


def test_matmul_original():
    func = matmul_original
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body.body, tir.stmt.SeqStmt)
    assert isinstance(rt_func.body.block.body.body.body[0].block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body.body.body[1], tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body.body[1].body.block, tir.stmt.Block)


def test_element_wise():
    func = element_wise
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.SeqStmt)
    assert isinstance(rt_func.body.block.body[0], tir.stmt.For)
    assert isinstance(rt_func.body.block.body[0].body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body[0].body.body.block, tir.stmt.Block)

    assert isinstance(rt_func.body.block.body[1], tir.stmt.For)
    assert isinstance(rt_func.body.block.body[1].body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body[1].body.body.block, tir.stmt.Block)


def test_predicate():
    func = predicate
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body.body, tir.stmt.For)
    assert isinstance(rt_func.body.block.body.body.body.body.block, tir.stmt.Block)


@tvm.script.tir
def for_thread_binding(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")

    for i in tir.thread_binding(0, 16, thread="threadIdx.x"):
        for j in tir.thread_binding(
            0, 16, thread="threadIdx.y", annotations={"attr_key": "attr_value"}
        ):
            A[i, j] = B[i, j] + tir.float32(1)


def test_for_thread_binding():
    func = for_thread_binding
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body, tir.stmt.For)
    assert rt_func.body.kind == 4
    assert rt_func.body.thread_binding.thread_tag == "threadIdx.x"
    assert isinstance(rt_func.body.body, tir.stmt.For)
    assert rt_func.body.body.kind == 4
    assert rt_func.body.body.thread_binding.thread_tag == "threadIdx.y"
    assert rt_func.body.body.annotations["attr_key"] == "attr_value"


@tvm.script.tir
def match_buffer_region(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16), "float32")
    B = tir.match_buffer(b, (1), "float32")

    with tir.block([16, 4]) as [vi, vj]:
        C = tir.match_buffer(A[0:16, vi, vj * 4 : vj * 4 + 4], (16, 1, 4))
        with tir.block([4]) as [vii]:
            D = tir.match_buffer(C[vii * 4 : vii * 4 + 4, 0, 0:4], (4, 1, 4))
            for i, j in tir.grid(4, 4):
                B[0] += D[i, 0, j]


def test_match_buffer_region():
    func = match_buffer_region
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body, tir.stmt.BlockRealize)
    root = rt_func.body.block

    assert isinstance(root.body, tir.stmt.For)
    assert isinstance(root.body.body, tir.stmt.For)
    assert isinstance(root.body.body.body, tir.stmt.BlockRealize)
    outer_block = root.body.body.body.block
    assert len(outer_block.match_buffers) == 1
    buffer_C = outer_block.match_buffers[0].buffer
    tvm.ir.assert_structural_equal(buffer_C.shape, [16, 1, 4])

    assert isinstance(outer_block.body, tir.stmt.For)
    assert isinstance(outer_block.body.body, tir.stmt.BlockRealize)
    inner_block = outer_block.body.body.block
    assert len(inner_block.match_buffers) == 1
    buffer_D = inner_block.match_buffers[0].buffer
    tvm.ir.assert_structural_equal(buffer_D.shape, [4, 1, 4])


@tvm.script.tir
def block_elements(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    B = tir.match_buffer(b, (1, 1), "float32")

    with tir.block([1], "update") as [vi]:
        tir.bind(vi, 0)
        tir.where(True)
        tir.reads(A[0:16, 0:16])
        tir.writes(B[0, 0])
        tir.block_attr({"attr_key": "attr_value"})
        C = tir.alloc_buffer((4, 4), dtype="float32")
        D = tir.match_buffer(A[0:4, 0], (4, 1))
        with tir.init():
            B[0, 0] = tir.float32(0)
        B[0, 0] = A[0, 0] + B[0, 0] + C[1, 1] + D[2]


def test_block_elements():
    func = block_elements
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.BlockRealize)
    assert isinstance(rt_func.body.block.body.block, tir.stmt.Block)
    block = rt_func.body.block.body.block
    assert isinstance(block.body, tir.stmt.BufferStore)
    assert isinstance(block.init, tir.stmt.BufferStore)
    assert len(block.annotations) == 1
    assert block.annotations["attr_key"] == "attr_value"


@tvm.script.tir
def opaque_block(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")

    for i in range(16):
        for j in range(16):
            with tir.block([]):
                tir.reads([])
                tir.writes(A[i, j])
                A[i, j] = tir.float32(0)
        with tir.block([]):
            tir.reads([A[i, 0:16]])
            tir.writes([B[i, 0:16]])
            for j in range(16):
                B[i, j] = A[i, j]


def test_opaque_block():
    func = opaque_block
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    root_block = rt_func.body.block
    assert isinstance(root_block, tir.stmt.Block)
    assert isinstance(root_block.body, tir.stmt.For)
    assert isinstance(root_block.body.body[0], tir.stmt.For)
    assert isinstance(root_block.body.body[0].body, tir.stmt.BlockRealize)
    assert isinstance(root_block.body.body[0].body.block, tir.stmt.Block)
    assert len(root_block.body.body[0].body.block.iter_vars) == 0
    assert isinstance(root_block.body.body[1], tir.stmt.BlockRealize)
    assert isinstance(root_block.body.body[1].block, tir.stmt.Block)
    assert len(root_block.body.body[1].block.iter_vars) == 0


@tvm.script.tir
def rank0(a: ty.handle) -> None:
    A = tir.match_buffer(a, (), "float32")
    B = tir.alloc_buffer((), "float32")
    A[()] = 2
    B[()] = A[()]


def test_rank0_buffers():
    func = rank0
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def rank0_block(a: ty.handle) -> None:
    A = tir.match_buffer(a, (), "float32")
    B = tir.alloc_buffer((), "float32")
    tir.store(B.data, 0, tir.load("float32", A.data, 0))

    with tir.block([], "update") as []:
        tir.reads([A[()]])
        tir.writes([B[()]])
        for i in range(1):
            B[()] = A[()]


def test_rank0_blocks():
    func = rank0_block
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def select(a: ty.handle) -> None:
    A = tir.match_buffer(a, (), "float32")
    A[()] = tir.Select(True, 1, 2)


def test_select():
    func = select
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def minmax(a: ty.handle) -> None:
    A = tir.match_buffer(a, (), "float32")
    A[()] = tir.min(1, 2)
    A[()] = tir.max(1, 2)


def test_minmax():
    func = minmax
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def abs(a: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")

    with tir.block([128, 128], "A") as [vi, vj]:
        A[vi, vj] = tir.abs(A[vi, vj])


def test_abs():
    func = abs
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def constant_folding(a: ty.handle) -> None:
    A = tir.match_buffer(a, (), "float32")
    A[()] = tir.min(2.2, 5.2)
    A[()] = tir.max(tir.float32(2.2), tir.float32(tir.float32(5.2)))
    A[()] = tir.min(2.2, 5.0)


def test_constant_folding():
    func = constant_folding
    rt_func = tvm.script.from_source(tvm.script.asscript(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)


@tvm.script.tir
def simplify_bracket() -> None:
    a = tir.var("int32")
    b = tir.var("int32")
    c = tir.var("int32")
    d = tir.var("int32")
    tir.evaluate(a + b * (c + d))


def test_simplify_bracket():
    func = simplify_bracket
    out_str = tvm.script.asscript(func, True)
    assert out_str.count("a + b*(c + d)") == 1


@tvm.script.tir
def var_with_same_name(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        A[vi, vj] = 0
    with tir.block([16, 16]) as [vi, vj]:
        A[vi, vj] = 0
    for i, j in tir.grid(16, 16):
        with tir.block([16, 16]) as [vi, vj]:
            A[vi, vj] = 0
    for i, j in tir.grid(16, 16):
        with tir.block([16, 16]) as [vi, vj]:
            A[vi, vj] = 0


def test_same_name_var():
    func = var_with_same_name
    out_str = tvm.script.asscript(func, True)
    rt_func = tvm.script.from_source(out_str)
    tvm.ir.assert_structural_equal(func, rt_func)

    assert out_str.count("with tir.block([16, 16]) as [vi, vj]") == 4
    assert out_str.find("vi_") == -1
    assert out_str.find("vj_") == -1

    assert out_str.count("for i0, i1 in tir.grid(16, 16)") == 2
    assert out_str.find("i0_") == -1
    assert out_str.find("i1_") == -1

    assert out_str.count("for i, j in tir.grid(16, 16)") == 2
    assert out_str.find("i_") == -1
    assert out_str.find("i_") == -1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
