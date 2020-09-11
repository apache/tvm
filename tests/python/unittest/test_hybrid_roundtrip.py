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
from tvm import tir
from tvm.hybrid import ty


@tvm.hybrid.script
class Module1:
    def mmult(A: ty.handle, B: ty.handle, C: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "mmult", "tir.noalias": True})
        # buffer definition
        C_global = tir.buffer_decl([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        packedB = tir.buffer_decl([32, 1024, 32], elem_offset=0, align=128, offset_factor=1)
        A_1 = tir.buffer_bind(A, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        B_1 = tir.buffer_bind(B, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        C_1 = tir.buffer_bind(C, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        # body
        tir.attr(packedB, "realize_scope", "")
        tir.realize(packedB[0:32, 0:1024, 0:32])
        for x in tir.range(0, 32, "parallel"):
            for y in tir.range(0, 1024):
                for z in tir.range(0, 32, "vectorized"):
                    packedB[x, y, z] = B_1[y, ((x * 32) + z)]
        tir.attr(C_1, "realize_scope", "")
        tir.realize(C_1[0:1024, 0:1024])
        for x_outer in tir.range(0, 32, "parallel"):
            for y_outer in tir.range(0, 32):
                tir.attr(C_global, "realize_scope", "global")
                tir.realize(
                    C_global[
                        (x_outer * 32) : ((x_outer * 32) + 32),
                        (y_outer * 32) : ((y_outer * 32) + 32),
                    ]
                )
                for x_c_init in tir.range(0, 32):
                    for y_c_init in tir.range(0, 32, "vectorized"):
                        C_global[
                            (x_c_init + (x_outer * 32)), (y_c_init + (y_outer * 32))
                        ] = tir.float32(0)
                for k_outer in tir.range(0, 256):
                    for x_c in tir.range(0, 32):
                        for k_inner in tir.range(0, 4, "unroll"):
                            for y_c in tir.range(0, 32, "vectorized"):
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
                for x_inner in tir.range(0, 32):
                    for y_inner in tir.range(0, 32):
                        C_1[(x_inner + (x_outer * 32)), (y_inner + (y_outer * 32))] = C_global[
                            (x_inner + (x_outer * 32)), (y_inner + (y_outer * 32))
                        ]


def test_opt_gemm_normalize():
    mod = Module1()
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.hybrid.script
class Module2:
    def mmult(A: ty.handle, B: ty.handle, C: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "mmult", "tir.noalias": True})
        # var definition
        C_global = tir.var("handle")
        packedB = tir.var("handle")
        A_1 = tir.buffer_bind(A, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        B_1 = tir.buffer_bind(B, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        C_1 = tir.buffer_bind(C, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
        # body
        tir.attr(packedB, "storage_scope", "global")
        tir.allocate(packedB, "float32x32", [32768])
        tir.attr(C_global, "storage_scope", "global")
        tir.allocate(C_global, "float32", [1024])
        for x in tir.range(0, 32, "parallel"):
            for y in tir.range(0, 1024):
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
        for x_outer in tir.range(0, 32):
            for y_outer in tir.range(0, 32):
                for x_c_init in tir.range(0, 32):
                    tir.store(
                        C_global,
                        tir.ramp((x_c_init * 32), 1, 32),
                        tir.broadcast(tir.float32(0), 32),
                        tir.broadcast(True, 32),
                    )
                for k_outer in tir.range(0, 256):
                    for x_c in tir.range(0, 32):
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
                for x_inner in tir.range(0, 32):
                    for y_inner in tir.range(0, 32):
                        C_1.data[
                            ((((x_outer * 32768) + (x_inner * 1024)) + (y_outer * 32)) + y_inner)
                        ] = tir.load("float32", C_global, ((x_inner * 32) + y_inner))


def test_opt_gemm_lower():
    mod = Module2()
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.hybrid.script
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
        C_global = tir.var("handle")
        packedB = tir.var("handle")
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
            "int32", tir.load("int64", arg0_shape, 0)
        ), "Argument arg0.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            "int32", tir.load("int64", arg0_shape, 1)
        ), "Argument arg0.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg0_strides, dtype="bool")):
            assert (1 == tir.cast("int32", tir.load("int64", arg0_strides, 1))) and (
                1024 == tir.cast("int32", tir.load("int64", arg0_strides, 0))
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
            "int32", tir.load("int64", arg1_shape, 0)
        ), "Argument arg1.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            "int32", tir.load("int64", arg1_shape, 1)
        ), "Argument arg1.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg1_strides, dtype="bool")):
            assert (1 == tir.cast("int32", tir.load("int64", arg1_strides, 1))) and (
                1024 == tir.cast("int32", tir.load("int64", arg1_strides, 0))
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
            "int32", tir.load("int64", arg2_shape, 0)
        ), "Argument arg2.shape[0] has an unsatisfied constraint"
        assert 1024 == tir.cast(
            "int32", tir.load("int64", arg2_shape, 1)
        ), "Argument arg2.shape[1] has an unsatisfied constraint"
        if not (tir.isnullptr(arg2_strides, dtype="bool")):
            assert (1 == tir.cast("int32", tir.load("int64", arg2_strides, 1))) and (
                1024 == tir.cast("int32", tir.load("int64", arg2_strides, 0))
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
            for x in tir.range(0, 32, "parallel"):
                for y in tir.range(0, 1024):
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
            for x_outer in tir.range(0, 32, "parallel"):
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
                    for y_outer in tir.range(0, 32):
                        for x_c_init in tir.range(0, 32):
                            tir.store(
                                C_global,
                                tir.ramp((x_c_init * 32), 1, 32),
                                tir.broadcast(tir.float32(0), 32),
                                tir.broadcast(True, 32),
                            )
                        for k_outer in tir.range(0, 256):
                            for x_c in tir.range(0, 32):
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
                        for x_inner in tir.range(0, 32):
                            for y_inner in tir.range(0, 32):
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
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.hybrid.script
def opt_conv_tensorcore_normalize(A: ty.handle, W: ty.handle, Conv: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # var definition
    blockIdx_x = tir.var("int32")
    blockIdx_y = tir.var("int32")
    blockIdx_z = tir.var("int32")
    threadIdx_x = tir.var("int32")
    threadIdx_y = tir.var("int32")
    threadIdx_z = tir.var("int32")
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
    A_1 = tir.buffer_bind(
        A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_1 = tir.buffer_bind(
        W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Conv_1 = tir.buffer_bind(
        Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    # body
    tir.attr(Conv_1, "realize_scope", "")
    tir.realize(Conv_1[0:16, 0:14, 0:14, 0:32, 0:16, 0:16])
    tir.attr(tir.iter_var(blockIdx_z, None, "ThreadIndex", "blockIdx.z"), "thread_extent", 196)
    tir.attr(tir.iter_var(blockIdx_x, None, "ThreadIndex", "blockIdx.x"), "thread_extent", 2)
    tir.attr(tir.iter_var(blockIdx_y, None, "ThreadIndex", "blockIdx.y"), "thread_extent", 4)
    tir.attr(tir.iter_var(threadIdx_y, None, "ThreadIndex", "threadIdx.y"), "thread_extent", 4)
    tir.attr(tir.iter_var(threadIdx_z, None, "ThreadIndex", "threadIdx.z"), "thread_extent", 2)
    tir.attr(Conv_wmma_accumulator, "realize_scope", "wmma.accumulator")
    tir.realize(
        Conv_wmma_accumulator[
            ((blockIdx_x * 8) + (threadIdx_y * 2)) : (((blockIdx_x * 8) + (threadIdx_y * 2)) + 2),
            tir.floordiv(blockIdx_z, 14) : (tir.floordiv(blockIdx_z, 14) + 1),
            tir.floormod(blockIdx_z, 14) : (tir.floormod(blockIdx_z, 14) + 1),
            ((blockIdx_y * 8) + (threadIdx_z * 4)) : (((blockIdx_y * 8) + (threadIdx_z * 4)) + 4),
            0:16,
            0:16,
        ]
    )
    for n_c_init in tir.range(0, 2):
        for o_c_init in tir.range(0, 4):
            tir.attr(
                [BC, Conv_wmma_accumulator],
                "buffer_bind_scope",
                tir.tvm_tuple(
                    (n_c_init + ((blockIdx_x * 8) + (threadIdx_y * 2))),
                    1,
                    tir.floordiv(blockIdx_z, 14),
                    1,
                    tir.floormod(blockIdx_z, 14),
                    1,
                    (o_c_init + ((blockIdx_y * 8) + (threadIdx_z * 4))),
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
    for ic_outer in tir.range(0, 8):
        for kh in tir.range(0, 3):
            tir.attr(Apad_shared, "realize_scope", "shared")
            tir.realize(
                Apad_shared[
                    (blockIdx_x * 8) : ((blockIdx_x * 8) + 8),
                    (tir.floordiv(blockIdx_z, 14) + kh) : ((tir.floordiv(blockIdx_z, 14) + kh) + 1),
                    tir.floormod(blockIdx_z, 14) : (tir.floormod(blockIdx_z, 14) + 3),
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    0:16,
                    0:16,
                ]
            )
            for ax2 in tir.range(0, 3):
                for ax3 in tir.range(0, 2):
                    for ax4_ax5_fused_outer in tir.range(0, 8):
                        tir.attr(
                            tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                            "thread_extent",
                            32,
                        )
                        Apad_shared[
                            ((threadIdx_z + (threadIdx_y * 2)) + (blockIdx_x * 8)),
                            (tir.floordiv(blockIdx_z, 14) + kh),
                            (ax2 + tir.floormod(blockIdx_z, 14)),
                            (ax3 + (ic_outer * 2)),
                            tir.floordiv((threadIdx_x + (ax4_ax5_fused_outer * 32)), 16),
                            tir.floormod((threadIdx_x + (ax4_ax5_fused_outer * 32)), 16),
                        ] = tir.if_then_else(
                            (
                                (
                                    (
                                        ((tir.floordiv(blockIdx_z, 14) + kh) >= 1)
                                        and (((tir.floordiv(blockIdx_z, 14) + kh) - 1) < 14)
                                    )
                                    and ((ax2 + tir.floormod(blockIdx_z, 14)) >= 1)
                                )
                                and (((ax2 + tir.floormod(blockIdx_z, 14)) - 1) < 14)
                            ),
                            A_1[
                                ((threadIdx_z + (threadIdx_y * 2)) + (blockIdx_x * 8)),
                                ((tir.floordiv(blockIdx_z, 14) + kh) - 1),
                                ((ax2 + tir.floormod(blockIdx_z, 14)) - 1),
                                (ax3 + (ic_outer * 2)),
                                tir.floordiv((threadIdx_x + (ax4_ax5_fused_outer * 32)), 16),
                                tir.floormod((threadIdx_x + (ax4_ax5_fused_outer * 32)), 16),
                            ],
                            tir.float16(0),
                            dtype="float16",
                        )
            tir.attr(W_shared, "realize_scope", "shared")
            tir.realize(
                W_shared[
                    kh : (kh + 1),
                    0:3,
                    (ic_outer * 2) : ((ic_outer * 2) + 2),
                    (blockIdx_y * 8) : ((blockIdx_y * 8) + 8),
                    0:16,
                    0:16,
                ]
            )
            for ax1 in tir.range(0, 3):
                for ax2_1 in tir.range(0, 2):
                    tir.attr(
                        tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                        "thread_extent",
                        32,
                    )
                    for ax4_ax5_fused_inner in tir.range(0, 8, "vectorized"):
                        W_shared[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((threadIdx_z + (threadIdx_y * 2)) + (blockIdx_y * 8)),
                            tir.floordiv((ax4_ax5_fused_inner + (threadIdx_x * 8)), 16),
                            tir.floormod((ax4_ax5_fused_inner + (threadIdx_x * 8)), 16),
                        ] = W_1[
                            kh,
                            ax1,
                            (ax2_1 + (ic_outer * 2)),
                            ((threadIdx_z + (threadIdx_y * 2)) + (blockIdx_y * 8)),
                            tir.floordiv((ax4_ax5_fused_inner + (threadIdx_x * 8)), 16),
                            tir.floormod((ax4_ax5_fused_inner + (threadIdx_x * 8)), 16),
                        ]
            for ic_inner in tir.range(0, 2):
                for kw in tir.range(0, 3):
                    tir.attr(Apad_shared_wmma_matrix_a, "realize_scope", "wmma.matrix_a")
                    tir.realize(
                        Apad_shared_wmma_matrix_a[
                            ((blockIdx_x * 8) + (threadIdx_y * 2)) : (
                                ((blockIdx_x * 8) + (threadIdx_y * 2)) + 2
                            ),
                            (tir.floordiv(blockIdx_z, 14) + kh) : (
                                (tir.floordiv(blockIdx_z, 14) + kh) + 1
                            ),
                            (kw + tir.floormod(blockIdx_z, 14)) : (
                                (kw + tir.floormod(blockIdx_z, 14)) + 1
                            ),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            0:16,
                            0:16,
                        ]
                    )
                    for ax0 in tir.range(0, 2):
                        tir.attr(
                            [buffer, Apad_shared],
                            "buffer_bind_scope",
                            tir.tvm_tuple(
                                (ax0 + ((blockIdx_x * 8) + (threadIdx_y * 2))),
                                1,
                                (tir.floordiv(blockIdx_z, 14) + kh),
                                1,
                                (kw + tir.floormod(blockIdx_z, 14)),
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
                                (ax0 + ((blockIdx_x * 8) + (threadIdx_y * 2))),
                                1,
                                (tir.floordiv(blockIdx_z, 14) + kh),
                                1,
                                (kw + tir.floormod(blockIdx_z, 14)),
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
                    tir.attr(W_shared_wmma_matrix_b, "realize_scope", "wmma.matrix_b")
                    tir.realize(
                        W_shared_wmma_matrix_b[
                            kh : (kh + 1),
                            kw : (kw + 1),
                            ((ic_outer * 2) + ic_inner) : (((ic_outer * 2) + ic_inner) + 1),
                            ((blockIdx_y * 8) + (threadIdx_z * 4)) : (
                                ((blockIdx_y * 8) + (threadIdx_z * 4)) + 4
                            ),
                            0:16,
                            0:16,
                        ]
                    )
                    for ax3_1 in tir.range(0, 4):
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
                                (ax3_1 + ((blockIdx_y * 8) + (threadIdx_z * 4))),
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
                                (ax3_1 + ((blockIdx_y * 8) + (threadIdx_z * 4))),
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
                    for n_c in tir.range(0, 2):
                        for o_c in tir.range(0, 4):
                            tir.attr(
                                [BA, Apad_shared_wmma_matrix_a],
                                "buffer_bind_scope",
                                tir.tvm_tuple(
                                    (n_c + ((blockIdx_x * 8) + (threadIdx_y * 2))),
                                    1,
                                    (tir.floordiv(blockIdx_z, 14) + kh),
                                    1,
                                    (tir.floormod(blockIdx_z, 14) + kw),
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
                                    (o_c + ((blockIdx_y * 8) + (threadIdx_z * 4))),
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
                                    (n_c + ((blockIdx_x * 8) + (threadIdx_y * 2))),
                                    1,
                                    tir.floordiv(blockIdx_z, 14),
                                    1,
                                    tir.floormod(blockIdx_z, 14),
                                    1,
                                    (o_c + ((blockIdx_y * 8) + (threadIdx_z * 4))),
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
    for n_inner in tir.range(0, 2):
        for o_inner in tir.range(0, 4):
            tir.attr(
                [buffer_4, Conv_wmma_accumulator],
                "buffer_bind_scope",
                tir.tvm_tuple(
                    ((((blockIdx_x * 4) + threadIdx_y) * 2) + n_inner),
                    1,
                    tir.floordiv(blockIdx_z, 14),
                    1,
                    tir.floormod(blockIdx_z, 14),
                    1,
                    ((((blockIdx_y * 2) + threadIdx_z) * 4) + o_inner),
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
                    ((((blockIdx_x * 4) + threadIdx_y) * 2) + n_inner),
                    1,
                    tir.floordiv(blockIdx_z, 14),
                    1,
                    tir.floormod(blockIdx_z, 14),
                    1,
                    ((((blockIdx_y * 2) + threadIdx_z) * 4) + o_inner),
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
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.hybrid.script
def opt_conv_tensorcore_lower(A: ty.handle, W: ty.handle, Conv: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # var definition
    Apad_shared = tir.var("handle")
    Apad_shared_wmma_matrix_a = tir.var("handle")
    Conv_wmma_accumulator = tir.var("handle")
    W_shared = tir.var("handle")
    W_shared_wmma_matrix_b = tir.var("handle")
    blockIdx_x = tir.var("int32")
    blockIdx_y = tir.var("int32")
    blockIdx_z = tir.var("int32")
    threadIdx_x = tir.var("int32")
    threadIdx_y = tir.var("int32")
    threadIdx_z = tir.var("int32")
    A_1 = tir.buffer_bind(
        A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    W_1 = tir.buffer_bind(
        W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )
    Conv_1 = tir.buffer_bind(
        Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1
    )
    # body
    tir.attr(tir.iter_var(blockIdx_z, None, "ThreadIndex", "blockIdx.z"), "thread_extent", 196)
    tir.attr(Conv_wmma_accumulator, "storage_scope", "wmma.accumulator")
    tir.allocate(Conv_wmma_accumulator, "float32", [2048])
    tir.attr(Apad_shared, "storage_scope", "shared")
    tir.allocate(Apad_shared, "float16", [12288])
    tir.attr(W_shared, "storage_scope", "shared")
    tir.allocate(W_shared, "float16", [12288])
    tir.attr(Apad_shared_wmma_matrix_a, "storage_scope", "wmma.matrix_a")
    tir.allocate(Apad_shared_wmma_matrix_a, "float16", [512])
    tir.attr(W_shared_wmma_matrix_b, "storage_scope", "wmma.matrix_b")
    tir.allocate(W_shared_wmma_matrix_b, "float16", [1024])
    tir.attr(tir.iter_var(blockIdx_x, None, "ThreadIndex", "blockIdx.x"), "thread_extent", 2)
    tir.attr(tir.iter_var(blockIdx_y, None, "ThreadIndex", "blockIdx.y"), "thread_extent", 4)
    tir.attr(tir.iter_var(threadIdx_y, None, "ThreadIndex", "threadIdx.y"), "thread_extent", 4)
    tir.attr(tir.iter_var(threadIdx_z, None, "ThreadIndex", "threadIdx.z"), "thread_extent", 2)
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
    for ic_outer in tir.range(0, 8):
        for kh in tir.range(0, 3):
            for ax2 in tir.range(0, 3):
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                            + threadIdx_x
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61440
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 32
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61408
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 64
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61376
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 96
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61344
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 128
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61312
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 160
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61280
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 192
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61248
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 224
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61216
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 256
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61184
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 288
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61152
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 320
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61120
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 352
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61088
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 384
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61056
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 416
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 61024
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                with tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                ):
                    Apad_shared[
                        (
                            (
                                (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                                + threadIdx_x
                            )
                            + 448
                        )
                    ] = tir.if_then_else(
                        (
                            (
                                (
                                    (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                    and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                                )
                                and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                            )
                            and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                        (
                                                            (blockIdx_x * 6422528)
                                                            + (threadIdx_y * 1605632)
                                                        )
                                                        + (threadIdx_z * 802816)
                                                    )
                                                    + (kh * 57344)
                                                )
                                                + (blockIdx_z * 4096)
                                            )
                                            + (ax2 * 4096)
                                        )
                                        + (ic_outer * 512)
                                    )
                                    + threadIdx_x
                                )
                                - 60992
                            ),
                        ),
                        tir.float16(0),
                        dtype="float16",
                    )
                tir.attr(
                    tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"),
                    "thread_extent",
                    32,
                )
                Apad_shared[
                    (
                        (
                            (((threadIdx_y * 3072) + (threadIdx_z * 1536)) + (ax2 * 512))
                            + threadIdx_x
                        )
                        + 480
                    )
                ] = tir.if_then_else(
                    (
                        (
                            (
                                (1 <= (tir.floordiv(blockIdx_z, 14) + kh))
                                and ((tir.floordiv(blockIdx_z, 14) + kh) < 15)
                            )
                            and (1 <= (ax2 + tir.floormod(blockIdx_z, 14)))
                        )
                        and ((ax2 + tir.floormod(blockIdx_z, 14)) < 15)
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
                                                    (
                                                        (blockIdx_x * 6422528)
                                                        + (threadIdx_y * 1605632)
                                                    )
                                                    + (threadIdx_z * 802816)
                                                )
                                                + (kh * 57344)
                                            )
                                            + (blockIdx_z * 4096)
                                        )
                                        + (ax2 * 4096)
                                    )
                                    + (ic_outer * 512)
                                )
                                + threadIdx_x
                            )
                            - 60960
                        ),
                    ),
                    tir.float16(0),
                    dtype="float16",
                )
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        (((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)), 1, 8
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (((kh * 393216) + (ic_outer * 16384)) + (blockIdx_y * 2048))
                                        + (threadIdx_y * 512)
                                    )
                                    + (threadIdx_z * 256)
                                )
                                + (threadIdx_x * 8)
                            ),
                            1,
                            8,
                        ),
                        tir.broadcast(True, 8),
                    ),
                    tir.broadcast(True, 8),
                )
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        ((((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)) + 2048),
                        1,
                        8,
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((kh * 393216) + (ic_outer * 16384))
                                                + (blockIdx_y * 2048)
                                            )
                                            + (threadIdx_y * 512)
                                        )
                                        + (threadIdx_z * 256)
                                    )
                                    + (threadIdx_x * 8)
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
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        ((((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)) + 4096),
                        1,
                        8,
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((kh * 393216) + (ic_outer * 16384))
                                                + (blockIdx_y * 2048)
                                            )
                                            + (threadIdx_y * 512)
                                        )
                                        + (threadIdx_z * 256)
                                    )
                                    + (threadIdx_x * 8)
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
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        ((((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)) + 6144),
                        1,
                        8,
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((kh * 393216) + (ic_outer * 16384))
                                                + (blockIdx_y * 2048)
                                            )
                                            + (threadIdx_y * 512)
                                        )
                                        + (threadIdx_z * 256)
                                    )
                                    + (threadIdx_x * 8)
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
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        ((((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)) + 8192),
                        1,
                        8,
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((kh * 393216) + (ic_outer * 16384))
                                                + (blockIdx_y * 2048)
                                            )
                                            + (threadIdx_y * 512)
                                        )
                                        + (threadIdx_z * 256)
                                    )
                                    + (threadIdx_x * 8)
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
            with tir.attr(
                tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32
            ):
                tir.store(
                    W_shared,
                    tir.ramp(
                        ((((threadIdx_y * 512) + (threadIdx_z * 256)) + (threadIdx_x * 8)) + 10240),
                        1,
                        8,
                    ),
                    tir.load(
                        "float16x8",
                        W_1.data,
                        tir.ramp(
                            (
                                (
                                    (
                                        (
                                            (
                                                ((kh * 393216) + (ic_outer * 16384))
                                                + (blockIdx_y * 2048)
                                            )
                                            + (threadIdx_y * 512)
                                        )
                                        + (threadIdx_z * 256)
                                    )
                                    + (threadIdx_x * 8)
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
            for ic_inner in tir.range(0, 2):
                for kw in tir.range(0, 3):
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
                                (((threadIdx_y * 3072) + (kw * 512)) + (ic_inner * 256)),
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
                                ((((threadIdx_y * 3072) + (kw * 512)) + (ic_inner * 256)) + 1536),
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
                                (((kw * 4096) + (ic_inner * 2048)) + (threadIdx_z * 1024)),
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
                                ((((kw * 4096) + (ic_inner * 2048)) + (threadIdx_z * 1024)) + 256),
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
                                ((((kw * 4096) + (ic_inner * 2048)) + (threadIdx_z * 1024)) + 512),
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
                                ((((kw * 4096) + (ic_inner * 2048)) + (threadIdx_z * 1024)) + 768),
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
                (
                    (
                        (((blockIdx_x * 12845056) + (threadIdx_y * 3211264)) + (blockIdx_z * 8192))
                        + (blockIdx_y * 2048)
                    )
                    + (threadIdx_z * 1024)
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
            1,
            tir.tvm_access_ptr(
                tir.type_annotation(dtype="float32"),
                Conv_1.data,
                (
                    (
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
                        (
                            (
                                ((blockIdx_x * 12845056) + (threadIdx_y * 3211264))
                                + (blockIdx_z * 8192)
                            )
                            + (blockIdx_y * 2048)
                        )
                        + (threadIdx_z * 1024)
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
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


@tvm.hybrid.script
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
        "int32", tir.load("int64", arg0_shape, 0)
    ), "Argument arg0.shape[0] has an unsatisfied constraint"
    assert 14 == tir.cast(
        "int32", tir.load("int64", arg0_shape, 1)
    ), "Argument arg0.shape[1] has an unsatisfied constraint"
    assert 14 == tir.cast(
        "int32", tir.load("int64", arg0_shape, 2)
    ), "Argument arg0.shape[2] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg0_shape, 3)
    ), "Argument arg0.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg0_shape, 4)
    ), "Argument arg0.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg0_shape, 5)
    ), "Argument arg0.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg0_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast("int32", tir.load("int64", arg0_strides, 5)))
                        and (16 == tir.cast("int32", tir.load("int64", arg0_strides, 4)))
                    )
                    and (256 == tir.cast("int32", tir.load("int64", arg0_strides, 3)))
                )
                and (4096 == tir.cast("int32", tir.load("int64", arg0_strides, 2)))
            )
            and (57344 == tir.cast("int32", tir.load("int64", arg0_strides, 1)))
        ) and (
            802816 == tir.cast("int32", tir.load("int64", arg0_strides, 0))
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
        "int32", tir.load("int64", arg1_shape, 0)
    ), "Argument arg1.shape[0] has an unsatisfied constraint"
    assert 3 == tir.cast(
        "int32", tir.load("int64", arg1_shape, 1)
    ), "Argument arg1.shape[1] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg1_shape, 2)
    ), "Argument arg1.shape[2] has an unsatisfied constraint"
    assert 32 == tir.cast(
        "int32", tir.load("int64", arg1_shape, 3)
    ), "Argument arg1.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg1_shape, 4)
    ), "Argument arg1.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg1_shape, 5)
    ), "Argument arg1.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg1_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast("int32", tir.load("int64", arg1_strides, 5)))
                        and (16 == tir.cast("int32", tir.load("int64", arg1_strides, 4)))
                    )
                    and (256 == tir.cast("int32", tir.load("int64", arg1_strides, 3)))
                )
                and (8192 == tir.cast("int32", tir.load("int64", arg1_strides, 2)))
            )
            and (131072 == tir.cast("int32", tir.load("int64", arg1_strides, 1)))
        ) and (
            393216 == tir.cast("int32", tir.load("int64", arg1_strides, 0))
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
        "int32", tir.load("int64", arg2_shape, 0)
    ), "Argument arg2.shape[0] has an unsatisfied constraint"
    assert 14 == tir.cast(
        "int32", tir.load("int64", arg2_shape, 1)
    ), "Argument arg2.shape[1] has an unsatisfied constraint"
    assert 14 == tir.cast(
        "int32", tir.load("int64", arg2_shape, 2)
    ), "Argument arg2.shape[2] has an unsatisfied constraint"
    assert 32 == tir.cast(
        "int32", tir.load("int64", arg2_shape, 3)
    ), "Argument arg2.shape[3] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg2_shape, 4)
    ), "Argument arg2.shape[4] has an unsatisfied constraint"
    assert 16 == tir.cast(
        "int32", tir.load("int64", arg2_shape, 5)
    ), "Argument arg2.shape[5] has an unsatisfied constraint"
    if not (tir.isnullptr(arg2_strides, dtype="bool")):
        assert (
            (
                (
                    (
                        (1 == tir.cast("int32", tir.load("int64", arg2_strides, 5)))
                        and (16 == tir.cast("int32", tir.load("int64", arg2_strides, 4)))
                    )
                    and (256 == tir.cast("int32", tir.load("int64", arg2_strides, 3)))
                )
                and (8192 == tir.cast("int32", tir.load("int64", arg2_strides, 2)))
            )
            and (114688 == tir.cast("int32", tir.load("int64", arg2_strides, 1)))
        ) and (
            1605632 == tir.cast("int32", tir.load("int64", arg2_strides, 0))
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
    tir.evaluate(tir.tvm_struct_set(stack_value, 0, 12, tir.cast("int64", 2), dtype="int32"))
    stack_tcode[0] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 1, 12, tir.cast("int64", dev_id), dtype="int32"))
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
    tir.evaluate(tir.tvm_struct_set(stack_value, 3, 12, tir.cast("int64", 196), dtype="int32"))
    stack_tcode[3] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 4, 12, tir.cast("int64", 2), dtype="int32"))
    stack_tcode[4] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 5, 12, tir.cast("int64", 4), dtype="int32"))
    stack_tcode[5] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 6, 12, tir.cast("int64", 4), dtype="int32"))
    stack_tcode[6] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 7, 12, tir.cast("int64", 2), dtype="int32"))
    stack_tcode[7] = 0
    tir.evaluate(tir.tvm_struct_set(stack_value, 8, 12, tir.cast("int64", 32), dtype="int32"))
    stack_tcode[8] = 0
    tir.evaluate(
        tir.tvm_call_packed_lowered(
            "default_function_kernel0", stack_value, stack_tcode, 0, 9, dtype="int32"
        )
    )


def test_opt_conv_tensorcore_mod_host():
    mod = opt_conv_tensorcore_mod_host
    rt_mod = tvm.hybrid.from_source(tvm.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod, True)


if __name__ == "__main__":
    test_opt_gemm_normalize()
    test_opt_gemm_mod_host()
    test_opt_gemm_lower()
    test_opt_conv_tensorcore_normalize()
    test_opt_conv_tensorcore_lower()
    test_opt_conv_tensorcore_mod_host()
