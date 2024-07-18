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

# This test runs the reduce_pad_branch_through_over_compute test to check if we are able to eliminate the redundant pad branch and overcompute the value.
# This helps to expose more opportunities to vectorize the code.

import tvm
import tvm.testing
from tvm import relax

import tvm.script
from tvm.script import tir as T, relax as R


@tvm.script.ir_module
class Add_PrimFunc_Before:
    @T.prim_func(private=True)
    def add(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.add", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "add",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("compute"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )
                T.writes(compute[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                compute[
                    v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6
                ] = T.if_then_else(
                    v_axis1 == T.int64(3)
                    and T.int64(4) <= v_axis4
                    or v_axis2 == T.int64(3)
                    and T.int64(4) <= v_axis5,
                    T.uint8(0),
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    + B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Add_PrimFunc_Before.add,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class Add_PrimFunc_Expected:
    @T.prim_func(private=True)
    def add(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.add", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "add",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5_0 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(2)
        ):
            for axis5_1_axis6_fused in T.vectorized(T.int64(128)):
                with T.block("compute"):
                    v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_axis1, v_axis2, v_axis3, v_axis4 = T.axis.remap(
                        "SSSS", [axis1, axis2, axis3, axis4]
                    )
                    v_axis5 = T.axis.spatial(
                        T.int64(8), axis5_0 * T.int64(4) + axis5_1_axis6_fused // T.int64(32)
                    )
                    v_axis6 = T.axis.spatial(T.int64(32), axis5_1_axis6_fused % T.int64(32))
                    T.reads(
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        + B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Add_PrimFunc_Expected.add,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class Sub_PrimFunc_Before:
    @T.prim_func(private=True)
    def sub(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.subtract", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "sub",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("compute"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )
                T.writes(compute[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                compute[
                    v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6
                ] = T.if_then_else(
                    v_axis1 == T.int64(3)
                    and T.int64(4) <= v_axis4
                    or v_axis2 == T.int64(3)
                    and T.int64(4) <= v_axis5,
                    T.uint8(0),
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    - B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Sub_PrimFunc_Before.sub,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class Sub_PrimFunc_Expected:
    @T.prim_func(private=True)
    def sub(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.subtract", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "sub",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5_0 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(2)
        ):
            for axis5_1_axis6_fused in T.vectorized(T.int64(128)):
                with T.block("compute"):
                    v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_axis1, v_axis2, v_axis3, v_axis4 = T.axis.remap(
                        "SSSS", [axis1, axis2, axis3, axis4]
                    )
                    v_axis5 = T.axis.spatial(
                        T.int64(8), axis5_0 * T.int64(4) + axis5_1_axis6_fused // T.int64(32)
                    )
                    v_axis6 = T.axis.spatial(T.int64(32), axis5_1_axis6_fused % T.int64(32))
                    T.reads(
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        - B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Sub_PrimFunc_Expected.sub,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class Mul_PrimFunc_Before:
    @T.prim_func(private=True)
    def mul(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.mul", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "mul",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("compute"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )
                T.writes(compute[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                compute[
                    v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6
                ] = T.if_then_else(
                    v_axis1 == T.int64(3)
                    and T.int64(4) <= v_axis4
                    or v_axis2 == T.int64(3)
                    and T.int64(4) <= v_axis5,
                    T.uint8(0),
                    A[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    * B[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Mul_PrimFunc_Before.mul,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class Mul_PrimFunc_Expected:
    @T.prim_func(private=True)
    def mul(
        A: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        B: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        compute: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
    ):
        T.func_attr(
            {
                "op_attrs": {"lhs_axis": 0, "op_name": "qnn.mul", "rhs_axis": 0},
                "op_pattern": 0,
                "operator_name": "mul",
                "tir.noalias": T.bool(True),
            }
        )
        # with T.block("root"):
        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_A_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5_0 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(2)
        ):
            for axis5_1_axis6_fused in T.vectorized(T.int64(128)):
                with T.block("compute"):
                    v_axis0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_axis1, v_axis2, v_axis3, v_axis4 = T.axis.remap(
                        "SSSS", [axis1, axis2, axis3, axis4]
                    )
                    v_axis5 = T.axis.spatial(
                        T.int64(8), axis5_0 * T.int64(4) + axis5_1_axis6_fused // T.int64(32)
                    )
                    v_axis6 = T.axis.spatial(T.int64(32), axis5_1_axis6_fused % T.int64(32))
                    T.reads(
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        A[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        * B[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        A: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        B: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            Mul_PrimFunc_Expected.mul,
            (A, B),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


def test_add_primfunc_overcompute():
    Add_PrimFunc_After = tvm.tir.transform.UseAssumeToReduceBranches()(Add_PrimFunc_Before)
    tvm.ir.structural_equal(
        Add_PrimFunc_After["add"], Add_PrimFunc_Expected["add"], map_free_vars=True
    )


def test_sub_primfunc_overcompute():
    Sub_PrimFunc_After = tvm.tir.transform.UseAssumeToReduceBranches()(Sub_PrimFunc_Before)
    tvm.ir.structural_equal(
        Sub_PrimFunc_After["sub"], Sub_PrimFunc_Expected["sub"], map_free_vars=True
    )


def test_mul_primfunc_overcompute():
    Mul_PrimFunc_After = tvm.tir.transform.UseAssumeToReduceBranches()(Mul_PrimFunc_Before)
    tvm.ir.structural_equal(
        Mul_PrimFunc_After["mul"], Mul_PrimFunc_Expected["mul"], map_free_vars=True
    )


if __name__ == "__main__":
    tvm.testing.main()
