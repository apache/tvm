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
# pylint: disable=missing-docstring, unused-variable

# The test attempts to eliminate redundant pad branch and overcompute the value for elementwise ops.
# This helps to expose more opportunities to vectorize the code.

import tvm
import tvm.testing

import tvm.script
from tvm.script import tir as T, relax as R


@tvm.script.ir_module
class AddBefore:
    @T.prim_func(private=True)
    def add(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    + b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            AddBefore.add,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class AddExpected:
    @T.prim_func(private=True)
    def add(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                T.reads(b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        + b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            AddExpected.add,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class SubBefore:
    @T.prim_func(private=True)
    def sub(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    - b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            SubBefore.sub,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class SubExpected:
    @T.prim_func(private=True)
    def sub(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                T.reads(b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        - b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            SubExpected.sub,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class MulBefore:
    @T.prim_func(private=True)
    def mul(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    == T.uint8(0)
                )

        for axis0, axis1, axis2, axis3, axis4, axis5, axis6 in T.grid(
            T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)
        ):
            with T.block("buffer_B_assumptions"):
                v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6 = T.axis.remap(
                    "SSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6]
                )
                T.reads(b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    not (
                        v_axis1 == T.int64(3)
                        and T.int64(4) <= v_axis4
                        or v_axis2 == T.int64(3)
                        and T.int64(4) <= v_axis5
                    )
                    or b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
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
                    a[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    * b[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            MulBefore.mul,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


@tvm.script.ir_module
class MulExpected:
    @T.prim_func(private=True)
    def mul(
        a: T.Buffer(
            (T.int64(1), T.int64(4), T.int64(4), T.int64(16), T.int64(8), T.int64(8), T.int64(32)),
            "uint8",
        ),
        b: T.Buffer(
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
                T.reads(a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                T.reads(b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6])
                T.writes()
                T.assume(
                    (v_axis1 < T.int64(3) or v_axis4 < T.int64(4))
                    and (v_axis2 < T.int64(3) or v_axis5 < T.int64(4))
                    or b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
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
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                        b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6],
                    )
                    T.writes(
                        compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )
                    compute[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6] = (
                        a[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                        * b[T.int64(0), v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6]
                    )

    @R.function
    def main(
        a: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
        b: R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"),
    ) -> R.Tensor((1, 4, 4, 16, 8, 8, 32), "uint8"):
        out = R.call_tir(
            MulExpected.mul,
            (a, b),
            out_sinfo=R.Tensor((1, 4, 4, 16, 8, 8, 32), dtype="uint8"),
        )
        return out


def test_add_primfunc_overcompute():
    add_after = tvm.tir.transform.UseAssumeToReduceBranches()(AddBefore)
    tvm.ir.structural_equal(add_after["add"], AddExpected["add"], map_free_vars=True)


def test_sub_primfunc_overcompute():
    sub_after = tvm.tir.transform.UseAssumeToReduceBranches()(SubBefore)
    tvm.ir.structural_equal(sub_after["sub"], SubExpected["sub"], map_free_vars=True)


def test_mul_primfunc_overcompute():
    mul_after = tvm.tir.transform.UseAssumeToReduceBranches()(MulBefore)
    tvm.ir.structural_equal(mul_after["mul"], MulExpected["mul"], map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
