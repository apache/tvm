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
# pylint: disable=missing-docstring

from tvm.script import tir as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,no-self-use,unused-argument,chained-comparison,misplaced-comparison-constant


@T.prim_func
def conv2d_winograd_cpu(
    X: T.Buffer[(1, 14, 14, 128), "float32"],  # type: ignore
    W: T.Buffer[(6, 6, 128, 128), "float32"],  # type: ignore
    conv2d_winograd: T.Buffer[(1, 12, 12, 128), "float32"],  # type: ignore
) -> None:
    # body
    data_pad = T.alloc_buffer([1, 16, 16, 128])
    input_tile = T.alloc_buffer([6, 6, 9, 128])
    B = T.alloc_buffer([6, 6])
    data_pack = T.alloc_buffer([6, 6, 9, 128])
    bgemm = T.alloc_buffer([6, 6, 9, 128])
    A = T.alloc_buffer([6, 4])
    inverse = T.alloc_buffer([4, 4, 9, 128])
    for i0, i1, i2, i3 in T.grid(1, 16, 16, 128):
        with T.block("data_pad"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.block_attr({"schedule_rule": "None"})
            T.reads([X[i0_1, i1_1, i2_1, i3_1]])
            T.writes([data_pad[i0_1, i1_1, i2_1, i3_1]])
            data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                0 <= i1_1 and i1_1 < 14 and 0 <= i2_1 and i2_1 < 14,  # type: ignore
                X[i0_1, i1_1, i2_1, i3_1],
                T.float32(0),
                dtype="float32",
            )
    for i0_2, i1_2, i2_2, i3_2 in T.grid(6, 6, 9, 128):
        with T.block("input_tile"):
            eps, nu, p, ci = T.axis.remap("SSSS", [i0_2, i1_2, i2_2, i3_2])
            T.block_attr({"schedule_rule": "None"})
            T.reads(
                data_pad[
                    T.floordiv(p, 9),  # type: ignore
                    ((T.floordiv(T.floormod(p, 9), 3) * 4) + eps),  # type: ignore
                    ((T.floormod(p, 3) * 4) + nu),  # type: ignore
                    ci,
                ]
            )
            T.writes([input_tile[eps, nu, p, ci]])
            input_tile[eps, nu, p, ci] = data_pad[
                T.floordiv(p, 9),  # type: ignore
                ((T.floordiv(T.floormod(p, 9), 3) * 4) + eps),  # type: ignore
                ((T.floormod(p, 3) * 4) + nu),  # type: ignore
                ci,
            ]
    for i0_3, i1_3 in T.grid(6, 6):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0_3, i1_3])
            T.block_attr({"schedule_rule": "meta_schedule.compute_inline"})
            T.writes([B[i, j]])
            # fmt: off
            B[i, j] = T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 0)), T.float32(0), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 5)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 4)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 3)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 2)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 0)), T.float32(1), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 5)), T.float32(-2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 4)), T.float32(-0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 3)), T.float32(2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 2)), T.float32(2.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 1)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 0)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 5)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 4)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 3)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 2)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 1)), T.float32(-2.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 0)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 4)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 3)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 2)), T.float32(-1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 0)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 5)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))  # type: ignore
            # fmt: on
    for i0_4, i1_4, i2_3, i3_3, i4, i5 in T.grid(6, 6, 9, 128, 6, 6):
        with T.block("data_pack"):
            eps_1, nu_1, p_1, ci_1, r_a, r_b = T.axis.remap(
                "SSSSRR", [i0_4, i1_4, i2_3, i3_3, i4, i5]
            )
            T.block_attr({"schedule_rule": "meta_schedule.winograd_data_pack.llvm"})
            T.reads(
                [
                    data_pack[eps_1, nu_1, p_1, ci_1],
                    input_tile[r_a, r_b, p_1, ci_1],
                    B[
                        T.min(r_a, r_b) : (  # type: ignore
                            T.min(r_a, r_b) + ((T.max(r_a, r_b) + 1) - T.min(r_a, r_b))  # type: ignore
                        ),
                        T.min(eps_1, nu_1) : (  # type: ignore
                            T.min(eps_1, nu_1) + ((T.max(eps_1, nu_1) + 1) - T.min(eps_1, nu_1))  # type: ignore
                        ),
                    ],
                ]
            )
            T.writes([data_pack[eps_1, nu_1, p_1, ci_1]])
            with T.init():
                data_pack[eps_1, nu_1, p_1, ci_1] = T.float32(0)
            data_pack[eps_1, nu_1, p_1, ci_1] = data_pack[eps_1, nu_1, p_1, ci_1] + (
                (input_tile[r_a, r_b, p_1, ci_1] * B[r_a, eps_1]) * B[r_b, nu_1]
            )
    for i0_5, i1_5, i2_4, i3_4, i4_1 in T.grid(6, 6, 9, 128, 128):
        with T.block("bgemm"):
            eps_2, nu_2, p_2, co, ci_2 = T.axis.remap("SSSSR", [i0_5, i1_5, i2_4, i3_4, i4_1])
            T.block_attr({"meta_schedule.write_cache_level": [2]})
            T.reads(
                [
                    bgemm[eps_2, nu_2, p_2, co],
                    data_pack[eps_2, nu_2, p_2, ci_2],
                    W[eps_2, nu_2, co, ci_2],
                ]
            )
            T.writes([bgemm[eps_2, nu_2, p_2, co]])
            with T.init():
                bgemm[eps_2, nu_2, p_2, co] = T.float32(0)
            bgemm[eps_2, nu_2, p_2, co] = (
                bgemm[eps_2, nu_2, p_2, co]
                + data_pack[eps_2, nu_2, p_2, ci_2] * W[eps_2, nu_2, co, ci_2]
            )
    for i0_6, i1_6 in T.grid(6, 4):
        with T.block("A"):
            i_1, j_1 = T.axis.remap("SS", [i0_6, i1_6])
            T.block_attr({"schedule_rule": "meta_schedule.compute_inline"})
            T.writes([A[i_1, j_1]])
            # fmt: off
            A[i_1, j_1] = T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 0)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 3)), T.float32(-8), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 2)), T.float32(4), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 1)), T.float32(-2), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 3)), T.float32(0.125), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 2)), T.float32(0.25), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 1)), T.float32(0.5), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 1)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 3)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 1)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 3)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))  # type: ignore
            # fmt: on
    for i0_7, i1_7, i2_5, i3_5, i4_2, i5_1 in T.grid(4, 4, 9, 128, 6, 6):
        with T.block("inverse"):
            vh, vw, p_3, co_1, r_a_1, r_b_1 = T.axis.remap(
                "SSSSRR", [i0_7, i1_7, i2_5, i3_5, i4_2, i5_1]
            )
            T.block_attr({"schedule_rule": "meta_schedule.winograd_inverse.llvm"})
            T.reads(
                [
                    inverse[vh, vw, p_3, co_1],
                    bgemm[r_a_1, r_b_1, p_3, co_1],
                    A[
                        T.min(r_a_1, r_b_1) : (  # type: ignore
                            T.min(r_a_1, r_b_1) + ((T.max(r_a_1, r_b_1) + 1) - T.min(r_a_1, r_b_1))  # type: ignore
                        ),
                        T.min(vh, vw) : (T.min(vh, vw) + ((T.max(vh, vw) + 1) - T.min(vh, vw))),  # type: ignore
                    ],
                ]
            )
            T.writes([inverse[vh, vw, p_3, co_1]])
            with T.init():
                inverse[vh, vw, p_3, co_1] = T.float32(0)
            inverse[vh, vw, p_3, co_1] = inverse[vh, vw, p_3, co_1] + (
                (bgemm[r_a_1, r_b_1, p_3, co_1] * A[r_a_1, vh]) * A[r_b_1, vw]
            )
    for i0_8, i1_8, i2_6, i3_6 in T.grid(1, 12, 12, 128):
        with T.block("conv2d_winograd"):
            n, h, w, co_2 = T.axis.remap("SSSS", [i0_8, i1_8, i2_6, i3_6])
            T.reads(
                [
                    inverse[
                        T.floormod(h, 4),  # type: ignore
                        T.floormod(w, 4),  # type: ignore
                        (((n * 9) + (T.floordiv(h, 4) * 3)) + T.floordiv(w, 4)),  # type: ignore
                        co_2,
                    ]
                ]
            )
            T.writes([conv2d_winograd[n, h, w, co_2]])
            conv2d_winograd[n, h, w, co_2] = inverse[
                T.floormod(h, 4),  # type: ignore
                T.floormod(w, 4),  # type: ignore
                (((n * 9) + (T.floordiv(h, 4) * 3)) + T.floordiv(w, 4)),  # type: ignore
                co_2,
            ]
