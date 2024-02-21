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
import enum

import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import tir as T


class OpPatternKind(enum.IntEnum):
    kElemWise = 0
    kBroadcast = 1
    kInjective = 2
    kCommReduce = 3
    kOutEWiseFusable = 4
    kTuple = 7
    kOpaque = 8


def test_annotate_opkind_outewisefusable():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.int32()
            n = T.int32()
            k = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


@pytest.mark.parametrize(
    "cast_pattern",
    [
        lambda a, b: T.Cast("float32", a * b),
        lambda a, b: T.Cast("float32", a) * T.Cast("float32", b),
        lambda a, b: T.Cast("float32", T.Cast("float16", a * b)),
    ],
)
def test_annotate_opkind_outewisefusable_with_cast(cast_pattern):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.int32()
            n = T.int32()
            k = T.int32()
            A = T.match_buffer(x, (m, n), "float16")
            B = T.match_buffer(y, (n, k), "float16")
            C = T.match_buffer(z, (m, k), "float32")

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + cast_pattern(A[vi, vk], B[vk, vj])

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_outewisefusable_int_var_signature():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle, m: T.int64, n: T.int64, k: T.int64):
            T.func_attr({"global_symbol": "tir_matmul"})
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_reduce():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def sum(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16,))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SR", [i, j])
                    with T.init():
                        B[vi] = 0.0
                    B[vi] += A[vi, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["sum"].attrs["op_pattern"] == OpPatternKind.kCommReduce


def test_annotate_opkind_ewise():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def elemwise(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 1.0

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["elemwise"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_broadcast():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def broadcast(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16, 16, 16))

            for i0, j0, i1, j1 in T.grid(16, 16, 16, 16):
                with T.block("matmul"):
                    vi0, vj0, vi1, vj1 = T.axis.remap("SSSS", [i0, j0, i1, j1])
                    B[vi0, vj0, vi1, vj1] = A[vj0, vj1]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["broadcast"].attrs["op_pattern"] == OpPatternKind.kBroadcast


def test_annotate_opkind_injective():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def injective(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (4, 4, 4, 4))
            B = T.match_buffer(y, (16, 16))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi // 4, vj // 4, vi % 4, vj % 4]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["injective"].attrs["op_pattern"] == OpPatternKind.kInjective


def test_annotate_opkind_bias_add():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_bias_add(
            A: T.Buffer((1, 1000), "float32"),
            B: T.Buffer((1000,), "float32"),
            C: T.Buffer((1, 1000), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "tir_bias_add", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0, i1 in T.grid(1, 1000):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[ax0, ax1], B[ax1])
                    T.writes(C[ax0, ax1])
                    C[ax0, ax1] = A[ax0, ax1] + B[ax1]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_bias_add"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_add_broadcast_with_unit_shape():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def add_with_unit_dim_len_broadcast(
            A: T.Buffer((1, 64, 112, 112), "float32"),
            B: T.Buffer((64, 1, 1), "float32"),
            C: T.Buffer((1, 64, 112, 112), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add5", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(1, 64, 112, 112):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(A[ax0, ax1, ax2, ax3], B[ax1, 0, 0])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    C[ax0, ax1, ax2, ax3] = A[ax0, ax1, ax2, ax3] + B[ax1, 0, 0]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["add_with_unit_dim_len_broadcast"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_add_zero_dim_element_wise():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def add_zero_dim(
            A: T.Buffer((128,), "float32"),
            B: T.Buffer((), "float32"),
            C: T.Buffer((128,), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add8", "tir.noalias": True})
            for i0 in T.serial(128):
                with T.block("T_add"):
                    ax0 = T.axis.spatial(128, i0)
                    T.reads(A[ax0], B[()])
                    T.writes(C[ax0])
                    C[ax0] = A[ax0] + B[()]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["add_zero_dim"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_pooling():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def max_pool2d(
            rxplaceholder_1: T.Buffer((1, 64, 112, 112), "float32"),
            tensor_1: T.Buffer((1, 64, 56, 56), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "max_pool2d", "T.noalias": True})
            # body
            # with T.block("root")
            pad_temp_1 = T.alloc_buffer([1, 64, 114, 114], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 64, 114, 114):
                with T.block("pad_temp"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1])
                    T.writes(pad_temp_1[ax0, ax1, ax2, ax3])
                    pad_temp_1[ax0, ax1, ax2, ax3] = T.if_then_else(
                        1 <= ax2 and ax2 < 113 and 1 <= ax3 and ax3 < 113,
                        rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1],
                        T.float32(-3.4028234663852886e38),
                        dtype="float32",
                    )
            for i0, i1, i2, i3, i4, i5 in T.grid(1, 64, 56, 56, 3, 3):
                with T.block("tensor"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(
                        tensor_1[ax0, ax1, ax2, ax3],
                        pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1],
                    )
                    T.writes(tensor_1[ax0, ax1, ax2, ax3])
                    with T.init():
                        tensor_1[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e38)
                    tensor_1[ax0, ax1, ax2, ax3] = T.max(
                        tensor_1[ax0, ax1, ax2, ax3],
                        pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1],
                    )

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["max_pool2d"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_softmax():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def softmax(
            rxplaceholder_1: T.Buffer((16, 16), "float32"),
            T_softmax_norm_1: T.Buffer((16, 16), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "softmax", "T.noalias": True})
            # body
            # with T.block("root")
            T_softmax_maxelem_1 = T.alloc_buffer([16], dtype="float32")
            T_softmax_exp_1 = T.alloc_buffer([16, 16], dtype="float32")
            T_softmax_expsum_1 = T.alloc_buffer([16], dtype="float32")
            for i0_7, i1_3 in T.grid(16, 16):
                with T.block("T_softmax_maxelem"):
                    i0_8, k = T.axis.remap("SR", [i0_7, i1_3])
                    T.reads(T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k])
                    T.writes(T_softmax_maxelem_1[i0_8])
                    with T.init():
                        T_softmax_maxelem_1[i0_8] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem_1[i0_8] = T.max(
                        T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k]
                    )
            for i0_9, i1_4 in T.grid(16, 16):
                with T.block("T_softmax_exp"):
                    i0_10, i1_5 = T.axis.remap("SS", [i0_9, i1_4])
                    T.reads(rxplaceholder_1[i0_10, i1_5], T_softmax_maxelem_1[i0_10])
                    T.writes(T_softmax_exp_1[i0_10, i1_5])
                    T_softmax_exp_1[i0_10, i1_5] = T.exp(
                        rxplaceholder_1[i0_10, i1_5] - T_softmax_maxelem_1[i0_10], dtype="float32"
                    )
            for i0_11, i1_6 in T.grid(16, 16):
                with T.block("T_softmax_expsum"):
                    i0_12, k = T.axis.remap("SR", [i0_11, i1_6])
                    T.reads(T_softmax_expsum_1[i0_12], T_softmax_exp_1[i0_12, k])
                    T.writes(T_softmax_expsum_1[i0_12])
                    with T.init():
                        T_softmax_expsum_1[i0_12] = T.float32(0)
                    T_softmax_expsum_1[i0_12] = (
                        T_softmax_expsum_1[i0_12] + T_softmax_exp_1[i0_12, k]
                    )
            for i0_13, i1_7 in T.grid(16, 16):
                with T.block("T_softmax_norm"):
                    i0_14, i1_8 = T.axis.remap("SS", [i0_13, i1_7])
                    T.reads(T_softmax_exp_1[i0_14, i1_8], T_softmax_expsum_1[i0_14])
                    T.writes(T_softmax_norm_1[i0_14, i1_8])
                    T.block_attr({"axis": 1})
                    T_softmax_norm_1[i0_14, i1_8] = (
                        T_softmax_exp_1[i0_14, i1_8] / T_softmax_expsum_1[i0_14]
                    )

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["softmax"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_multiple_bufer_stores_fallback():
    @tvm.script.ir_module
    class CumsumModule:
        @T.prim_func
        def cumsum(var_rxplaceholder: T.handle, out_buf: T.Buffer(160, "float32")):
            rxplaceholder = T.match_buffer(
                var_rxplaceholder, [10, 16], dtype="float32", offset_factor=1
            )
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[0:10, 0:16])
                T.writes(out_buf[0:160])
                for fused in T.parallel(1):
                    out_buf[fused * 160] = rxplaceholder[fused * 160 // 16, fused * 160 % 16]
                    for v_k in T.serial(159):
                        out_buf[fused * 160 + (v_k + 1)] = (
                            out_buf[fused * 160 + (v_k + 1 - 1)]
                            + rxplaceholder[
                                (fused * 160 + (v_k + 1)) // 16,
                                (fused * 160 + (v_k + 1)) % 16,
                            ]
                        )

    mod = CumsumModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["cumsum"].attrs["op_pattern"] == OpPatternKind.kOpaque


def test_sum_sqsum():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def sum_sqsum(
            A: T.Buffer((32, 64), "float32"),
            vsum: T.Buffer((32,), "float32"),
            sqsum: T.Buffer((32,), "float32"),
        ):
            for ax0, k0 in T.grid(32, 64):
                with T.block("block"):
                    v_ax0, v_k0 = T.axis.remap("SR", [ax0, k0])
                    T.reads(A[v_ax0, v_k0])
                    T.writes(vsum[v_ax0], sqsum[v_ax0])
                    with T.init():
                        vsum[v_ax0] = T.float32(0)
                        sqsum[v_ax0] = T.float32(0)
                    v_vsum: T.float32 = vsum[v_ax0] + A[v_ax0, v_k0]
                    v_sqsum: T.float32 = sqsum[v_ax0] + A[v_ax0, v_k0] * A[v_ax0, v_k0]
                    vsum[v_ax0] = v_vsum
                    sqsum[v_ax0] = v_sqsum

    mod = Module
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["sum_sqsum"].attrs["op_pattern"] == OpPatternKind.kCommReduce


def test_no_buffer_stores():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def no_buffer_stores(A: T.Buffer((32, 64), "float32"), vsum: T.Buffer((32,), "float32")):
            for ax0, k0 in T.grid(32, 64):
                with T.block("block"):
                    v_ax0, v_k0 = T.axis.remap("SR", [ax0, k0])
                    T.reads(A[v_ax0, v_k0])
                    T.writes(vsum[v_ax0])
                    # absence of buffer stores usually happens when there is an external call for
                    # computation. We assume opaque in all such cases.
                    T.call_packed("some_func")

    mod = Module
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["no_buffer_stores"].attrs["op_pattern"] == OpPatternKind.kOpaque


if __name__ == "__main__":
    tvm.testing.main()
