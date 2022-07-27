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
from typing import List
import pytest
import tvm
import tvm.testing
from tvm.tir.function import TensorIntrin
from tvm.tir.tensor_intrin.x86 import dot_product_16x4_u8i8i32_desc
from tvm.tir.tensor_intrin.cuda import WMMA_SYNC_16x16x16_f16f16f32_INTRIN


from tvm.tir import Evaluate, For, ForKind, IndexMap, Var, decl_buffer, floordiv, floormod, Schedule
from tvm.tir.analysis import expr_deep_equal
from tvm.tir.schedule.analysis import (
    get_auto_tensorize_mapping_info,
    suggest_index_map,
    get_tensorize_loop_mapping,
    TensorizeInfo,
)
from tvm.script import tir as T
from tvm.tir.stmt_functor import pre_order_visit
from tvm.meta_schedule.testing import te_workload
from tvm.te import create_prim_func


def _make_vars(*args: str) -> List[Var]:
    return [Var(arg, dtype="int32") for arg in args]


def _make_loops(loop_vars: List[Var], extents: List[int]) -> List[For]:
    assert len(loop_vars) == len(extents)
    return [
        For(
            loop_var=loop_var,
            min_val=0,
            extent=extent,
            kind=ForKind.SERIAL,
            body=Evaluate(0),
        )
        for loop_var, extent in zip(loop_vars, extents)
    ]


def test_suggest_index_map_simple():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        buffer=decl_buffer(shape=[8, 256]),
        indices=[
            floordiv(i, 16) * 4 + floordiv(j, 16),
            floormod(i, 16) * 16 + floormod(j, 16),
        ],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[32, 64],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x, y: [
            floordiv(x, 4),
            floordiv(y, 16),
            floormod(x, 4),
            floormod(y, 16),
        ],
    )
    assert index_map.is_equivalent_to(expected_index_map)


def test_suggest_index_map_bijective():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        buffer=decl_buffer(shape=[8]),
        indices=[floormod(j, 4) * 2 + i],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[2, 32],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x: [
            floormod(x, 2),
            floordiv(x, 2),
        ],
    )
    assert index_map.is_equivalent_to(expected_index_map)


@tvm.script.ir_module
class DenseVNNIModule:
    @T.prim_func
    def main(
        placeholder: T.Buffer[(1024, 1024), "uint8"],
        placeholder_1: T.Buffer[(64, 256, 16, 4), "int8"],
        compute: T.Buffer[(1024, 1024), "int32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        with T.block("root"):
            T.reads()
            T.writes()
            for i0, i1, i2 in T.grid(1024, 1024, 1024):
                with T.block("compute"):
                    i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                    T.reads(placeholder[i, k], placeholder_1[j // 16, k // 4, j % 16, k % 4])
                    T.writes(compute[i, j])
                    with T.init():
                        compute[i, j] = 0
                    compute[i, j] = compute[i, j] + T.cast(placeholder[i, k], "int32") * T.cast(
                        placeholder_1[j // 16, k // 4, j % 16, k % 4], "int32"
                    )


@tvm.script.ir_module
class Conv2dNCHWcVNNIModule:
    @T.prim_func
    def main(
        placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"],
        placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"],
        conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                (
                    n,
                    oc_chunk,
                    oh,
                    ow,
                    oc_block,
                    kh,
                    kw,
                    ic_outer,
                    ic_f_inner,
                    ic_s_inner,
                ) = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                )
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                    n, oc_chunk, oh, ow, oc_block
                ] + T.cast(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32"
                ) * T.cast(
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                    "int32",
                )


def collect_loops(prim_func):
    loops = []

    def callback(node):
        if isinstance(node, tvm.tir.For):
            loops.append(node)
        return True

    pre_order_visit(prim_func.body, callback)

    return loops


def test_get_tensorize_loop_mapping_dense_vnni():
    s = Schedule(DenseVNNIModule)
    block = s.get_block("compute")

    info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)

    assert isinstance(info, TensorizeInfo)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(dot_product_16x4_u8i8i32_desc)
    _, loop_j, loop_k = s.get_loops(block)

    assert desc_loops[0] in desc_loop_to_sref and desc_loops[1] in desc_loop_to_sref
    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(loop_j)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(loop_k)


def test_get_tensorize_loop_mapping_conv2d_nchwc_vnni():
    s = Schedule(Conv2dNCHWcVNNIModule)
    block = s.get_block("conv2d_NCHWc_int8")

    info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(dot_product_16x4_u8i8i32_desc)

    # i4 corresonds to the inner output channel axis of the NCHWc output tensor
    # for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
    _, _, _, _, i4, _, _, _, _, i9 = s.get_loops(block)

    assert desc_loops[0] in desc_loop_to_sref and desc_loops[1] in desc_loop_to_sref
    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i4)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i9)


def test_get_tensorize_loop_mapping_matmul_mma():
    @T.prim_func
    def matmul_16x16x16xf16f16f16_desc(
        A: T.Buffer((16, 16), "float16", align=128, offset_factor=1),
        B: T.Buffer((16, 16), "float16", align=128, offset_factor=1),
        C: T.Buffer((16, 16), "float16", align=128, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j, k in T.grid(16, 16, 16):
                with T.block("update"):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

    matmul = create_prim_func(
        te_workload.matmul_relu(
            n=512,
            m=512,
            k=512,
        )
    )

    s = Schedule(matmul)
    block = s.get_block("C")
    i0, i1, i2 = s.get_loops(block)
    desc_loops = collect_loops(matmul_16x16x16xf16f16f16_desc)

    for do_reorder in [False, True]:
        # Mapping should be invariant to the loop permutation
        if do_reorder:
            s.reorder(i2, i0, i1)

        info = get_tensorize_loop_mapping(s, block, matmul_16x16x16xf16f16f16_desc)
        assert info is not None
        desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

        for i in range(3):
            assert desc_loops[i] in desc_loop_to_sref

        assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i0)
        assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i1)
        assert s.get(desc_loop_to_sref[desc_loops[2]]) == s.get(i2)


def check_index_map(workload, block_name, intrin_name, expected_index_map):
    s = Schedule(workload)
    block = s.get_block(block_name)
    desc_func = TensorIntrin.get(intrin_name).desc
    info = get_auto_tensorize_mapping_info(s, block, desc_func)
    assert len(info.mappings) == 1
    assert IndexMap.from_func(expected_index_map).is_equivalent_to(info.mappings[0])


def test_get_auto_tensorize_mapping_info_conv2d():
    conv2d = create_prim_func(
        te_workload.conv2d_nhwc(4, 16, 16, 64, 64, 3, 1, 1, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(
        conv2d,
        "conv2d_nhwc",
        WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
        lambda n, h, w, c, rh, rw, rc: (n * 256 + h * 16 + w, c, rh * 192 + rw * 64 + rc),
    )


def test_get_auto_tensorize_mapping_info_conv2d_unit_batch():
    conv2d = create_prim_func(
        te_workload.conv2d_nhwc(1, 16, 16, 64, 64, 3, 1, 1, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(
        conv2d,
        "conv2d_nhwc",
        WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
        # unit iter is not mapped
        lambda n, h, w, c, rh, rw, rc: (n, h * 16 + w, c, rh * 192 + rw * 64 + rc),
    )


@pytest.mark.parametrize("b,m,n,k", [(1, 512, 512, 512), (16, 32, 32, 32)])
def test_get_auto_tensorize_mapping_info_batch_matmul(b, m, n, k):
    matmul = create_prim_func(
        te_workload.batch_matmul_nkkm(b, m, n, k, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(
        matmul, "Z", WMMA_SYNC_16x16x16_f16f16f32_INTRIN, lambda b, m, n, k: (b, m, n, k)
    )


if __name__ == "__main__":
    tvm.testing.main()
