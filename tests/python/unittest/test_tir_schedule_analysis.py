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

import tvm
from tvm.tir.tensor_intrin.x86 import dot_product_16x4_u8i8i32_desc


from tvm.tir import Evaluate, For, ForKind, IndexMap, Var, decl_buffer, floordiv, floormod, Schedule
from tvm.tir.analysis import expr_deep_equal
from tvm.tir.schedule.analysis import suggest_index_map, get_tensorize_loop_mapping
from tvm.script import tir as T
from tvm.tir.stmt_functor import pre_order_visit
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.te import create_prim_func
from tvm import relay
import numpy as np
from tvm.meta_schedule.tune import Parse


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


def _assert_equal_index_map(map1: IndexMap, map2: IndexMap) -> None:
    iters_1 = map1.map_indices(map2.initial_indices)
    iters_2 = map2.final_indices
    assert len(iters_1) == len(iters_2)
    for iter1, iter2 in zip(iters_1, iters_2):
        assert expr_deep_equal(iter1, iter2)


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
    _assert_equal_index_map(index_map, expected_index_map)


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
    _assert_equal_index_map(index_map, expected_index_map)


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
    _, _, _, _, i4, _, _, _, _, i9 = s.get_loops(block)

    assert desc_loops[0] in desc_loop_to_sref and desc_loops[1] in desc_loop_to_sref
    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i4)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i9)


def test_get_tensorize_loop_mapping_matmul_mma():
    @T.prim_func
    def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
        B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
        C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

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

    info = get_tensorize_loop_mapping(s, block, mma_desc)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(mma_desc)
    i0, i1, i2 = s.get_loops(block)

    for i in range(3):
        assert desc_loops[i] in desc_loop_to_sref

    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i0)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i1)
    assert s.get(desc_loop_to_sref[desc_loops[2]]) == s.get(i2)


def test_get_tensorize_loop_mapping_conv2d_nhwc_arm():
    @T.prim_func
    def gemm_4x4x4_i8i8i32(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 4), offset_factor=1, dtype="int8")
        B = T.match_buffer(b, (4, 4), offset_factor=1, dtype="int8")
        C = T.match_buffer(c, (4, 4), offset_factor=1, dtype="int8")

        with T.block("root"):
            T.reads(C[0:4, 0:4], A[0:4, 0:4], B[0:4, 0:4])
            T.writes(C[0:4, 0:4])
            for i, j, k in T.grid(4, 4, 4):
                with T.block("update"):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

    data_shape = (8, 64, 56, 56)
    weight_shape = (64, 64, 3, 3)

    data_dtype = "int8"
    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    out_channel = weight_shape[0]
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=weight_shape[2:],
        channels=out_channel,
        padding=(1, 1),
        strides=(1, 1),
        out_dtype=out_dtype,
    )

    relay_mod = tvm.IRModule.from_expr(conv2d)

    data = np.random.randint(low=-127, high=128, size=data_shape).astype("int8")
    weight_np = np.random.randint(low=-127, high=128, size=weight_shape).astype("int8")

    def convert_conv2d_layout(mod, desired_layouts):
        with tvm.transform.PassContext(opt_level=3):
            seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
            return seq(mod)

    relay_mod = convert_conv2d_layout(relay_mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    params = {"weight": weight_np}

    target = "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    conv2d_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    mod = Parse._mod(conv2d_tasks[0].dispatched[0])

    s = tvm.tir.Schedule(mod)

    block = s.get_block("C")

    info = get_tensorize_loop_mapping(s, block, gemm_4x4x4_i8i8i32)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(gemm_4x4x4_i8i8i32)

    for i in range(3):
        assert desc_loops[i] in desc_loop_to_sref

    _, i1_5, i2_4, i3_3 = s.get_loops(block)

    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i1_5)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i2_4)
    assert s.get(desc_loop_to_sref[desc_loops[2]]) == s.get(i3_3)


if __name__ == "__main__":
    # test_suggest_index_map_simple()
    # test_suggest_index_map_bijective()
    # test_get_tensorize_loop_mapping_dense_vnni()
    # test_get_tensorize_loop_mapping_conv2d_nchwc_vnni()
    # test_get_tensorize_loop_mapping_matmul_mma()
    test_get_tensorize_loop_mapping_conv2d_nhwc_arm()
