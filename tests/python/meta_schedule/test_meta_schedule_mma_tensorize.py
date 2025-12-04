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
import numpy as np
from tvm.script import tir as T
from tvm.tir.schedule import Schedule
import tvm.tir.tensor_intrin  # pylint: disable=unused-import
import tvm.testing

import pytest

torch = pytest.importorskip("torch")

M, N, K = 4096, 4096, 4096
np.random.seed(0)


@tvm.script.ir_module
class Gemm_F16F16F16:
    # fmt: off
    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),  # type: ignore
        B: T.Buffer((K, N), "float16"),  # type: ignore
        C: T.Buffer((M, N), "float16"),  # type: ignore
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class Gemm_F16F16F32:
    # fmt: off
    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),  # type: ignore
        B: T.Buffer((K, N), "float16"),  # type: ignore
        C: T.Buffer((M, N), "float32"),  # type: ignore
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_run_target(mod=None, tgt_str=None, in_dtype="float16", out_dtype="float16"):
    if mod is None:
        return
    tgt_str = tgt_str or "cuda"
    target = tvm.target.Target(target=tgt_str)
    with tvm.transform.PassContext(opt_level=3):
        lib: tvm.runtime.Module = tvm.compile(mod, target=target)

    dev = tvm.device(tgt_str, 0)
    a_np = np.random.rand(M, K).astype(in_dtype)
    b_np = np.random.rand(K, N).astype(in_dtype)
    c_np = np.ones((M, N), dtype=out_dtype)
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    c = tvm.runtime.tensor(c_np, dev)

    f = lib["main"]
    f(a, b, c)

    c_th = torch.matmul(torch.tensor(a_np).to(tgt_str), torch.tensor(b_np).to(tgt_str)).to(
        torch.float32 if out_dtype == "float32" else torch.float16
    )
    c_f = torch.tensor(c.numpy()).to(tgt_str)
    torch.allclose(c_th, c_f, rtol=0.05, atol=0.05)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_f16f16f16_mma_gemm():
    # fmt: off
    mod = Gemm_F16F16F16
    sch = Schedule(mod)
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    b2 = sch.reindex(block=b0, buffer=("write", 0))
    b3 = sch.reindex(block=b0, buffer=("read", 0))
    b4 = sch.reindex(block=b0, buffer=("read", 1))
    sch.transform_layout(block=b0, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=True)
    sch.transform_layout(block=b0, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=True)
    sch.transform_layout(block=b0, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=True)
    sch.transform_block_layout(block=b2, index_map=lambda vi, vj: (vi, vj,))
    sch.transform_block_layout(block=b3, index_map=lambda vi, vk: (vi, vk,))
    sch.transform_block_layout(block=b4, index_map=lambda vj, vk: (vk, vj,))
    sch.transform_block_layout(block=b0, index_map=lambda vi, vj, vk: (vi, vj, vk,))
    l5, l6, l7 = sch.get_loops(block=b0)
    l8, l9 = sch.split(loop=l7, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l10, l11 = sch.split(loop=l6, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l12, l13 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True, disable_predication=False)
    l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
    sch.reorder(l16, l18, l13, l11, l9)
    b20 = sch.blockize(target=l13, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize", ann_val="mma_sync_m16n8k8_f16f16f16")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize_init", ann_val="mma_init_m16n8k8_f16")
    sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
    l21, l22, l23 = sch.get_loops(block=b20)
    v24, v25, v26, v27, v28 = sch.sample_partitioned_tile(loop=l21, n=5, partition_pos=3, innerpart_factor=2, decision=[2, 16, 4, 1, 2])
    l29, l30, l31, l32, l33 = sch.split(loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True, disable_predication=False)
    v34, v35, v36, v37, v38 = sch.sample_partitioned_tile(loop=l22, n=5, partition_pos=3, innerpart_factor=4, decision=[2, 16, 4, 1, 4])
    l39, l40, l41, l42, l43 = sch.split(loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True, disable_predication=False)
    v44, v45, v46 = sch.sample_perfect_tile(loop=l23, n=3, max_innermost_factor=4, decision=[128, 1, 4])
    l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], preserve_unit_iters=True, disable_predication=False)
    sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
    l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="blockIdx.y")
    l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
    sch.bind(loop=l51, thread_axis="blockIdx.x")
    l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
    sch.bind(loop=l52, thread_axis="threadIdx.y")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b53 = sch.write_at(loop=l52, block=b20, write_buffer_index=0, storage_scope="m16n8k8.matrixC")
    sch.reverse_compute_inline(block=b2)
    b54 = sch.read_at(loop=l47, block=b20, read_buffer_index=0, storage_scope="shared.dyn")
    sch.annotate(block_or_loop=b54, ann_key="permuted_layout", ann_val="g2s_A")
    b55 = sch.read_at(loop=l47, block=b20, read_buffer_index=1, storage_scope="shared.dyn")
    sch.annotate(block_or_loop=b55, ann_key="permuted_layout", ann_val="g2s_B")
    b56 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="m16n8k8.matrixA")
    sch.compute_at(block=b56, loop=l48, preserve_unit_loops=True, index=-1)
    l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b56)
    l64, l65 = sch.split(loop=l63, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l66, l67 = sch.split(loop=l62, factors=[None, 32], preserve_unit_iters=True, disable_predication=False)
    l68, l69, l70, l71, l72, l73, l74, l75, l76 = sch.get_loops(block=b56)
    sch.reorder(l75, l67, l65)
    b77 = sch.blockize(target=l67, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b77, ann_key="meta_schedule.auto_tensorize", ann_val="mma_load_m16n8k8_f16_A_shared_dyn")
    sch.annotate(block_or_loop=b77, ann_key="permuted_layout", ann_val="s2l_A")
    b78 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="m16n8k8.matrixB")
    sch.compute_at(block=b78, loop=l48, preserve_unit_loops=True, index=-1)
    l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b78)
    l86, l87 = sch.split(loop=l85, factors=[None, 32], preserve_unit_iters=True, disable_predication=False)
    l88, l89 = sch.split(loop=l84, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b78)
    sch.reorder(l97, l89, l87)
    b99 = sch.blockize(target=l89, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b99, ann_key="meta_schedule.auto_tensorize", ann_val="mma_load_m16n8k8_f16_B_shared_dyn")
    sch.annotate(block_or_loop=b99, ann_key="permuted_layout", ann_val="s2l_B")
    b100, = sch.get_producers(block=b54)
    sch.compute_inline(block=b100)
    sch.storage_align(block=b54, buffer_index=0, axis=-2, factor=32, offset=8)
    b101, = sch.get_producers(block=b55)
    sch.compute_inline(block=b101)
    sch.storage_align(block=b55, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.annotate(block_or_loop=b54, ann_key="vector_bytes", ann_val=16)
    sch.annotate(block_or_loop=b55, ann_key="vector_bytes", ann_val=16)
    sch.annotate(block_or_loop=l48, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
    sch.annotate(block_or_loop=l48, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_async_stages", ann_val=[0])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_stage", ann_val=[0, 0, 1, 2, 2])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    v102 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v102)
    sch.enter_postproc()
    b103 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b103, ann_key="meta_schedule.unroll_explicit")
    b104, b105, b106, b107, b108, b109 = sch.get_child_blocks(b103)
    l110, l111, l112, l113 = sch.get_loops(block=b104)
    l114, l115, l116, l117 = sch.get_loops(block=b105)
    l118, l119, l120, l121, l122, l123, l124 = sch.get_loops(block=b106)
    l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b107)
    l132, l133, l134, l135, l136, l137, l138, l139, l140, l141 = sch.get_loops(block=b108)
    l142, l143, l144 = sch.get_loops(block=b109)
    b145 = sch.get_block(name="C_o", func_name="main")
    l146, l147, l148, l149, l150, l151, l152, l153, l154, l155 = sch.get_loops(block=b145)
    b156 = sch.decompose_reduction(block=b145, loop=l149)
    sch.unannotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize")
    sch.annotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize", ann_val="mma_init_m16n8k8_f16")
    sch.unannotate(block_or_loop=b145, ann_key="meta_schedule.auto_tensorize_init")
    sch.unannotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize_init")
    b157 = sch.get_block(name="C_o_init", func_name="main")
    sch.unannotate(block_or_loop=b157, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b157, tensor_intrin="mma_init_m16n8k8_f16", preserve_unit_iters=True)
    b158 = sch.get_block(name="A_reindex_shared.dyn_m16n8k8.matrixA_o", func_name="main")
    sch.unannotate(block_or_loop=b158, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b158, tensor_intrin="mma_load_m16n8k8_f16_A_shared_dyn", preserve_unit_iters=True)
    b159 = sch.get_block(name="B_reindex_shared.dyn_m16n8k8.matrixB_o", func_name="main")
    sch.unannotate(block_or_loop=b159, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b159, tensor_intrin="mma_load_m16n8k8_f16_B_shared_dyn", preserve_unit_iters=True)
    b160 = sch.get_block(name="C_o_update", func_name="main")
    sch.unannotate(block_or_loop=b160, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b160, tensor_intrin="mma_sync_m16n8k8_f16f16f16", preserve_unit_iters=True)
    mod = sch.mod
    test_run_target(mod)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_f16f16f32_mma_gemm():
    mod = Gemm_F16F16F32
    sch = Schedule(mod)
    # fmt: off
    sch = Schedule(mod)
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    b2 = sch.reindex(block=b0, buffer=("write", 0))
    b3 = sch.reindex(block=b0, buffer=("read", 0))
    b4 = sch.reindex(block=b0, buffer=("read", 1))
    sch.transform_layout(block=b0, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=True)
    sch.transform_layout(block=b0, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=True)
    sch.transform_layout(block=b0, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=True)
    sch.transform_block_layout(block=b2, index_map=lambda vi, vj: (vi, vj,))
    sch.transform_block_layout(block=b3, index_map=lambda vi, vk: (vi, vk,))
    sch.transform_block_layout(block=b4, index_map=lambda vj, vk: (vk, vj,))
    sch.transform_block_layout(block=b0, index_map=lambda vi, vj, vk: (vi, vj, vk,))
    l5, l6, l7 = sch.get_loops(block=b0)
    l8, l9 = sch.split(loop=l7, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l10, l11 = sch.split(loop=l6, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l12, l13 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True, disable_predication=False)
    l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
    sch.reorder(l16, l18, l13, l11, l9)
    b20 = sch.blockize(target=l13, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize", ann_val="mma_sync_m16n8k8_f16f16f32")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize_init", ann_val="mma_init_m16n8k8_f32")
    sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
    l21, l22, l23 = sch.get_loops(block=b20)
    v24, v25, v26, v27, v28 = sch.sample_partitioned_tile(loop=l21, n=5, partition_pos=3, innerpart_factor=2, decision=[1, 16, 2, 2, 4])
    l29, l30, l31, l32, l33 = sch.split(loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True, disable_predication=False)
    v34, v35, v36, v37, v38 = sch.sample_partitioned_tile(loop=l22, n=5, partition_pos=3, innerpart_factor=4, decision=[2, 16, 2, 4, 2])
    l39, l40, l41, l42, l43 = sch.split(loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True, disable_predication=False)
    v44, v45, v46 = sch.sample_perfect_tile(loop=l23, n=3, max_innermost_factor=4, decision=[128, 1, 4])
    l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], preserve_unit_iters=True, disable_predication=False)
    sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
    l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="blockIdx.y")
    l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
    sch.bind(loop=l51, thread_axis="blockIdx.x")
    l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
    sch.bind(loop=l52, thread_axis="threadIdx.y")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b53 = sch.write_at(loop=l52, block=b20, write_buffer_index=0, storage_scope="m16n8k8.matrixC")
    sch.reverse_compute_inline(block=b2)
    b54 = sch.read_at(loop=l47, block=b20, read_buffer_index=0, storage_scope="shared.dyn")
    sch.annotate(block_or_loop=b54, ann_key="permuted_layout", ann_val="g2s_A")
    b55 = sch.read_at(loop=l47, block=b20, read_buffer_index=1, storage_scope="shared.dyn")
    sch.annotate(block_or_loop=b55, ann_key="permuted_layout", ann_val="g2s_B")
    b56 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="m16n8k8.matrixA")
    sch.compute_at(block=b56, loop=l48, preserve_unit_loops=True, index=-1)
    l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b56)
    l64, l65 = sch.split(loop=l63, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l66, l67 = sch.split(loop=l62, factors=[None, 32], preserve_unit_iters=True, disable_predication=False)
    l68, l69, l70, l71, l72, l73, l74, l75, l76 = sch.get_loops(block=b56)
    sch.reorder(l75, l67, l65)
    b77 = sch.blockize(target=l67, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b77, ann_key="meta_schedule.auto_tensorize", ann_val="mma_load_m16n8k8_f16_A_shared_dyn")
    sch.annotate(block_or_loop=b77, ann_key="permuted_layout", ann_val="s2l_A")
    b78 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="m16n8k8.matrixB")
    sch.compute_at(block=b78, loop=l48, preserve_unit_loops=True, index=-1)
    l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b78)
    l86, l87 = sch.split(loop=l85, factors=[None, 32], preserve_unit_iters=True, disable_predication=False)
    l88, l89 = sch.split(loop=l84, factors=[None, 8], preserve_unit_iters=True, disable_predication=False)
    l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b78)
    sch.reorder(l97, l89, l87)
    b99 = sch.blockize(target=l89, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b99, ann_key="meta_schedule.auto_tensorize", ann_val="mma_load_m16n8k8_f16_B_shared_dyn")
    sch.annotate(block_or_loop=b99, ann_key="permuted_layout", ann_val="s2l_B")
    b100, = sch.get_producers(block=b54)
    sch.compute_inline(block=b100)
    sch.storage_align(block=b54, buffer_index=0, axis=-2, factor=32, offset=8)
    b101, = sch.get_producers(block=b55)
    sch.compute_inline(block=b101)
    sch.storage_align(block=b55, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.annotate(block_or_loop=b54, ann_key="vector_bytes", ann_val=16)
    sch.annotate(block_or_loop=b55, ann_key="vector_bytes", ann_val=16)
    sch.annotate(block_or_loop=l48, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
    sch.annotate(block_or_loop=l48, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_async_stages", ann_val=[0])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_stage", ann_val=[0, 0, 1, 2, 2])
    sch.annotate(block_or_loop=l47, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    v102 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v102)
    sch.enter_postproc()
    b103 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b103, ann_key="meta_schedule.unroll_explicit")
    b104, b105, b106, b107, b108, b109 = sch.get_child_blocks(b103)
    l110, l111, l112, l113 = sch.get_loops(block=b104)
    l114, l115, l116, l117 = sch.get_loops(block=b105)
    l118, l119, l120, l121, l122, l123, l124 = sch.get_loops(block=b106)
    l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b107)
    l132, l133, l134, l135, l136, l137, l138, l139, l140, l141 = sch.get_loops(block=b108)
    sch.annotate(block_or_loop=l132, ann_key="pragma_auto_unroll_max_step", ann_val=0)
    sch.annotate(block_or_loop=l132, ann_key="pragma_unroll_explicit", ann_val=1)
    l142, l143, l144 = sch.get_loops(block=b109)
    b145 = sch.get_block(name="C_o", func_name="main")
    l146, l147, l148, l149, l150, l151, l152, l153, l154, l155 = sch.get_loops(block=b145)
    b156 = sch.decompose_reduction(block=b145, loop=l149)
    sch.unannotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize")
    sch.annotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize", ann_val="mma_init_m16n8k8_f32")
    sch.unannotate(block_or_loop=b145, ann_key="meta_schedule.auto_tensorize_init")
    sch.unannotate(block_or_loop=b156, ann_key="meta_schedule.auto_tensorize_init")
    b157 = sch.get_block(name="C_o_init", func_name="main")
    sch.unannotate(block_or_loop=b157, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b157, tensor_intrin="mma_init_m16n8k8_f32", preserve_unit_iters=True)
    b158 = sch.get_block(name="A_reindex_shared.dyn_m16n8k8.matrixA_o", func_name="main")
    sch.unannotate(block_or_loop=b158, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b158, tensor_intrin="mma_load_m16n8k8_f16_A_shared_dyn", preserve_unit_iters=True)
    b159 = sch.get_block(name="B_reindex_shared.dyn_m16n8k8.matrixB_o", func_name="main")
    sch.unannotate(block_or_loop=b159, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b159, tensor_intrin="mma_load_m16n8k8_f16_B_shared_dyn", preserve_unit_iters=True)
    b160 = sch.get_block(name="C_o_update", func_name="main")
    sch.unannotate(block_or_loop=b160, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b160, tensor_intrin="mma_sync_m16n8k8_f16f16f32", preserve_unit_iters=True)
    mod = sch.mod
    test_run_target(mod, out_dtype="float32")


if __name__ == """__main__""":
    test_f16f16f16_mma_gemm()
    test_f16f16f32_mma_gemm()
