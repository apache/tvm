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
# pylint: disable=missing-docstring,no-member,invalid-name,unused-variable
import logging
import tempfile

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

# fmt: off
@tvm.script.ir_module
class WinogradConv2d:
    @T.prim_func
    def main(p0: T.Buffer[(2, 2048, 50, 75), "float32"], p1: T.Buffer[(4, 4, 2048, 2048), "float32"], p2: T.Buffer[(1, 2048, 1, 1), "float32"], T_relu: T.Buffer[(2, 2048, 50, 75), "float32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        data_pad = T.alloc_buffer([2, 2048, 52, 77], dtype="float32")
        input_tile = T.alloc_buffer([2048, 1900, 4, 4], dtype="float32")
        B = T.alloc_buffer([4, 4], dtype="float32")
        data_pack = T.alloc_buffer([4, 4, 2048, 1900], dtype="float32")
        bgemm = T.alloc_buffer([4, 4, 2048, 1900], dtype="float32")
        A = T.alloc_buffer([4, 2], dtype="float32")
        inverse = T.alloc_buffer([2048, 1900, 2, 2], dtype="float32")
        conv2d_winograd = T.alloc_buffer([2, 2048, 50, 75], dtype="float32")
        T_add = T.alloc_buffer([2, 2048, 50, 75], dtype="float32")
        for i0, i1, i2, i3 in T.grid(2, 2048, 52, 77):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i2_1 and i2_1 < 51 and 1 <= i3_1 and i3_1 < 76, p0[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3 in T.grid(2048, 1900, 4, 4):
            with T.block("input_tile"):
                ci, p, eps, nu = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data_pad[p // 950, ci, p % 950 // 38 * 2 + eps, p % 38 * 2 + nu])
                T.writes(input_tile[ci, p, eps, nu])
                T.block_attr({"schedule_rule":"None"})
                input_tile[ci, p, eps, nu] = data_pad[p // 950, ci, p % 950 // 38 * 2 + eps, p % 38 * 2 + nu]
        for i0, i1 in T.grid(4, 4):
            with T.block("B"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(B[i, j])
                T.block_attr({"schedule_rule":"None"})
                B[i, j] = T.Select(i % 4 == 3 and j % 4 == 3, T.float32(1), T.Select(i % 4 == 3 and j % 4 == 2, T.float32(0), T.Select(i % 4 == 3 and j % 4 == 1, T.float32(0), T.Select(i % 4 == 3 and j % 4 == 0, T.float32(0), T.Select(i % 4 == 2 and j % 4 == 3, T.float32(0), T.Select(i % 4 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 4 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 4 == 2 and j % 4 == 0, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 4 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 4 == 1 and j % 4 == 0, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 4 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 2048, 1900, 4, 4):
            with T.block("data_pack"):
                eps, nu, ci, p, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(input_tile[ci, p, r_a, r_b], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                T.writes(data_pack[eps, nu, ci, p])
                T.block_attr({"schedule_rule":"conv2d_nchw_winograd_data_pack"})
                with T.init():
                    data_pack[eps, nu, ci, p] = T.float32(0)
                data_pack[eps, nu, ci, p] = data_pack[eps, nu, ci, p] + input_tile[ci, p, r_a, r_b] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(4, 4, 2048, 1900, 2048):
            with T.block("bgemm"):
                eps, nu, co, p, ci = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                T.reads(data_pack[eps, nu, ci, p], p1[eps, nu, ci, co])
                T.writes(bgemm[eps, nu, co, p])
                with T.init():
                    bgemm[eps, nu, co, p] = T.float32(0)
                bgemm[eps, nu, co, p] = bgemm[eps, nu, co, p] + data_pack[eps, nu, ci, p] * p1[eps, nu, ci, co]
        for i0, i1 in T.grid(4, 2):
            with T.block("A"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(A[i, j])
                T.block_attr({"schedule_rule":"None"})
                A[i, j] = T.Select(i % 4 == 3 and j % 2 == 1, T.float32(1), T.Select(i % 4 == 3 and j % 2 == 0, T.float32(0), T.Select(i % 4 == 2 and j % 2 == 1, T.float32(1), T.Select(i % 4 == 2 and j % 2 == 0, T.float32(1), T.Select(i % 4 == 1 and j % 2 == 1, T.float32(-1), T.Select(i % 4 == 1 and j % 2 == 0, T.float32(1), T.Select(i % 4 == 0 and j % 2 == 1, T.float32(0), T.Select(i % 4 == 0 and j % 2 == 0, T.float32(1), T.float32(0)))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(2048, 1900, 2, 2, 4, 4):
            with T.block("inverse"):
                co, p, vh, vw, r_a, r_b = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                T.reads(bgemm[r_a, r_b, co, p], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                T.writes(inverse[co, p, vh, vw])
                T.block_attr({"schedule_rule":"conv2d_nchw_winograd_inverse"})
                with T.init():
                    inverse[co, p, vh, vw] = T.float32(0)
                inverse[co, p, vh, vw] = inverse[co, p, vh, vw] + bgemm[r_a, r_b, co, p] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("conv2d_winograd"):
                n, co, h, w = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(inverse[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2])
                T.writes(conv2d_winograd[n, co, h, w])
                conv2d_winograd[n, co, h, w] = inverse[co, n * 950 + h // 2 * 38 + w // 2, h % 2, w % 2]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[0, ax1, 0, 0])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[0, ax1, 0, 0]
        for i0, i1, i2, i3 in T.grid(2, 2048, 50, 75):
            with T.block("T_relu"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3])
                T.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add[ax0, ax1, ax2, ax3], T.float32(0))


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def two_step(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.alloc_buffer((1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j in T.grid(1024, 1024):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(1024, 1024):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 3.0
# fmt: on


@tvm.testing.requires_llvm
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=16")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


@tvm.testing.requires_cuda
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/geforce-rtx-3070")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


def test_tune_run_module_via_rpc():
    target = tvm.target.Target("llvm")
    rt_mod = tvm.build(matmul, target)

    # construct the input
    input_data = {}
    input_shape = (128, 128)
    input_dtype = "float32"
    a_np = np.random.uniform(size=input_shape).astype(input_dtype)
    b_np = np.random.uniform(size=input_shape).astype(input_dtype)
    c_np = np.zeros(input_shape).astype(input_dtype)
    for i in range(128):
        for j in range(128):
            for k in range(128):
                c_np[i, j] = c_np[i, j] + a_np[i, k] * b_np[j, k]
    input_data["a"] = a_np
    input_data["b"] = b_np
    input_data["c"] = np.zeros(input_shape).astype(input_dtype)

    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )

        def f_timer(rt_mod, dev, input_data):
            rt_mod(input_data["a"], input_data["b"], input_data["c"])
            return input_data["c"]

        result = run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=target.kind.name,
            args=input_data,
            continuation=f_timer,
        )
        tvm.testing.assert_allclose(result.numpy(), c_np, rtol=1e-3)


def test_tune_block_cpu():
    @ms.derived_object
    class RemoveBlock(ms.schedule_rule.PyScheduleRule):
        def _initialize_with_tune_context(self, context: ms.TuneContext) -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV):
            if sch.get(block).name_hint == "root":
                return [sch]
            sch = sch.copy()
            sch.compute_inline(block)
            return [sch]

        def clone(self) -> "RemoveBlock":
            return RemoveBlock()

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=16")
        database = ms.tir_integration.tune_tir(
            mod=two_step,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            space=ms.space_generator.PostOrderApply(
                f_block_filter=lambda block: block.name_hint == "A",
                sch_rules=[RemoveBlock()],
                postprocs=[],
                mutator_probs={},
            ),
        )
        sch = ms.tir_integration.compile_tir(database, two_step, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


@pytest.skip("Slow test and requires rtx-3070")
def test_tune_winograd_conv2d_cuda():
    mod = WinogradConv2d
    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.tune_tir(
            mod, target="nvidia/geforce-rtx-3070", max_trials_global=10, work_dir=work_dir
        )
        records = database.get_top_k(database.commit_workload(mod), 1)
        assert len(records) == 1, "No valid schedule found!"


if __name__ == """__main__""":
    test_tune_matmul_cpu()
    test_tune_matmul_cuda()
    test_tune_run_module_via_rpc()
    test_tune_block_cpu()
    test_tune_winograd_conv2d_cuda()
