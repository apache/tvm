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
import logging
import tempfile
from typing import List
from os import path as osp
import numpy as np
import pytest
import tvm
from tvm import relay, tir
from tvm.contrib import graph_executor
from tvm.ir import IRModule
from tvm.tir.schedule.trace import Trace
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload, JSONDatabase
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.tune import tune_relay, tune_extracted_tasks, extract_task_from_relay, Parse
from tvm.meta_schedule.utils import derived_object
from tvm.target.target import Target
from tvm.script import tir as T
from tvm.script.registry import register
from tvm._ffi import register_func
import tempfile

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off
@tvm.script.ir_module
class tvmgen_default_fused_layout_transform:
    @T.prim_func
    def main(
        placeholder: T.Buffer[(1, 3, 16, 16), "float32"],
        T_layout_trans: T.Buffer[(1, 1, 16, 16, 3), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2, i3, i4 in T.grid(1, 1, 16, 16, 3):
            with T.block("T_layout_trans"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder[ax0, ax1 * 3 + ax4, ax2, ax3])
                T.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4])
                T_layout_trans[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(
                    ax0 < 1 and ax1 * 3 + ax4 < 3 and ax2 < 16 and ax3 < 16,
                    placeholder[ax0, ax1 * 3 + ax4, ax2, ax3],
                    T.float32(0),
                    dtype="float32",
                )


@tvm.script.ir_module
class tvmgen_default_fused_nn_contrib_conv2d_NCHWc:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 1, 16, 16, 3), "float32"], placeholder_1: T.Buffer[(2, 1, 5, 5, 3, 4), "float32"], conv2d_NCHWc: T.Buffer[(1, 2, 16, 16, 4), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        data_pad = T.alloc_buffer([1, 1, 20, 20, 3], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 1, 20, 20, 3):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder[i0_1, i1_1, i2_1 - 2, i3_1 - 2, i4_1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(2 <= i2_1 and i2_1 < 18 and 2 <= i3_1 and i3_1 < 18, placeholder[i0_1, i1_1, i2_1 - 2, i3_1 - 2, i4_1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(1, 2, 16, 16, 4, 3, 5, 5):
            with T.block("conv2d_NCHWc"):
                n, oc_chunk, oh, ow, oc_block, ic, kh, kw = T.axis.remap("SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                T.reads(data_pad[n, ic // 3, oh + kh, ow + kw, ic % 3], placeholder_1[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
                T.writes(conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block])
                T.block_attr({"workload":["conv2d_NCHWc.x86", ["TENSOR", [1, 1, 16, 16, 3], "float32"], ["TENSOR", [2, 1, 5, 5, 3, 4], "float32"], [1, 1], [2, 2, 2, 2], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                with T.init():
                    conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // 3, oh + kh, ow + kw, ic % 3] * placeholder_1[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block]

@tvm.script.ir_module
class tvmgen_default_fused_layout_transform_1:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 2, 16, 16, 4), "float32"], T_layout_trans: T.Buffer[(1, 8, 16, 16), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2, i3 in T.grid(1, 8, 16, 16):
            with T.block("T_layout_trans"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(placeholder[ax0, ax1 // 4, ax2, ax3, ax1 % 4])
                T.writes(T_layout_trans[ax0, ax1, ax2, ax3])
                T_layout_trans[ax0, ax1, ax2, ax3] = T.if_then_else(ax0 < 1 and ax1 < 8 and ax2 < 16 and ax3 < 16, placeholder[ax0, ax1 // 4, ax2, ax3, ax1 % 4], T.float32(0), dtype="float32")

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


@pytest.mark.skip("Integration test")
@pytest.mark.parametrize(
    "model_name, input_shape, target",
    [
        ("resnet_18", [1, 3, 224, 224], "llvm --num-cores=16"),
        ("resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070"),
        ("mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=16"),
        ("mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070"),
        ("bert_base", [1, 64], "llvm --num-cores=16"),
        ("bert_base", [1, 64], "nvidia/geforce-rtx-3070"),
    ],
)
def test_meta_schedule_tune_relay(
    model_name: str,
    input_shape: List[int],
    target: str,
):
    dev = tvm.cpu() if str(target).startswith("llvm") else tvm.cuda()
    if model_name.startswith("bert"):
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

    mod, params, (input_name, _, _) = get_network(name=model_name, input_shape=input_shape)
    target = Target(target)
    with tempfile.TemporaryDirectory() as work_dir:
        rt_mod1: tvm.runtime.Module = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            work_dir=work_dir,
            database=JSONDatabase(
                osp.join(work_dir, "workload.json"), osp.join(work_dir, "records.json")
            ),
        )
        # Compile without meta-scheduler for correctness check
        with tvm.transform.PassContext(opt_level=0):
            rt_mod2 = relay.build(mod, target=target, params=params)

        def get_output(data, lib):
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input(input_name, data)
            module.run()
            return module.get_output(0).numpy()

        # Check correctness
        actual_output = get_output(data, rt_mod1)
        expected_output = get_output(data, rt_mod2)
        assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


def test_meta_schedule_te2primfunc_argument_order():
    @derived_object
    class TestDummyDatabase(PyDatabase):
        def __init__(self):
            super().__init__()
            self.records = []
            self.workload_reg = []

        def has_workload(self, mod: IRModule) -> Workload:
            for workload in self.workload_reg:
                if tvm.ir.structural_equal(workload.mod, mod):
                    return True
            # The database has already put in all correct workloads
            raise ValueError(
                "The workload searched for is not in given database!"
                + " Incorrect TIR was generated from TE subgraph."
            )

        def commit_tuning_record(self, record: TuningRecord) -> None:
            self.records.append(record)

        def commit_workload(self, mod: IRModule) -> Workload:
            for workload in self.workload_reg:
                if tvm.ir.structural_equal(workload.mod, mod):
                    return workload
            workload = Workload(mod)
            self.workload_reg.append(workload)
            return workload

        def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
            return list(
                filter(
                    lambda x: x.workload == workload,
                    sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
                )
            )[: int(top_k)]

        def __len__(self) -> int:
            return len(self.records)

        def print_results(self) -> None:
            print("\n".join([str(r) for r in self.records]))

    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    data_sample = np.random.rand(*data_shape).astype("float32")
    weight_sample = np.random.rand(*weight_shape).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    input_name = "data"
    dev = tvm.cpu()
    target = Target("llvm --num-cores=16")
    data = tvm.nd.array(data_sample, dev)

    database = TestDummyDatabase()
    database.commit_workload(tvmgen_default_fused_layout_transform)
    database.commit_workload(tvmgen_default_fused_layout_transform_1)
    database.commit_workload(tvmgen_default_fused_nn_contrib_conv2d_NCHWc)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            rt_mod1 = relay.build(mod, target=target, params=params)

    # Compile without meta-scheduler for correctness check
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=target, params=params)

    def get_output(data, lib):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1)
    expected_output = get_output(data, rt_mod2)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


def test_meta_schedule_relay_lowering():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    data_sample = np.random.rand(*data_shape).astype("float32")
    weight_sample = np.random.rand(*weight_shape).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    input_name = "data"
    dev = tvm.cpu()
    target = Target("llvm --num-cores=16")
    data = tvm.nd.array(data_sample, dev)

    with tempfile.TemporaryDirectory() as work_dir:
        database = JSONDatabase(
            osp.join(work_dir, "workload.json"), osp.join(work_dir, "records.json")
        )

        database.commit_tuning_record(
            TuningRecord(
                Trace([], {}),
                [0.0],
                database.commit_workload(tvmgen_default_fused_nn_contrib_conv2d_NCHWc),
                target=target,
                args_info=[],
            )
        )

        with ApplyHistoryBest(database):
            with tvm.transform.PassContext(
                opt_level=3,
                config={"relay.backend.use_meta_schedule": True},
            ):
                rt_mod1 = relay.build(mod, target=target, params=params)

        # Compile without meta-scheduler for correctness check
        with tvm.transform.PassContext(opt_level=0):
            rt_mod2 = relay.build(mod, target=target, params=params)

        def get_output(data, lib):
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input(input_name, data)
            module.run()
            return module.get_output(0).numpy()

        # Check correctness
        actual_output = get_output(data, rt_mod1)
        expected_output = get_output(data, rt_mod2)
        assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


@register
def int32x16(imm, span):
    return imm.astype("int32x16", span)


@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8", offset_factor=1)
    B = T.match_buffer(b, (16, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (16,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])
        for i in T.serial(0, 16):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8", offset_factor=1)
    B = T.match_buffer(b, (16, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (16,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])

        A_u8x4 = A.vload([0], "uint8x4")
        A_i32 = T.reinterpret(A_u8x4, dtype="int32")

        B_i8x64 = B.vload([0, 0], dtype="int8x64")
        B_i32x16 = T.reinterpret(B_i8x64, dtype="int32x16")

        C[T.ramp(T.int32(0), 1, 16)] += T.call_llvm_pure_intrin(  # Note: this is an update +=
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
            T.uint32(0),
            T.int32x16(0),
            T.broadcast(A_i32, 16),
            B_i32x16,
            dtype="int32x16",
        )


tir.TensorIntrin.register(
    "dot_16x1x16_uint8_int8_int32_cascadelake", dot_product_desc, dot_product_intrin
)


def schedule_dense(block, M, do_tune, sch):
    post_blocks = sch.get_consumers(block)

    if len(post_blocks) > 0:
        while True:
            next_post_blocks = []
            for post_block in post_blocks:
                next_consumers = sch.get_consumers(post_block)

                if len(next_consumers) > 0:
                    sch.compute_inline(post_block)

                next_post_blocks += next_consumers

            if len(next_post_blocks) == 0:
                assert len(post_blocks) == 1
                outer_block = post_blocks[0]
                a_y, a_x = sch.get_loops(outer_block)[-2:]
                break

            post_blocks = next_post_blocks
    else:
        a_y, a_x, _ = sch.get_loops(block)[-3:]
        outer_block = block

    if do_tune:
        y_factors = sch.sample_perfect_tile(a_y, n=2, max_innermost_factor=128)
        a_yo, a_yi = sch.split(a_y, factors=y_factors)
    else:
        a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 64)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 16])
    sch.reorder(a_yo, a_xo, a_yi, a_xi)
    fused = sch.fuse(a_yo, a_xo)

    if outer_block != block:
        sch.vectorize(a_xi)
        sch.compute_at(block, a_yi)

    a_xi, a_k = sch.get_loops(block)[-2:]
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_ko, a_xi, a_ki)

    sch.parallel(fused)

    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, "dot_16x1x16_uint8_int8_int32_cascadelake")


def manual_tir_common(do_tune=False):
    M, N, K = 1024, 1024, 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype="int32")
    out = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(bias_add, 0), "uint8"),
        relay.cast(relay.expand_dims(bias_add, 0), "int8"),
        out_dtype="int32",
    )

    relay_mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake -num-cores 4"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        .evaluate()(*[data, weight_np, bias_np])
        .numpy()
    )

    params = {"weight": weight_np, "bias": bias_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: "dense" in task.task_name,
            extracted_tasks,
        )
    )

    with tempfile.TemporaryDirectory() as work_dir:
        if do_tune:
            config = ReplayTraceConfig(
                num_trials_per_iter=64,
                num_trials_total=64,
            )
            database = tune_extracted_tasks(tune_tasks, target, config, work_dir=work_dir)
        else:
            database = JSONDatabase(
                path_workload=osp.join(work_dir, "database_workload.json"),
                path_tuning_record=osp.join(work_dir, "database_tuning_record.json"),
            )

            for task in tune_tasks:
                mod = Parse._mod(task.dispatched[0])
                workload = database.commit_workload(mod)

                sch = tvm.tir.Schedule(mod)
                block = sch.get_block("compute")
                schedule_rule = sch.get(block).annotations["schedule_rule"]

                if "dense_vnni" in schedule_rule:
                    schedule_dense(block, M, False, sch)

                tune_rec = TuningRecord(sch.trace, [0.0], workload, tvm.target.Target(target), [])

                database.commit_tuning_record(tune_rec)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


@pytest.mark.skip("Requires cascadelake")
def test_tune_relay_manual_tir_vnni():
    manual_tir_common(do_tune=False)

    def schedule_rule_dense_vnni(sch, block):
        schedule_dense(block, None, True, sch)
        return [sch]

    register_func("meta_schedule.dense_vnni", schedule_rule_dense_vnni)

    manual_tir_common(do_tune=True)


if __name__ == """__main__""":
    # test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "llvm --num-cores=16")
    # test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    # test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=16")
    # test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    # test_meta_schedule_tune_relay("bert_base", [1, 64], "llvm --num-cores=16")
    # test_meta_schedule_tune_relay("bert_base", [1, 64], "nvidia/geforce-rtx-3070")
    # test_meta_schedule_te2primfunc_argument_order()
    # test_meta_schedule_relay_lowering()
    test_tune_relay_manual_tir_vnni()
