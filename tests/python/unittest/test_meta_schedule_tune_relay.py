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
from multiprocessing.sharedctypes import Value
import tempfile
from typing import List
from os import path as osp
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir import IRModule
from tvm.tir.schedule.schedule import Schedule
from tvm.tir.schedule.trace import Trace
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload, JSONDatabase
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.tune import tune_relay
from tvm.meta_schedule.utils import derived_object
from tvm.target.target import Target
from tvm.script import tir as T

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
                T.reads(conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block], data_pad[n, ic // 3, oh + kh, ow + kw, ic % 3], placeholder_1[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
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


if __name__ == """__main__""":
    test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("bert_base", [1, 64], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("bert_base", [1, 64], "nvidia/geforce-rtx-3070")
    test_meta_schedule_te2primfunc_argument_order()
    test_meta_schedule_relay_lowering()
