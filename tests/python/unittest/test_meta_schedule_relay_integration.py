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
"""Integration test for MetaSchedule"""
import tempfile
from typing import List

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm import relay, te, tir
from tvm._ffi import register_func
from tvm.contrib import graph_executor
from tvm.ir.transform import PassContext
from tvm.meta_schedule.database import TuningRecord, Workload
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.meta_schedule.tune_context import _normalize_mod
from tvm.script import tir as T
from tvm.target import Target

# pylint: disable=no-member,line-too-long,too-many-nested-blocks,unbalanced-tuple-unpacking,no-self-argument,missing-docstring,invalid-name


@tvm.script.ir_module
class MockModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:  # type: ignore
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        for i in T.serial(0, 16):
            with T.block("matmul"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]


# pylint: enable=no-member,line-too-long,too-many-nested-blocks,unbalanced-tuple-unpacking,no-self-argument


def _has_torch():
    import importlib.util  # pylint: disable=unused-import,import-outside-toplevel

    spec = importlib.util.find_spec("torch")
    return spec is not None


requires_torch = pytest.mark.skipif(not _has_torch(), reason="torch is not installed")


def test_meta_schedule_dynamic_loop_extent():
    a = relay.var("a", shape=(1, 8, 8, 512), dtype="float32")
    b = relay.nn.adaptive_avg_pool2d(a, (7, 7), "NHWC")
    mod = IRModule({"main": relay.Function([a], b)})
    extracted_tasks = ms.relay_integration.extract_tasks(mod, target="llvm", params={})
    assert not extracted_tasks


@requires_torch
def test_meta_schedule_integration_extract_from_resnet():
    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    extracted_tasks = ms.relay_integration.extract_tasks(mod, target="llvm", params=params)
    expected_task_names = [
        "fused_" + s
        for s in [
            "nn_max_pool2d",
            "nn_adaptive_avg_pool2d",
            "nn_dense_add",
            "nn_conv2d_add",
            "nn_conv2d_add_1",
            "nn_conv2d_add_2",
            "nn_conv2d_add_add_nn_relu",
            "nn_conv2d_add_add_nn_relu_1",
            "nn_conv2d_add_nn_relu",
            "nn_conv2d_add_nn_relu_1",
            "nn_conv2d_add_nn_relu_2",
            "nn_conv2d_add_nn_relu_3",
            "nn_conv2d_add_nn_relu_4",
            "nn_conv2d_add_nn_relu_5",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1",
            # The two tasks below are purely spatial and are ruled out by AutoScheduler
            "layout_transform",
            "layout_transform_reshape_squeeze",
        ]
    ]

    assert len(extracted_tasks) == len(expected_task_names)
    for t in extracted_tasks:
        assert t.task_name in expected_task_names, t.task_name


@requires_torch
def test_task_extraction_anchor_block():
    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    extracted_tasks = ms.relay_integration.extract_tasks(
        mod, target="llvm", params=params, module_equality="anchor-block"
    )

    # Note that there is no task from residual blocks
    expected_task_names = [
        "fused_" + s
        for s in [
            "nn_max_pool2d",
            "nn_adaptive_avg_pool2d",
            "nn_dense_add",
            "nn_conv2d_add",
            "nn_conv2d_add_1",
            "nn_conv2d_add_2",
            "nn_conv2d_add_nn_relu",
            "nn_conv2d_add_nn_relu_1",
            "nn_conv2d_add_nn_relu_2",
            "nn_conv2d_add_nn_relu_3",
            "nn_conv2d_add_nn_relu_4",
            "nn_conv2d_add_nn_relu_5",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1",
            "layout_transform",
            "layout_transform_reshape_squeeze",
        ]
    ]

    assert len(extracted_tasks) == len(expected_task_names)
    for t in extracted_tasks:
        assert t.task_name in expected_task_names, t.task_name


@requires_torch
def test_meta_schedule_integration_extract_from_bert_base():
    pytest.importorskip(
        "transformers", reason="transformers package is required to import bert_base"
    )

    expected = {
        "fused_nn_dense_2": (
            12,
            [[64, 3072], [768, 3072], [64, 768]],
        ),
        "fused_nn_dense": (
            48,
            [[64, 768], [768, 768], [64, 768]],
        ),
        "fused_nn_dense_1": (
            12,
            [[64, 768], [3072, 768], [64, 3072]],
        ),
        "fused_subtract_add_rsqrt_multiply_multiply_add": (
            25,
            [[1, 64, 768], [1, 64, 1], [1, 64, 1], [768], [768], [1, 64, 768]],
        ),
        "fused_nn_batch_matmul": (
            24,
            [[12, 64, 64], [12, 64, 64], [12, 64, 64]],
        ),
        "fused_reshape_add_add": (
            24,
            [[64, 768], [768], [1, 64, 768], [1, 64, 768]],
        ),
        "fused_variance": (
            25,
            [[1, 64, 768], [1, 64, 1], [1, 64, 1]],
        ),
        "fused_mean": (
            25,
            [[1, 64, 768], [1, 64, 1]],
        ),
        "fused_reshape_add_reshape_transpose_reshape": (
            12,
            [[64, 768], [768], [12, 64, 64]],
        ),
        "fused_reshape_add_multiply_fast_erf_multiply_add_multiply_reshape": (
            12,
            [[64, 3072], [3072], [64, 3072]],
        ),
        "fused_nn_fast_softmax": (
            12,
            [[1, 12, 64, 64], [1, 12, 64, 64]],
        ),
        "fused_reshape_add_reshape_transpose_reshape_1": (
            24,
            [[64, 768], [768], [12, 64, 64]],
        ),
        "fused_reshape_divide_add": (
            12,
            [[12, 64, 64], [1, 1, 1, 64], [1, 12, 64, 64]],
        ),
        "fused_reshape_transpose_reshape": (
            12,
            [[12, 64, 64], [64, 768]],
        ),
        "fused_nn_dense_add_fast_tanh": (
            1,
            [[1, 768], [768, 768], [1, 768], [1, 768]],
        ),
        "fused_cast_take_add": (
            1,
            [[1, 64], [30522, 768], [1, 64, 768], [1, 64, 768]],
        ),
        "fused_take": (
            1,
            [[1, 64, 768], [1, 768]],
        ),
        "fused_reshape": (
            12,
            [[1, 12, 64, 64], [12, 64, 64]],
        ),
        "fused_reshape_1": (
            24,
            [[1, 64, 768], [64, 768]],
        ),
    }
    mod, params, _ = get_network(name="bert_base", input_shape=[1, 64])
    extracted_tasks = ms.relay_integration.extract_tasks(mod, target="llvm", params=params)
    assert len(extracted_tasks) == len(expected)
    for t in extracted_tasks:
        prim_func = None
        for _, v in t.dispatched[0].functions.items():
            prim_func = v
        shape = [[int(x) for x in prim_func.buffer_map[b].shape] for b in prim_func.params]
        assert t.task_name in expected
        expected_weight, expected_shape = expected[t.task_name]
        assert expected_weight == t.weight, t.task_name
        assert expected_shape == shape, t.task_name


@requires_torch
def test_meta_schedule_integration_extract_from_resnet_with_filter_func():
    @register_func("relay.backend.tir_converter.remove_purely_spatial", override=True)
    def filter_func(args, _) -> bool:
        from tvm.te import create_prim_func  # pylint: disable=import-outside-toplevel

        has_complex_op = False
        visited = set()

        def traverse(t):
            nonlocal has_complex_op
            assert t.handle is not None
            if t.handle.value in visited:
                return
            if isinstance(t.op, te.PlaceholderOp):
                pass
            elif isinstance(t.op, te.ComputeOp):
                has_complex_op = has_complex_op or any(isinstance(e, tir.Reduce) for e in t.op.body)
                for x in t.op.input_tensors:
                    traverse(x)
            visited.add(t.handle.value)

        for t in args:
            traverse(t)
        if not has_complex_op:
            return None
        return create_prim_func(args)

    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    extracted_tasks = ms.relay_integration.extract_tasks(
        mod,
        target="llvm",
        params=params,
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "remove_purely_spatial",
        },
    )
    expected_task_names = [
        "fused_" + s
        for s in [
            "nn_max_pool2d",
            "nn_adaptive_avg_pool2d",
            "nn_dense_add",
            "nn_conv2d_add",
            "nn_conv2d_add_1",
            "nn_conv2d_add_2",
            "nn_conv2d_add_add_nn_relu",
            "nn_conv2d_add_add_nn_relu_1",
            "nn_conv2d_add_nn_relu",
            "nn_conv2d_add_nn_relu_1",
            "nn_conv2d_add_nn_relu_2",
            "nn_conv2d_add_nn_relu_3",
            "nn_conv2d_add_nn_relu_4",
            "nn_conv2d_add_nn_relu_5",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1",
        ]
    ]

    assert len(extracted_tasks) == len(expected_task_names)
    for t in extracted_tasks:
        assert t.task_name in expected_task_names, t.task_name


@pytest.mark.skip("Too slow on CI")
def extract_task_qbert():
    def _test(mod, params, target):
        extracted_tasks = ms.relay_integration.extract_tasks(mod, target, params)
        tune_tasks = list(
            filter(
                lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
                extracted_tasks,
            )
        )
        # three int8 dense, two int8 bmm, and one fp32 dense
        assert len(tune_tasks) == 6

        for task in tune_tasks:
            relay_func = list(task.mod.functions.values())[0]
            out_type = relay_func.body.checked_type

            if out_type.dtype == "float32":
                continue

            sch = tvm.tir.Schedule(_normalize_mod(task.dispatched[0]))
            block = sch.get_block("compute")
            annotations = sch.get(block).annotations

            assert "schedule_rule" in annotations
            assert "vnni" in annotations["schedule_rule"]

    mod, params, _ = load_quantized_bert_base(batch_size=1, seq_len=128)
    _test(mod, params, target="llvm -mcpu=cascadelake")


@tvm.testing.skip_if_32bit(reason="Apparently the LLVM version on i386 image is too old")
def test_extract_task_arm_conv2d_nchwc():
    data_shape = (1, 64, 128, 128)
    weight_shape = (32, 64, 1, 1)
    bias_shape = (weight_shape[0],)
    padding = (1, 1)

    data = relay.var("data", shape=data_shape, dtype="int8")
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    bias = relay.var("bias", shape=bias_shape, dtype="int32")
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=weight_shape[2:],
        channels=weight_shape[0],
        padding=padding,
        strides=(1, 1),
        out_dtype="int32",
    )
    bias_add = relay.nn.bias_add(conv2d, bias)
    relay_mod = tvm.IRModule.from_expr(bias_add)

    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=bias_shape).astype("int32")

    params = {"weight": weight_np, "bias": bias_np}

    target = "llvm -device arm_cpu -mtriple aarch64-linux-gnu -mattr=+neon"
    extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
    tune_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    assert len(tune_tasks) == 1

    relay_func = list(tune_tasks[0].mod.functions.values())[0]
    out_type = relay_func.body.checked_type

    # Check that the output is in NCHWc layout
    assert list(out_type.shape) == [1, 8, 130, 130, 4]


def test_meta_schedule_te2primfunc_argument_order_and_lowering():
    # pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
    # fmt: off
    @tvm.script.ir_module
    class _fused_layout_transform:
        @T.prim_func
        def main( # type: ignore
            placeholder: T.Buffer[(T.int64(1), T.int64(3), T.int64(16), T.int64(16)), "float32"], # type: ignore
            T_layout_trans: T.Buffer[(T.int64(1), T.int64(1), T.int64(16), T.int64(16), T.int64(3)), "float32"], # type: ignore
        ) -> None: # type: ignore
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0, i1, i2, i3, i4 in T.grid(T.int64(1), T.int64(1), T.int64(16), T.int64(16), T.int64(3)):
                with T.block("T_layout_trans"):
                    ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(placeholder[ax0, ax1 * T.int64(3) + ax4, ax2, ax3])
                    T.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4])
                    T_layout_trans[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(
                        ax0 < T.int64(1) and ax1 * T.int64(3) + ax4 < T.int64(3) and ax2 < T.int64(16) and ax3 < T.int64(16), # type: ignore
                        placeholder[ax0, ax1 * T.int64(3) + ax4, ax2, ax3],
                        T.float32(0),
                        dtype="float32",
                    )

    @tvm.script.ir_module
    class _fused_layout_transform_1:
        @T.prim_func
        def main(placeholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(16), T.int64(16), T.int64(4)), "float32"], T_layout_trans: T.Buffer[(T.int64(1), T.int64(8), T.int64(16), T.int64(16)), "float32"]) -> None: # type: ignore
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(8), T.int64(16), T.int64(16)):
                with T.block("T_layout_trans"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(placeholder[ax0, ax1 // T.int64(4), ax2, ax3, ax1 % T.int64(4)]) # type: ignore
                    T.writes(T_layout_trans[ax0, ax1, ax2, ax3])
                    T_layout_trans[ax0, ax1, ax2, ax3] = T.if_then_else(ax0 < T.int64(1) and ax1 < T.int64(8) and ax2 < T.int64(16) and ax3 < T.int64(16), placeholder[ax0, ax1 // T.int64(4), ax2, ax3, ax1 % T.int64(4)], T.float32(0), dtype="float32") # type: ignore

    @tvm.script.ir_module
    class _fused_nn_contrib_conv2d_NCHWc:
        @T.prim_func
        def main(placeholder: T.Buffer[(T.int64(1), T.int64(1), T.int64(16), T.int64(16), T.int64(3)), "float32"], placeholder_1: T.Buffer[(T.int64(2), T.int64(1), T.int64(5), T.int64(5), T.int64(3), T.int64(4)), "float32"], conv2d_NCHWc: T.Buffer[(T.int64(1), T.int64(2), T.int64(16), T.int64(16), T.int64(4)), "float32"]) -> None: # type: ignore
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            data_pad = T.alloc_buffer([T.int64(1), T.int64(1), T.int64(20), T.int64(20), T.int64(3)], dtype="float32")
            for i0, i1, i2, i3, i4 in T.grid(T.int64(1), T.int64(1), T.int64(20), T.int64(20), T.int64(3)):
                with T.block("data_pad"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(placeholder[i0_1, i1_1, i2_1 - T.int64(2), i3_1 - T.int64(2), i4_1])
                    T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                    data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(T.int64(2) <= i2_1 and i2_1 < T.int64(18) and T.int64(2) <= i3_1 and i3_1 < T.int64(18), placeholder[i0_1, i1_1, i2_1 - T.int64(2), i3_1 - T.int64(2), i4_1], T.float32(0), dtype="float32") # type: ignore # pylint: disable=R1716
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(2), T.int64(16), T.int64(16), T.int64(4), T.int64(3), T.int64(5), T.int64(5)):
                with T.block("conv2d_NCHWc"):
                    n, oc_chunk, oh, ow, oc_block, ic, kh, kw = T.axis.remap("SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    T.reads(data_pad[n, ic // T.int64(3), oh + kh, ow + kw, ic % T.int64(3)], placeholder_1[oc_chunk, ic // T.int64(3), kh, kw, ic % T.int64(3), oc_block]) # type: ignore
                    T.writes(conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block])
                    with T.init():
                        conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                    conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // T.int64(3), oh + kh, ow + kw, ic % T.int64(3)] * placeholder_1[oc_chunk, ic // T.int64(3), kh, kw, ic % T.int64(3), oc_block] # type: ignore

    # fmt: on
    # pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument

    def _create_verification_database():
        @ms.derived_object
        class VerificationDatabase(ms.database.PyDatabase):
            def __init__(self):
                super().__init__()
                self.tuning_records_: List[TuningRecord] = []
                self.workloads_: List[Workload] = []

            def has_workload(self, mod: IRModule) -> bool:
                for workload in self.workloads_:
                    if tvm.ir.structural_equal(mod, workload.mod):
                        return True
                # Note: The database has already put in all correct workloads
                # This is where we can check if the workload is correct
                raise ValueError(
                    "The workload searched for is not in given database!"
                    + " Incorrect TIR was generated from TE subgraph."
                )

            def commit_workload(self, mod: IRModule) -> ms.database.Workload:
                # No need to deduplicate workload because they are specified
                workload = ms.database.Workload(mod)
                self.workloads_.append(workload)
                return workload

            def commit_tuning_record(self, record: TuningRecord) -> None:
                self.tuning_records_.append(record)

            def get_all_tuning_records(self) -> List[TuningRecord]:
                return self.tuning_records_

            def get_top_k(self, workload: ms.database.Workload, top_k: int) -> List[TuningRecord]:
                return sorted(
                    list(
                        filter(
                            lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                            self.tuning_records_,
                        )
                    ),
                    key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
                )[:top_k]

            def __len__(self) -> int:
                return len(self.tuning_records_)

        database = VerificationDatabase()

        def _commit(mod):
            workload = database.commit_workload(mod)
            database.commit_tuning_record(
                ms.database.TuningRecord(
                    tir.schedule.Trace([], {}),
                    workload=workload,
                    run_secs=[0.1],
                )
            )

        _commit(_fused_layout_transform)
        _commit(_fused_layout_transform_1)
        _commit(_fused_nn_contrib_conv2d_NCHWc)
        return database

    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)

    def _create_relay_mod():
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
        return mod

    mod = _create_relay_mod()
    dev = tvm.cpu()
    target = Target("llvm --num-cores=16")
    params = {
        "weight": np.random.rand(*weight_shape).astype("float32"),
    }
    data = tvm.nd.array(
        np.random.rand(*data_shape).astype("float32"),
        dev,
    )

    with target, _create_verification_database(), PassContext(  # pylint: disable=not-context-manager
        opt_level=3,
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": 7,
            "relay.backend.tir_converter": "default",
        },
    ):
        rt_mod1 = relay.build(mod, target=target, params=params)

    # Compile without meta-schedule for correctness check
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=target, params=params)

    def get_output(data, lib):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input("data", data)
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1)
    expected_output = get_output(data, rt_mod2)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


def test_rewrite_layout_link_params():
    I, O, H, W = 64, 64, 56, 56
    kH = kW = 3

    strides = (1, 1)
    padding = (1, 1)

    data_shape = (1, H, W, I)
    w_shape = (kH, kW, I, O)
    bias_shape = (1, 1, 1, O)

    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight1", shape=w_shape, dtype="float32")
    bias = relay.var("bias", shape=bias_shape, dtype="float32")

    conv = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=(kH, kW),
        channels=O,
        padding=padding,
        strides=strides,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )

    mod = tvm.IRModule.from_expr(conv + bias)

    weight_np = np.random.randn(*w_shape).astype("float32")
    bias_np = np.random.randn(*bias_shape).astype("float32")

    params = {"weight1": weight_np, "bias": bias_np}

    data_np = np.random.randn(*data_shape).astype("float32")

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight_np, bias_np])
        .numpy()
    )

    link_params = True

    target = "llvm --num-cores=4"

    executor = relay.backend.Executor("graph", {"link-params": link_params})
    mod = mod.with_attr("executor", executor)

    for strategy in ["replay-trace", "evolutionary"]:
        with tempfile.TemporaryDirectory() as work_dir:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                params=params,
                work_dir=work_dir,
                max_trials_global=4,
                strategy=strategy,
            )

            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        runtime.set_input("data", data_np)
        runtime.run()

        out = runtime.get_output(0).numpy()

        np.testing.assert_allclose(ref, out, rtol=1e-4, atol=1e-4)


def test_module_equality_ignore_ndarray():
    target = "llvm --num-cores=4"

    data_shape = (128, 128)
    weight_shape1 = (128, 128)
    weight_shape2 = (128, 128)

    data = relay.var("data", shape=data_shape, dtype="float32")
    weight1 = relay.var("weight1", shape=weight_shape1, dtype="float32")
    weight2 = relay.var("weight2", shape=weight_shape2, dtype="float32")
    dense1 = relay.nn.dense(data, weight1)
    dense2 = relay.nn.dense(dense1, weight2)
    mod = tvm.IRModule.from_expr(dense2)

    weight1_np = np.random.randn(*weight_shape1).astype("float32")
    weight2_np = np.random.randn(*weight_shape2).astype("float32")

    params = {"weight1": weight1_np, "weight2": weight2_np}

    executor = relay.backend.Executor("graph", {"link-params": True})
    mod = mod.with_attr("executor", executor)

    # Without using ignore-ndarray for module equality, we get duplicated tasks
    assert len(ms.relay_integration.extract_tasks(mod, target, params)) == 2

    module_eqality = "ignore-ndarray"
    extracted_tasks = ms.relay_integration.extract_tasks(
        mod, target, params, module_equality=module_eqality
    )

    assert len(extracted_tasks) == 1

    with tempfile.TemporaryDirectory() as work_dir:
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks, work_dir, strategy="replay-trace"
        )
        database = ms.tune.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=4,
            module_equality=module_eqality,
        )
        lib = ms.relay_integration.compile_relay(database, mod, target, params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    data_np = np.random.randn(*data_shape).astype("float32")

    runtime.set_input("data", data_np)
    runtime.run()

    out = runtime.get_output(0).numpy()

    ref = np.dot(np.dot(data_np, weight1_np.transpose()), weight2_np.transpose())
    np.testing.assert_allclose(ref, out, rtol=1e-4, atol=1e-4)


def _test_anchor_tuning(target):
    data_shape = (128, 128)
    weight_shape1 = (128, 128)
    weight_shape2 = (128, 128)

    data = relay.var("data", shape=data_shape, dtype="float32")
    weight1 = relay.var("weight1", shape=weight_shape1, dtype="float32")
    weight2 = relay.var("weight2", shape=weight_shape2, dtype="float32")
    dense1 = relay.nn.dense(data, weight1)
    dense2 = relay.nn.dense(dense1 + relay.const(1.0, dtype="float32"), weight2)
    mod = tvm.IRModule.from_expr(dense2 - data + relay.const(1.0, dtype="float32"))

    weight1_np = np.random.randn(*weight_shape1).astype("float32")
    weight2_np = np.random.randn(*weight_shape2).astype("float32")

    data_np = np.random.randn(*data_shape).astype("float32")
    params = {"weight1": weight1_np, "weight2": weight2_np}

    module_equality = "anchor-block"

    extracted_tasks = ms.relay_integration.extract_tasks(
        mod, target, params, module_equality=module_equality
    )

    assert len(extracted_tasks) == 1

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=4,
            strategy="replay-trace",
            module_equality=module_equality,
        )
        lib = ms.relay_integration.compile_relay(database, mod, target, params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data_np)
    runtime.run()
    out = runtime.get_output(0).numpy()

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight1_np, weight2_np])
        .numpy()
    )

    np.testing.assert_allclose(ref, out, atol=1e-3)


def test_anchor_tuning_cpu():
    _test_anchor_tuning("llvm --num-cores=4")


def test_anchor_tuning_cpu_link_params():
    data_shape = (128, 128)
    weight_shape1 = (128, 128)
    weight_shape2 = (128, 128)

    data = relay.var("data", shape=data_shape, dtype="float32")
    weight1 = relay.var("weight1", shape=weight_shape1, dtype="float32")
    weight2 = relay.var("weight2", shape=weight_shape2, dtype="float32")
    dense1 = relay.nn.dense(data, weight1)
    dense2 = relay.nn.dense(dense1, weight2)
    mod = tvm.IRModule.from_expr(dense2 + relay.const(1.0, dtype="float32"))

    weight1_np = np.random.randn(*weight_shape1).astype("float32")
    weight2_np = np.random.randn(*weight_shape2).astype("float32")

    data_np = np.random.randn(*data_shape).astype("float32")
    params = {"weight1": weight1_np, "weight2": weight2_np}

    module_equality = "anchor-block"
    target = "llvm --num-cores=4"

    executor = relay.backend.Executor("graph", {"link-params": True})
    mod = mod.with_attr("executor", executor)

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=4,
            strategy="replay-trace",
            module_equality=module_equality,
        )
        lib = ms.relay_integration.compile_relay(database, mod, target, params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data_np)
    runtime.run()
    out = runtime.get_output(0).numpy()

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight1_np, weight2_np])
        .numpy()
    )

    np.testing.assert_allclose(ref, out, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
