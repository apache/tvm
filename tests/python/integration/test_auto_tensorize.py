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
"""Integration test for MetaSchedule's auto tensorization."""
import tempfile

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing import relay_workload
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.tir.tensor_intrin.arm_cpu import DP4A_INTRIN
from tvm.tir.tensor_intrin.rocm import AMDGPU_SDOT4_INTRIN
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

SCH_RULES_FOR_VNNI = [
    ms.schedule_rule.AutoInline(
        into_producer=False,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=True,
        require_ordered=True,
        disallow_op=["tir.exp"],
    ),
    ms.schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
    ms.schedule_rule.MultiLevelTilingWithIntrin(
        VNNI_INTRIN,
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=64,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
    ms.schedule_rule.RandomComputeLocation(),
]


def _get_sch_rules_for_dp4a(intrin):
    return [
        ms.schedule_rule.MultiLevelTilingWithIntrin(
            intrin,
            structure="SSSRRSRS",
            tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            max_innermost_factor=64,
            vector_load_lens=[1, 2, 3, 4],
            reuse_read=ms.schedule_rule.ReuseType(
                req="must",
                levels=[4],
                scope="shared",
            ),
            reuse_write=ms.schedule_rule.ReuseType(
                req="must",
                levels=[3],
                scope="local",
            ),
        ),
        ms.schedule_rule.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        ms.schedule_rule.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
        ms.schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ]


SCH_RULES_FOR_DP4A = _get_sch_rules_for_dp4a(DP4A_INTRIN)
SCH_RULES_FOR_SDOT4 = _get_sch_rules_for_dp4a(AMDGPU_SDOT4_INTRIN)

POSTPROCS_FOR_VNNI = [
    ms.postproc.DisallowDynamicLoop(),
    ms.postproc.RewriteParallelVectorizeUnroll(),
    ms.postproc.RewriteReductionBlock(),
    ms.postproc.RewriteTensorize(vectorize_init_loop=True),
]

POSTPROCS_FOR_DP4A = [
    ms.postproc.DisallowDynamicLoop(),
    ms.postproc.RewriteCooperativeFetch(),
    ms.postproc.RewriteUnboundBlock(),
    ms.postproc.RewriteParallelVectorizeUnroll(),
    ms.postproc.RewriteReductionBlock(),
    ms.postproc.RewriteTensorize(),
    ms.postproc.VerifyGPUCode(),
]


def tune_and_test(relay_mod, data_np, weight_np, op_name, target, sch_rules, postprocs):
    """Test tuning."""
    tgt = "cuda" if "nvidia" in target else target
    dev = tvm.device(tgt, 0)
    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=tgt)
        .evaluate()(*[data_np, weight_np])
        .numpy()
    )
    params = {"weight": weight_np}
    tune_tasks = list(
        filter(
            lambda task: op_name in task.task_name,
            ms.relay_integration.extract_tasks(relay_mod, target, params),
        )
    )
    with tempfile.TemporaryDirectory() as work_dir:
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=tune_tasks,
            work_dir=work_dir,
            space=ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
            ),
        )
        database = ms.tune.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=32,
        )
    with database, tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        lib = relay.build(relay_mod, target=target, params=params)

    if "cascadelake" in target:
        asm = lib.lib.get_source("asm")
        assert "vpdpbusd" in asm

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input("data", data_np)
    runtime.run()
    out = runtime.get_output(0).numpy()
    np.testing.assert_equal(out, ref)


def _test_dense(data_dtype, sch_rules, postprocs, target):
    dim_m, dim_n, dim_k = 1024, 1024, 1024
    data_shape = (dim_m, dim_k)
    weight_shape = (dim_n, dim_k)

    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    dense = relay.nn.dense(data, weight, out_dtype=out_dtype)

    relay_mod = tvm.IRModule.from_expr(dense)

    data_np = np.random.uniform(1, 10, size=data_shape).astype(data_dtype)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)

    tune_and_test(relay_mod, data_np, weight_np, "dense", target, sch_rules, postprocs)


def _test_conv2d(data_dtype, sch_rules, postprocs, target):
    d_shape = (1, 64, 56, 56)
    w_shape = (64, 64, 3, 3)

    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[0]
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=(1, 1),
        strides=(1, 1),
        out_dtype=out_dtype,
    )

    relay_mod = tvm.IRModule.from_expr(conv2d)

    data_np = np.random.uniform(1, 10, d_shape).astype(data_dtype)
    weight_np = np.random.uniform(1, 10, size=w_shape).astype("int8")

    tune_and_test(relay_mod, data_np, weight_np, "conv2d", target, sch_rules, postprocs)


def _test_bert_int8(relay_mod, params, input_info, target, sch_rules, postprocs):
    relay_mod = relay.transform.FastMath()(relay_mod)
    tune_tasks = [
        task
        for task in ms.relay_integration.extract_tasks(relay_mod, target, params)
        if "dense" in task.task_name or "batch_matmul" in task.task_name
    ]
    with tempfile.TemporaryDirectory() as work_dir:
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=tune_tasks,
            work_dir=work_dir,
            space=ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
            ),
        )
        database = ms.tune.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_per_task=32,
            max_trials_global=20000,
        )
    with database, tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.device("cuda" if "nvidia" in target else target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    inputs = []
    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)
    print(runtime.benchmark(dev, number=1, repeat=50).mean)


@tvm.testing.requires_cascadelake
def test_vnni_dense():
    _test_dense(
        "uint8", SCH_RULES_FOR_VNNI, POSTPROCS_FOR_VNNI, "llvm -mcpu=cascadelake -num-cores 4"
    )


@pytest.mark.skip("Only tested locally on sm_86 (for cuda) which is not supported by CI")
@tvm.testing.requires_gpu
def test_dp4a_dense():
    _test_dense("int8", SCH_RULES_FOR_DP4A, POSTPROCS_FOR_DP4A, "nvidia/geforce-rtx-3070")
    # Uncomment to test on vulkan or rocm target
    # _test_dense(
    #     "int8", SCH_RULES_FOR_DP4A, POSTPROCS_FOR_DP4A, "vulkan -from_device=0"
    # )
    # _test_dense(
    #     "int8", SCH_RULES_FOR_SDOT4, POSTPROCS_FOR_DP4A, "rocm"
    # )


@tvm.testing.requires_cascadelake
def test_vnni_conv2d():
    _test_conv2d(
        "uint8", SCH_RULES_FOR_VNNI, POSTPROCS_FOR_VNNI, "llvm -mcpu=cascadelake -num-cores 4"
    )


@pytest.mark.skip("Only tested locally on sm_86 (for cuda) which is not supported by CI")
@tvm.testing.requires_gpu
def test_dp4a_conv2d():
    _test_conv2d("int8", SCH_RULES_FOR_DP4A, POSTPROCS_FOR_DP4A, "nvidia/geforce-rtx-3070")
    # Uncomment to test on vulkan or rocm target
    # _test_conv2d(
    #     "int8", SCH_RULES_FOR_DP4A, POSTPROCS_FOR_DP4A, "vulkan -from_device=0"
    # )
    # _test_conv2d(
    #     "int8", SCH_RULES_FOR_SDOT4, POSTPROCS_FOR_DP4A, "rocm"
    # )


@tvm.testing.requires_cascadelake
@pytest.mark.skip_if(tvm.testing.IS_IN_CI, reason="Slow on CI")
def test_vnni_bert_int8():
    relay_mod, params, input_info = load_quantized_bert_base()
    _test_bert_int8(
        relay_mod,
        params,
        input_info,
        "llvm -mcpu=cascadelake -num-cores 4",
        SCH_RULES_FOR_VNNI,
        POSTPROCS_FOR_VNNI,
    )


@tvm.testing.requires_gpu
@pytest.mark.skip("Slow on CI")
def test_dp4a_bert_int8():
    relay_mod, params, input_info = load_quantized_bert_base()
    _test_bert_int8(
        relay_mod,
        params,
        input_info,
        "nvidia/geforce-rtx-3070",
        SCH_RULES_FOR_DP4A,
        POSTPROCS_FOR_DP4A,
    )
    # Uncomment to test on vulkan or rocm target
    # _test_bert_int8(
    #     relay_mod,
    #     params,
    #     input_info,
    #     "vulkan -from_device=0",
    #     SCH_RULES_FOR_DP4A,
    #     POSTPROCS_FOR_DP4A,
    # )
    # _test_bert_int8(
    #     relay_mod,
    #     params,
    #     input_info,
    #     "rocm",
    #     SCH_RULES_FOR_SDOT4
    #     POSTPROCS_FOR_DP4A,
    # )


@tvm.testing.requires_gpu
@pytest.mark.skip("Slow on CI")
@pytest.mark.parametrize(
    ["model_name", "input_shape"],
    [("bert_base", (8, 128)), ("resnet_18", (16, 3, 224, 224)), ("resnet_50", (16, 3, 224, 224))],
)
def test_cuda_tensor_core(model_name, input_shape):
    """Integration tests of auto tensorization with CUDA tensor core"""
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    dev = tvm.cuda()
    if model_name.startswith("bert"):
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)
    mod, params, (input_name, _, _) = relay_workload.get_network(model_name, input_shape)
    seq = tvm.transform.Sequential(
        [
            relay.transform.ToMixedPrecision(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    def convert_layout(mod):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]})]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

    with tempfile.TemporaryDirectory() as work_dir:
        with ms.Profiler() as profiler:
            converted_mod = convert_layout(mod)
            database = ms.relay_integration.tune_relay(
                mod=converted_mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=3000,
                params=params,
            )
            rt_mod1 = ms.relay_integration.compile_relay(
                database=database,
                mod=converted_mod,
                target=target,
                params=params,
            )
        print(profiler.table())

        # Compile without MetaSchedule for correctness check
        with tvm.transform.PassContext(opt_level=0):
            rt_mod2 = relay.build(mod, target=target, params=params)

        def get_output(data, lib):
            module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
            module.set_input(input_name, data)
            module.run()
            return module.get_output(0).numpy()

        # Check correctness
        actual_output = get_output(data, rt_mod1)
        expected_output = get_output(data, rt_mod2)
        assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=2e-2)


if __name__ == "__main__":
    tvm.testing.main()
