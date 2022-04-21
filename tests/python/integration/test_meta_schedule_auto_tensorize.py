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
import pytest
import tvm
from tvm import relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule import ApplyHistoryBest
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm import meta_schedule as ms
from tvm.tir.tensor_intrin import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN, DP4A_INTRIN
import tempfile
import tvm.topi.testing


config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=64,
    max_trials_per_task=64,
    max_trials_global=20000,
)

sch_rules_for_vnni = [
    schedule_rule.AutoInline(
        into_producer=False,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=True,
        require_ordered=True,
        disallow_op=["tir.exp"],
    ),
    schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
    schedule_rule.MultiLevelTilingWithIntrin(
        VNNI_INTRIN,
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=64,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
    schedule_rule.RandomComputeLocation(),
]

postprocs_for_vnni = [
    postproc.DisallowDynamicLoop(),
    postproc.RewriteParallelVectorizeUnroll(),
    postproc.RewriteReductionBlock(),
    postproc.RewriteTensorize(vectorize_init_loop=True),
]


def tune_vnni(relay_mod, data_np, weight_np, op_name):
    target = "llvm -mcpu=cascadelake -num-cores 4"
    dev = tvm.device(target, 0)

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        .evaluate()(*[data_np, weight_np]).numpy()
    )

    params = {"weight": weight_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: op_name in task.task_name,
            extracted_tasks,
        )
    )

    with tempfile.TemporaryDirectory() as work_dir:
        database = tune_extracted_tasks(
            tune_tasks,
            config,
            work_dir=work_dir,
            sch_rules=lambda: sch_rules_for_vnni,
            postprocs=lambda: postprocs_for_vnni,
        )

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    asm = lib.lib.get_source("asm")
    assert "vpdpbusd" in asm

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data_np)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


@pytest.mark.skip("Requires cascadelake")
def test_vnni_dense():
    M, N, K = 1024, 1024, 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    dense = relay.nn.dense(data, weight, out_dtype=out_dtype)

    relay_mod = tvm.IRModule.from_expr(dense)

    data_np = np.random.uniform(1, 10, size=data_shape).astype(data_dtype)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)

    tune_vnni(relay_mod, data_np, weight_np, "dense")


@pytest.mark.skip("Requires cascadelake")
def test_vnni_conv2d():
    d_shape = (1, 64, 56, 56)
    w_shape = (64, 64, 3, 3)

    data_dtype = "uint8"
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

    data_np = np.random.uniform(1, 10, d_shape).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=w_shape).astype("int8")

    tune_vnni(relay_mod, data_np, weight_np, "conv2d")


@pytest.mark.skip("Requires cascadelake")
def test_bert_int8():
    relay_mod, params, input_info = load_quantized_bert_base()

    relay_mod = relay.transform.FastMath()(relay_mod)

    target = "llvm -mcpu=cascadelake -num-cores 4"

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = []

    for task in filter(
        lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
        extracted_tasks,
    ):
        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if out_type.dtype != "float32":
            tune_tasks.append(task)

    with tempfile.TemporaryDirectory() as work_dir:
        database = tune_extracted_tasks(
            tune_tasks,
            config,
            work_dir=work_dir,
            sch_rules=lambda: sch_rules_for_vnni,
            postprocs=lambda: postprocs_for_vnni,

        )

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    inputs = []

    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)

    print(runtime.benchmark(dev, number=1, repeat=50).mean)


if __name__ == "__main__":
    # test_vnni_dense()
    # test_vnni_conv2d()
    test_bert_int8()
