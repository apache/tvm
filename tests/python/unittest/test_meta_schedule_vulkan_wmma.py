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
import tvm
from tvm import te
import tvm.testing
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
from tvm import auto_scheduler
from tvm import meta_schedule as ms
import os
from os import path as osp
import argparse
import sys
from types import MappingProxyType
import tempfile
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_INTRIN,
    LDMATRIX_16x16_B_INTRIN,
    LDMATRIX_16x16_B_TRANS_INTRIN,
    LDMATRIX_16x32_A_INTRIN,
    LDMATRIX_32x16_B_INTRIN,
    LDMATRIX_16x32_B_TRANS_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_f16f16f16_INTRIN,
    MMA_f16f16f16_TRANS_INTRIN,
    MMA_i8i8i32_INTRIN,
    MMA_i8i8i32_TRANS_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_fill_16x16_f16_INTRIN,
    MMA_fill_16x16_i32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f16_global_INTRIN,
    MMA_store_16x16_i32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
    shared_32x16_to_ldmatrix_32x16_layout,
    shared_16x32_to_ldmatrix_32x16_layout,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
)


def _create_tmp_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "database_workloads.json")
    path_tuning_record = osp.join(tmpdir, "database_tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)


@tvm.testing.requires_vulkan
def test_wmma(
    batch_size=1,
    in_channels=16,
    height=16,
    width=16,
    out_channels=16,
    kernel_h=1,
    kernel_w=1,
    in_dtype="float16",
    out_dtype="float32",
    padding=[0, 0, 0, 0],
):
    # compute as conv2d
    if sys.platform == "win32":
        target_host = "llvm"
    else:
        target_host = "llvm -mtriple=x86_64-linux-gnu"

    target_str = "vulkan -from_device=0"
    target_str += " -supports_cooperative_matrix=1"

    target = tvm.target.Target(target_str, host=target_host)
    if not target.supports_cooperative_matrix:
        return

    data_shape = (batch_size, in_channels, height, width)
    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)

    data = relay.var("data", shape=data_shape, dtype=in_dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=in_dtype)
    conv = relay.nn.conv2d(
        data,
        kernel,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=padding,
        strides=[1, 1],
        out_dtype=out_dtype,
        channels=out_channels,
        kernel_size=(kernel_h, kernel_w),
    )

    func = relay.Function([data, kernel], conv)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.relay.transform.InferType()(mod)
    kernel_np = np.random.uniform(size=kernel_shape).astype(in_dtype)

    mod_params = {"kernel": kernel_np}

    from tvm.meta_schedule.relay_integration import extract_tasks

    tasks = extract_tasks(
        mod,
        target_str,
        mod_params,
        pass_config=MappingProxyType(
            {
                "relay.backend.use_meta_schedule": True,
                "relay.backend.use_meta_schedule_dispatch": True,
                "relay.backend.tir_converter": "default",
                "relay.backend.use_int32_const": True,
            }
        ),
    )

    dev = tvm.vulkan(0)
    data_np = np.random.uniform(size=data_shape).astype(in_dtype)
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        assert osp.exists(database.path_workload)
        assert osp.exists(database.path_tuning_record)
        work_dir = tmpdir

        runner = ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(
                number=3,
                repeat=1,
                min_repeat_ms=300,
                enable_cpu_cache_flush="llvm" in str(target_str),
            )
        )
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=tasks,
            work_dir=work_dir,
            strategy="evolutionary",
        )
        database = ms.tune.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=work_dir,
            max_trials_global=128,
            max_trials_per_task=128,
            num_trials_per_iter=128,
            runner=runner,
            database=database,
            min_design_space=5,
        )

        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target_str,
            params=mod_params,
            backend="graph",
            pass_config=MappingProxyType(
                {
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.use_meta_schedule_dispatch": True,
                    "relay.backend.tir_converter": "default",
                    "relay.backend.use_int32_const": True,
                }
            ),
        )

        cpu_dev = tvm.cpu(0)
        ref = (
            relay.create_executor(
                "vm",
                mod=mod,
                device=cpu_dev,
                target="llvm",
            )
            .evaluate()(data_np, kernel_np)
            .numpy()
        )
        runtime = graph_executor.GraphModule(lib["default"](dev))
        runtime.set_input("data", data_np)
        runtime.run()
        out = runtime.get_output(0).asnumpy()
        np.testing.assert_allclose(out, ref, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
