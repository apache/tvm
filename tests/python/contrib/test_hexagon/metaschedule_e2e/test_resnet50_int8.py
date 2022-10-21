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
import os
import numpy as np
import pytest
import tempfile

import tvm
import tvm.testing
from tvm import relay
from tvm.meta_schedule import postproc, schedule_rule
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8i8i32_INTRIN, VRMPY_u8u8i32_INTRIN
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm import meta_schedule as ms
from ..infrastructure import get_hexagon_target


executor = relay.backend.Executor("graph", {"link-params": True})
target = get_hexagon_target("v68")
target_llvm = tvm.target.Target("llvm")
model_json = "resnet50_int8.json"
model_params = "resnet50_int8.params"


def tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher):
    sch_rules = [
        schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        ),
        # VRMPY_u8i8i32_INTRIN is used for conv2d. See topi/hexagon/conv2d_alter_op.py
        # for why we use different intrins for conv2d and dense.
        schedule_rule.MultiLevelTilingWithIntrin(
            VRMPY_u8i8i32_INTRIN,
            structure="SRSRS",
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
        # VRMPY_u8u8i32_INTRIN is used for dense
        schedule_rule.MultiLevelTilingWithIntrin(
            VRMPY_u8u8i32_INTRIN,
            structure="SRSRS",
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
            max_vectorize_extent=128,
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        ),
    ]

    postprocs = [
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorize(vectorize_init_loop=True),
    ]

    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            # for faster tuning
            max_trials_global=20000,
            max_trials_per_task=8,
            num_trials_per_iter=8,
            strategy="replay-trace",
            # max_trials_global=20000,
            # num_trials_per_iter=32,
            # max_trials_per_task=128,
            # strategy="evolutionary",
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
            space=ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs={},
            ),
            # Without this, the same workloads with different constant weights
            # are treated as distinct tuning tasks.
            module_equality="ignore-ndarray",
        )

        return ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
        )


@pytest.mark.skip("End-to-end tuning is skipped on CI.")
@tvm.testing.requires_hexagon
def test_resnet50(hexagon_launcher):
    if not os.path.exists(model_json):
        pytest.skip(msg="Run python export_models.py first.")

    with open(model_json, "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open(model_params, "rb") as fi:
        params = relay.load_param_dict(fi.read())
    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    do_tune = True

    if do_tune:
        hexagon_lowered = tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher)
    else:
        with tvm.transform.PassContext(opt_level=3):
            hexagon_lowered = relay.build(
                mod,
                tvm.target.Target(target, host=target),
                params=params,
                executor=executor,
            )

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

    with hexagon_launcher.start_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())
        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())
        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()

        np.testing.assert_allclose(ref_result, hexagon_output, atol=1e-4, rtol=1e-5)

        time_ms = graph_mod.benchmark(session.device, number=1, repeat=20).mean * 1e3

        print("time elapsed: ", time_ms)

        debug_ex = session.get_graph_debug_executor(
            hexagon_lowered.get_graph_json(), hexagon_lowered.lib
        )
        print(debug_ex.profile(input_name=inp.copy()))
