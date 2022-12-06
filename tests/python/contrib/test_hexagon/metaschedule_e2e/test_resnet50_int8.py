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
"""Test Resnet50 int8 with MetaSchedule"""

import os
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relay
from tvm._ffi import register_func
from tvm.contrib.hexagon.meta_schedule import (
    get_hexagon_local_builder,
    get_hexagon_rpc_runner,
)
from tvm.meta_schedule import postproc, schedule_rule
from tvm.tir.schedule import BlockRV, Schedule
from tvm.tir.schedule.analysis import has_block
from tvm.tir.tensor_intrin.hexagon import (
    VRMPY_u8i8i32_INTRIN,
    VRMPY_u8u8i32_INTRIN,
    VRMPY_u8i8i32_VTCM_INTRIN,
)

from ..infrastructure import get_hexagon_target

MODEL_JSON = "resnet50_int8.json"
EXECUTOR = relay.backend.Executor("graph", {"link-params": True})
TARGET_LLVM = tvm.target.Target("llvm")
TARGET_HEXAGON = get_hexagon_target("v68")
MODEL_PARAMS = "resnet50_int8.params"


def tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher):
    """Tune VRMPY with auto tensorization."""
    sch_rules = [
        schedule_rule.ApplyCustomRule(),
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
    mod = mod.with_attr("executor", EXECUTOR)

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=TARGET_HEXAGON,
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
            # This enables anchor-block tuning, where different subgraphs
            # with the same anchor block workload will be identified as equal.
            # It reduces the number of conv2d tuning tasks in the int8 resnet50 model
            # from 36 to 23, with negligible performance difference.
            module_equality="anchor-block",
        )
        return ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=TARGET_HEXAGON,
            params=params,
        )


@tvm.testing.requires_hexagon
def test_resnet50(hexagon_launcher):
    """Test Resnet50."""

    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    if not os.path.exists(MODEL_JSON):
        pytest.skip(msg="Run python export_models.py first.")

    with open(MODEL_JSON, "r") as file:
        mod = tvm.ir.load_json(file.read())

    with open(MODEL_PARAMS, "rb") as file:
        params = relay.load_param_dict(file.read())
    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    do_tune = True

    if do_tune:
        hexagon_lowered = tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher)
    else:
        with tvm.transform.PassContext(opt_level=3):
            hexagon_lowered = relay.build(
                mod,
                tvm.target.Target(TARGET_HEXAGON, host=TARGET_HEXAGON),
                params=params,
                executor=EXECUTOR,
            )

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(TARGET_LLVM, host=TARGET_LLVM),
            params=params,
        )

    with hexagon_launcher.create_session() as session:
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


def evaluate_mod(hexagon_launcher, hexagon_lowered, llvm_lowered, input_name, inp, benchmark=False):
    """Evaluate the Modules against llvm version."""
    with hexagon_launcher.create_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())
        graph_mod.run()
        output = graph_mod.get_output(0).numpy()

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())
        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()

        if benchmark:
            time_ms = graph_mod.benchmark(session.device, number=1, repeat=1).mean * 1e3
            print("hexagon time elapsed: ", time_ms)
            debug_ex = session.get_graph_debug_executor(
                hexagon_lowered.get_graph_json(), hexagon_lowered.lib
            )
            print(debug_ex.profile(input_name=inp.copy()))

        np.testing.assert_allclose(ref_result, output, atol=1e-4, rtol=1e-5)


def load_model():
    """Load renset50 model."""
    if not os.path.exists(MODEL_JSON):
        pytest.skip(msg="Run python export_models.py first.")

    with open(MODEL_JSON, "r") as file:
        mod = tvm.ir.load_json(file.read())

    with open(MODEL_PARAMS, "rb") as file:
        params = relay.load_param_dict(file.read())

    return mod, params


def _schedule_packed_8x8x32_conv2d():
    """Manually schedule a conv2d block, created from TE compute op via CreatePrimFunc,
    using 8x8x32 packed layout.
    """

    def schedule_fn(sch, conv2d_block: Optional[BlockRV] = None) -> bool:
        if conv2d_block is None:
            if has_block(sch, "conv2d_NCHWc_int8"):
                conv2d_block = sch.get_block("conv2d_NCHWc_int8")
            else:
                return False

        assert "conv2d_NCHWc_int8" in sch.get(conv2d_block).annotations["schedule_rule"]

        # Apply scheduling

        post_blocks = sch.get_consumers(conv2d_block)
        if len(post_blocks) > 0:
            # Fuse all intermediate post ops into the last op.
            # This is equivalent to the traverse_inline function used in TE schedules.
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
                    break
                post_blocks = next_post_blocks
        else:
            outer_block = conv2d_block

        # Move the conv2d mma into the injective post mma compute block
        if outer_block != conv2d_block:
            loops = sch.get_loops(outer_block)
            # TODO(csullivan): Currently does all post conv2d mma steps
            # directly after accumulation for one spatial pixel. May
            # be desirable to do this with coarser spatial granularity
            sch.compute_at(conv2d_block, loops[4])

        def index_map_nchw32c_nchw8h8w32c(n_batch, channel, height, width, channel_32):
            return [n_batch, channel, height // 8, width // 8, height % 8, width % 8, channel_32]

        # Add cache for input and output activation layout transform,
        # note that weight is already in correct layout
        # pylint: disable=unused-variable
        input_cache = sch.cache_read(conv2d_block, 0, "global.vtcm")
        output_cache = sch.cache_write(outer_block, 0, "global.vtcm")
        # Transform the layout of the input
        sch.transform_layout(
            conv2d_block, ("read", 0), index_map=index_map_nchw32c_nchw8h8w32c, pad_value=0
        )
        # Transform the layout of the int32 accumulator
        sch.transform_layout(
            conv2d_block, ("write", 0), index_map=index_map_nchw32c_nchw8h8w32c, pad_value=0
        )
        # Transform the layout of the output
        sch.transform_layout(
            outer_block, ("write", 0), index_map=index_map_nchw32c_nchw8h8w32c, pad_value=0
        )
        return True

    return schedule_fn


def tune_conv2d_template(
    mod,
    scheduler,
    schedule_tag,
    params,
    hexagon_launcher,
    pass_config: Mapping[str, Any] = MappingProxyType({}),
):
    """Generate packed 8*8*32 template."""

    def schedule_rule_conv2d(sch: Schedule, conv2d_block: BlockRV):
        scheduler()(sch, conv2d_block)
        return [sch]

    register_func(
        "meta_schedule.conv2d_NCHWc_int8.{}.hexagon".format(schedule_tag), schedule_rule_conv2d
    )

    def schedule_conv2d_for_tune(sch: Schedule):
        scheduler()(sch)

    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", EXECUTOR)

    pass_context = None
    if len(pass_config.items()) > 0:
        pass_context = (
            tvm.transform.PassContext(opt_level=3, config=pass_config)
            if pass_config is not None
            else None
        )

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=TARGET_HEXAGON,
            params=params,
            work_dir=work_dir,
            max_trials_global=20000,
            max_trials_per_task=1,
            num_trials_per_iter=1,
            strategy="replay-trace",
            builder=get_hexagon_local_builder(pass_context),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=1),
            # Apply MS auto scheduling rules for all blocks, but utilize
            # the custom block scheduling strategy registered above for
            # blocks annotated as `schedule_rule:meta_schedule.conv2d_NCHWc_int8`
            # space=ms.space_generator.PostOrderApply(
            #     f_block_filter=None,
            #     sch_rules="from-target",
            #     postprocs=[],
            #     mutator_probs="from-target",
            # ),
            # Constrain search space to only be the single
            # schedule provided for all blocks. No auto
            # scheduling will be possible.
            space=ms.space_generator.ScheduleFn(
                schedule_conv2d_for_tune,
                sch_rules=[],
                postprocs=[],
                mutator_probs={},
            ),
            # Without this, the same workloads with different constant weights
            # are treated as distinct tuning tasks.
            module_equality="ignore-ndarray",
        )

        # Add default options so that it still uses the base config.
        pass_config["relay.backend.use_meta_schedule"] = True
        pass_config["relay.backend.tir_converter"] = "default"
        return ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=TARGET_HEXAGON,
            params=params,
            pass_config=pass_config,
        )


@tvm.testing.requires_hexagon
def test_packed_8x8x32_resnet50(hexagon_launcher):
    """Test packed 8*8*32 Resnet50"""

    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    mod, params = load_model()

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    do_tune = True

    if do_tune:
        hexagon_lowered = tune_conv2d_template(
            mod, _schedule_packed_8x8x32_conv2d, "packed_8x8x32", params, hexagon_launcher
        )
    else:
        with tvm.transform.PassContext(opt_level=3):
            hexagon_lowered = relay.build(
                mod,
                tvm.target.Target(TARGET_HEXAGON, host=TARGET_HEXAGON),
                params=params,
                executor=EXECUTOR,
            )

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(TARGET_LLVM, host=TARGET_LLVM),
            params=params,
        )

    evaluate_mod(hexagon_launcher, hexagon_lowered, llvm_lowered, input_name, inp)


def _schedule_async_dma_conv2d():
    """Manually schedule a conv2d block, created from TE compute op via CreatePrimFunc,
    using 8x8x32 packed layout.
    """

    def schedule_fn(sch, conv2d_block: Optional[BlockRV] = None) -> bool:
        if conv2d_block is None:
            if has_block(sch, "conv2d_NCHWc_int8"):
                conv2d_block = sch.get_block("conv2d_NCHWc_int8")
            else:
                return False

        assert "conv2d_NCHWc_int8" in sch.get(conv2d_block).annotations["schedule_rule"]

        # Apply scheduling

        post_blocks = sch.get_consumers(conv2d_block)
        if len(post_blocks) > 0:
            # Fuse all intermediate post ops into the last op.
            # This is equivalent to the traverse_inline function used in TE schedules.
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
                    break
                post_blocks = next_post_blocks
        else:
            outer_block = conv2d_block

        # Move the conv2d mma into the injective post mma compute block
        if outer_block != conv2d_block:
            loops = sch.get_loops(outer_block)
            # Compute at the second loop for pipelining.
            sch.compute_at(conv2d_block, loops[1], preserve_unit_loops=True)

        # Add cache for input and output for copying data to vtcm.
        input_a_cache = sch.cache_read(conv2d_block, 0, "global.vtcm")
        sch.compute_at(input_a_cache, sch.get_loops(conv2d_block)[1])
        sch.fuse(*sch.get_loops(input_a_cache)[2:])

        input_b_cache = sch.cache_read(conv2d_block, 1, "global.vtcm")
        sch.compute_at(input_b_cache, sch.get_loops(conv2d_block)[1])
        sch.fuse(*sch.get_loops(input_b_cache)[2:])

        output_cache_write = sch.cache_write(conv2d_block, 0, "global.vtcm")
        sch.fuse(*sch.get_loops(output_cache_write)[2:])

        conv2d_loops = sch.get_loops(block=conv2d_block)
        o_c, k_h, k_w, x_0, x_1, i_c = conv2d_loops[-6:]
        ic_o, ic_i = sch.split(loop=i_c, factors=[None, 4], preserve_unit_iters=True)
        oc_o, oc_i = sch.split(loop=o_c, factors=[None, 32], preserve_unit_iters=True)
        sch.reorder(oc_o, k_h, k_w, x_0, x_1, ic_o, oc_i, ic_i)
        new_loops = sch.get_loops(block=conv2d_block)
        sch.parallel(new_loops[4])
        sch.unroll(new_loops[5])
        # TODO(nverke): Add compute optimizations here.
        sch.blockize(loop=oc_i)

        sch.tensorize(oc_i, VRMPY_u8i8i32_VTCM_INTRIN)

        pipeline_loop = conv2d_loops[1]
        sch.annotate(pipeline_loop, "software_pipeline_stage", [0, 0, 1, 2, 3])
        sch.annotate(pipeline_loop, "software_pipeline_order", [0, 1, 2, 3, 4])
        sch.annotate(pipeline_loop, "software_pipeline_async_stages", [0, 2])

        return True

    return schedule_fn


@tvm.testing.requires_hexagon
def test_async_dma_resnet50(hexagon_launcher):
    """Test async dma Resnet50"""

    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    mod, params = load_model()

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    pass_config = {
        "tir.use_async_copy": 1,
        "tir.merge_async_commit_queue_scope": False,
        "relay.backend.use_meta_schedule": True,
        "relay.backend.tir_converter": "default",
    }

    hexagon_lowered = tune_conv2d_template(
        mod, _schedule_async_dma_conv2d, "async_dma", params, hexagon_launcher, pass_config
    )
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod, tvm.target.Target(TARGET_LLVM, host=TARGET_LLVM), params=params
        )
    evaluate_mod(hexagon_launcher, hexagon_lowered, llvm_lowered, input_name, inp, True)


if __name__ == "__main__":
    tvm.testing.main()
