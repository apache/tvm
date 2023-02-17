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
""" Test rpc based launcher for hexagon """
import tempfile

import numpy as np
import pytest
import tvm.testing
import tvm.topi.testing
from tvm import meta_schedule as ms
from tvm import relay
from tvm.contrib.hexagon.meta_schedule import (
    get_hexagon_local_builder,
    get_hexagon_rpc_runner,
)
from tvm.meta_schedule import postproc, schedule_rule
from tvm.tir.tensor_intrin.hexagon import (
    VRMPY_u8u8i32_VTCM_READS_INTRIN,
)

from .infrastructure import get_hexagon_target


def tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher):
    sch_rules_async = [
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
        schedule_rule.MultiLevelTilingHexagon(
            intrin_groups=[
                {"compute": VRMPY_u8u8i32_VTCM_READS_INTRIN},
            ],
            structure="SRSRS",
            tile_binds=None,
            max_innermost_factor=64,  # 64 // tensor intrin size
            vector_load_lens=None,
            reuse_read=ms.schedule_rule.ReuseType(
                req="must",
                levels=[2],
                scope="global.vtcm",
            ),
            reuse_write=None,
            use_software_pipeline=True,
        ),
        schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,
            max_vectorize_extent=-1,
            unroll_max_steps=[8, 16, 32],
            unroll_explicit=True,
        ),
    ]

    postprocs = [
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorize(vectorize_init_loop=True),
        postproc.VerifyVTCMLimit(),
        postproc.DisallowAsyncStridedMemCopy(merge_async_commit_queue_scope=False),
    ]

    target = get_hexagon_target("v68")
    executor = relay.backend.Executor("graph", {"link-params": True})
    mod = mod.with_attr("executor", executor)


    config = {
        "tir.use_async_copy": True,
        "tir.merge_async_commit_queue_scope": False,
    }

    ctx = tvm.transform.PassContext(
        opt_level=3,
        config=config,
    )
    sch_rules = sch_rules_async

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=20000,
            max_trials_per_task=16,
            num_trials_per_iter=16,
            strategy="replay-trace",
            builder=get_hexagon_local_builder(ctx),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
            space=ms.space_generator.PostOrderApply(
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs={},
            ),
        )

        config.update(
            {
                "relay.backend.use_meta_schedule": True,
                "relay.backend.tir_converter": "default",
            }
        )

        return ms.relay_integration.compile_relay(
            database=database, mod=mod, target=target, params=params, pass_config=config
        )


@tvm.testing.requires_hexagon
def test_conv2d_relay_auto_schedule(hexagon_launcher):
    """Test conv2d using auto schedule."""
    if hexagon_launcher.is_simulator():
        pytest.skip(msg="Tuning on simulator not supported.")

    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    i_size, o_size, h_size, w_size = 64, 64, 56, 56
    k_height_size = k_width_size = 3

    strides = (1, 1)
    padding = (0, 0)

    d_shape = (1, i_size, h_size, w_size)
    w_shape = (o_size, i_size, k_height_size, k_width_size)
    bias_shape = (1, o_size, 1, 1)

    data = relay.var("data", shape=d_shape, dtype="uint8")
    weight = relay.var("weight", shape=w_shape, dtype="uint8")
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=(k_height_size, k_width_size),
        channels=o_size,
        padding=padding,
        strides=strides,
        out_dtype="int32",
    )
    mod = tvm.IRModule.from_expr(conv2d)

    data_np = np.random.uniform(1, 10, size=d_shape).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=w_shape).astype("uint8")
    bias_np = np.random.uniform(1, 10, size=bias_shape).astype("int32")
    params = {"weight": weight_np, "bias": bias_np}

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight_np])
        .numpy()
    )

    lib = tune_vrmpy_auto_tensorize(mod, params, hexagon_launcher)

    with hexagon_launcher.create_session() as session:
        rt_mod = session.get_executor_from_factory(lib)
        rt_mod.set_input("data", data_np)
        rt_mod.run()

        out = rt_mod.get_output(0).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-4, rtol=1e-5)
