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
from typing import Optional

import numpy as np  # type: ignore
import pytest
import tvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm._ffi import register_func
from tvm.tir.schedule import BlockRV, Schedule
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


def _schedule_dense(m: Optional[int], do_tune: bool):
    """Manually schedule a dense block, created from TE compute op via CreatePrimFunc,
    using VNNI instruction.
    """

    def schedule_fn(sch, dense_block: Optional[BlockRV] = None) -> bool:
        if "dense" not in sch.mod.attrs["task_name"]:
            return False
        if dense_block is None:
            dense_block = sch.get_block("compute")
            assert "dense_vnni" in sch.get(dense_block).annotations["schedule_rule"]

        post_blocks = sch.get_consumers(dense_block)
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
                    a_y, a_x = sch.get_loops(outer_block)[-2:]
                    break
                post_blocks = next_post_blocks
        else:
            a_y, a_x, _ = sch.get_loops(dense_block)[-3:]
            outer_block = dense_block
        if do_tune:
            y_factors = sch.sample_perfect_tile(a_y, n=2, max_innermost_factor=128)
            a_yo, a_yi = sch.split(a_y, factors=y_factors)
        else:
            a_yo, a_yi = sch.split(a_y, factors=[None, min(m, 64)])
        a_xo, a_xi = sch.split(a_x, factors=[None, 16])
        sch.reorder(a_yo, a_xo, a_yi, a_xi)
        fused = sch.fuse(a_yo, a_xo)
        if outer_block != dense_block:
            # Handle the case when dense is fused with post ops.
            sch.vectorize(a_xi)
            sch.compute_at(dense_block, a_yi)
        a_xi, a_k = sch.get_loops(dense_block)[-2:]
        a_ko, a_ki = sch.split(a_k, factors=[None, 4])
        sch.reorder(a_ko, a_xi, a_ki)
        # We need to parallelize before decompose_reduction, otherwise the so-called "Compact dataflow"
        # condition is violated.
        sch.parallel(fused)
        dec = sch.decompose_reduction(dense_block, a_ko)
        init_loop = sch.get_loops(dec)[-1]
        sch.vectorize(init_loop)
        sch.tensorize(a_xi, VNNI_INTRIN)
        return True

    return schedule_fn


def _relay_dense(m, n, k):
    data = relay.var("data", shape=(m, k), dtype="uint8")
    weight = relay.var("weight", shape=(n, k), dtype="int8")
    bias = relay.var("bias", shape=(n,), dtype="int32")
    # dense is tuned by the TIR schedule above, bmm is scheduled by TE (topi/x86/batch_matmul.py)
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype="int32")
    out = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(bias_add, 0), "uint8"),
        relay.cast(relay.expand_dims(bias_add, 0), "int8"),
        out_dtype="int32",
    )
    relay_mod = tvm.IRModule.from_expr(out)
    data = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
    params = {
        "weight": np.random.uniform(1, 10, size=(n, k)).astype("int8"),
        "bias": np.random.uniform(1, 10, size=(n,)).astype("int32"),
    }

    def f_check(lib, dev):
        ref = (
            relay.create_executor(
                "vm",
                mod=relay_mod,
                device=dev,
                target="llvm",
            )
            .evaluate()(data, params["weight"], params["bias"])
            .numpy()
        )
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        runtime.set_input("data", data)
        runtime.run()
        out = runtime.get_output(0).numpy()
        np.testing.assert_equal(out, ref)

    return relay_mod, params, f_check


@tvm.testing.requires_cascadelake
def test_vnni_schedule_fn_database():
    m, n, k = 1024, 1024, 1024
    target = tvm.target.Target("llvm -mcpu=cascadelake -num-cores 4")
    dev = tvm.cpu(0)
    relay_mod, params, f_check = _relay_dense(m, n, k)

    with ms.database.ScheduleFnDatabase(
        _schedule_dense(
            m=m,
            do_tune=False,
        )
    ), tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        # pylint: disable=W0105
        """The log should say
        Warning: Cannot find workload: tvmgen_default_fused_expand_dims
        Warning: Cannot find workload: tvmgen_default_fused_cast
        Warning: Cannot find workload: tvmgen_default_fused_cast_1
        Warning: Cannot find workload: tvmgen_default_fused_nn_batch_matmul

        This means batch matmul and others are scheduled by TE, and dense (the one not warned)
        is found in the meta schedule tuning database during compilation
        """
        # pylint: enable=W0105
        lib = relay.build(relay_mod, target=target, params=params)
    f_check(lib, dev)


@tvm.testing.requires_cascadelake
def test_vnni_schedule_fn_tune():
    # pylint: disable=W0105
    """
    We can inject and apply a custom TIR scheduling to a TE compute of interest, using
    the "schedule_rule" annotation. For example, in topi/x86/dense.py we have the following
    declaration for int8 dense targeting the VNNI instruction.

    C = te.compute(
        ...
        attrs={"schedule_rule": "meta_schedule.dense_vnni"},
    )

    When the MetaSchedule encounters a TensorIR block with the "schedule_rule" annotation,
    it looks up the packed func registry for a function that is associated with the given schedule
    rule key ("meta_schedule.dense_vnni" in this example). The signature of such custom schedule
    functions must be

       (tir.schedule.Schedule, tir.schedule.BlockRV) -> [tir.schedule.Schedule].

    The BlockRV argument corresponds to the TE compute annotated with "schedule_rule".

    The relevant code is in meta_schedule/space_generator/post_order_apply.cc.
    """

    def schedule_rule_dense_vnni(sch: Schedule, dense_block: BlockRV):
        _schedule_dense(m=None, do_tune=True)(sch, dense_block)
        return [sch]

    register_func("meta_schedule.dense_vnni", schedule_rule_dense_vnni)

    m, n, k = 1024, 1024, 1024
    target = tvm.target.Target("llvm -mcpu=cascadelake -num-cores 4")
    dev = tvm.cpu(0)
    relay_mod, params, f_check = _relay_dense(m, n, k)

    extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
    with tempfile.TemporaryDirectory() as work_dir:
        # postprocs=lambda: [] is important to prevent default post processors from
        # tampering with the manual schedule.
        tasks = ms.relay_integration.extracted_tasks_to_tune_contexts(
            list(
                filter(
                    lambda task: "dense" in task.task_name,
                    extracted_tasks,
                )
            ),
            work_dir=work_dir,
            space=ms.space_generator.PostOrderApply(
                f_block_filter=None,
                sch_rules=None,
                postprocs=[],
                mutator_probs=None,
            ),
        )
        database = ms.relay_integration.tune_tasks(
            tasks=tasks,
            task_weights=[1.0] * len(tasks),
            work_dir=work_dir,
            max_trials_global=20000,
        )
    with database, tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        # pylint: disable=W0105
        """The log should say
        Warning: Cannot find workload: tvmgen_default_fused_expand_dims
        Warning: Cannot find workload: tvmgen_default_fused_cast
        Warning: Cannot find workload: tvmgen_default_fused_cast_1
        Warning: Cannot find workload: tvmgen_default_fused_nn_batch_matmul

        This means batch matmul and others are scheduled by TE, and dense (the one not warned)
        is found in the meta schedule tuning database during compilation
        """
        # pylint: enable=W0105
        lib = relay.build(relay_mod, target=target, params=params)
    f_check(lib, dev)


if __name__ == """__main__""":
    test_vnni_schedule_fn_database()
    test_vnni_schedule_fn_tune()
