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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import add_rfactor
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.target import Target
from tvm.te.operation import create_prim_func


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    return ctx


def test_cpu_matmul():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)",
            "b8 = sch.rfactor(loop=l7, factor_axis=2)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l6, l7 = sch.split(loop=l3, factors=[v4, v5], preserve_unit_iters=True)",
            "b8 = sch.rfactor(loop=l6, factor_axis=2)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
        ],
    ]
    target = Target("llvm --num-cores=32")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=4,
                m=4,
                k=512,
            )
        ),
        target=target,
        rule=add_rfactor(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_cpu_matmul()
