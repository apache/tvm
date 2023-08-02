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
# pylint: disable=invalid-name,missing-module-docstring,missing-function-docstring,missing-class-docstring
from typing import List
import tempfile
import pytest

import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.schedule_rule import ApplyCustomRule, MultiLevelTiling
from tvm.meta_schedule.testing.space_generation import generate_design_space
from tvm.script import tir as T


def create_matmul(rule_name: str):
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                T.block_attr({"schedule_rule": rule_name})
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return func


@tvm.register_func("meta_schedule.cpu.test_apply_custom_rule")
def sch_fn(sch: tvm.tir.Schedule, block: tvm.tir.Block) -> List[tvm.tir.Schedule]:
    raise ValueError("Intended for meta_schedule.cpu.test_apply_custom_rule")


def test_custom_rule():
    with pytest.raises(ValueError) as e_info:
        with tempfile.TemporaryDirectory() as tmpdir:
            sch_rules = [ApplyCustomRule()]
            space_gen = ms.space_generator.PostOrderApply(sch_rules=sch_rules)
            ms.tune_tir(
                mod=create_matmul(rule_name="test_apply_custom_rule"),
                target="llvm -num-cores=1",
                work_dir=tmpdir,
                max_trials_global=10,
                space=space_gen,
            )
    assert "ValueError: Intended for meta_schedule.cpu.test_apply_custom_rule" in str(e_info.value)


def test_custom_rule_with_none():
    """Should ignore custom_rule and apply MultiLevelTiling"""
    schs = generate_design_space(
        "llvm",
        mod=create_matmul(rule_name="None"),
        target=tvm.target.Target("llvm -num-cores=1"),
        types=None,
        sch_rules=[ApplyCustomRule(), MultiLevelTiling("SSR")],
    )
    assert len(schs) == 1
    tiling_annotations = [
        inst
        for inst in schs[0].trace.insts
        if inst.kind.name == "Annotate" and inst.attrs[0] == "meta_schedule.tiling_structure"
    ]
    assert len(tiling_annotations) == 1, "Tiling rule was not applied"


if __name__ == "__main__":
    test_custom_rule()
