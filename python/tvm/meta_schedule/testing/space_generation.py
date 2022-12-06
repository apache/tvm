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
from typing import List, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on

from tvm import meta_schedule as ms
from tvm.ir import IRModule, structural_equal
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import Trace
from tvm.tir.schedule.testing import verify_trace_roundtrip


def get_rules(
    kind: Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"],
    types: Union[type, Tuple[type, ...]],
) -> List[ms.ScheduleRule]:
    """Get default schedule rules"""
    rules = ms.ScheduleRule.create(kind)
    return [rule for rule in rules if isinstance(rule, types)]


def generate_design_space(
    kind: Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"],
    mod: IRModule,
    target: Target,
    types: Union[type, Tuple[type, ...]],
    sch_rules: Optional[List[ms.ScheduleRule]] = None,
) -> List[Schedule]:
    if sch_rules is None:
        sch_rules = get_rules(kind, types)
    else:
        assert types is None
    return ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=[],
            mutator_probs={},
        ),
        task_name="test",
    ).generate_design_space()


def _find_match_sketch_id(
    mod: IRModule,
    sketches: List[Schedule],
    expected_mod: IRModule,
    expected_decision: List[Tuple[str, List[int]]],
    *,
    debug_mask="all",
) -> Optional[int]:
    for sketch_id, sketch in enumerate(sketches):
        i = 0
        new_decisions = {}
        for inst in sketch.trace.insts:
            if not inst.kind.name.startswith("Sample"):
                continue
            assert i < len(expected_decision)
            if inst.kind.name == expected_decision[i][0]:
                new_decisions[inst] = expected_decision[i][1]
                i += 1
        if len(new_decisions) != len(expected_decision):
            continue
        sch = Schedule(mod, debug_mask=debug_mask)
        Trace(
            insts=sketch.trace.insts,
            decisions=new_decisions,
        ).apply_to_schedule(sch, remove_postproc=True)
        if structural_equal(sch.mod, expected_mod):
            verify_trace_roundtrip(sch=sch, mod=mod, debug_mask=debug_mask)
            return sketch_id
    return None


def check_sketches(
    mod: IRModule,
    sketches: List[Schedule],
    expected_mods: List[IRModule],
    expected_decisions: List[List[Tuple[str, List[int]]]],
    *,
    debug_mask="all",
):
    assert len(expected_mods) == len(expected_decisions)
    assert len(sketches) == len(expected_mods)
    expected_mods = [
        IRModule({"main": m}) if not isinstance(m, IRModule) else m for m in expected_mods
    ]
    sketches = list(sketches)
    for expected_id, (expected_mod, expected_decision) in enumerate(
        zip(expected_mods, expected_decisions)
    ):
        sketch_id = _find_match_sketch_id(
            mod,
            sketches,
            expected_mod,
            expected_decision,
            debug_mask=debug_mask,
        )
        if sketch_id is None:
            raise AssertionError(
                f"Expected sketch #{expected_id} doesn't exist in the generated sketches."
            )
        sketches.pop(sketch_id)


def print_sketches(sketches: List[Schedule]):
    for i, sch in enumerate(sketches):
        print(f"###### {i}")
        sch.mod.show()
        for inst in sch.trace.insts:
            if inst in sch.trace.decisions:
                print(f'("{inst.kind.name}", {sch.trace.decisions[inst]}),')
