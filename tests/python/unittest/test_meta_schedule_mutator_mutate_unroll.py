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
from typing import List

from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule

# pylint: disable=invalid-name, no-member


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [512, 512])
    B = T.match_buffer(b, [512, 512])
    C = T.match_buffer(c, [512, 512])
    for i, j, k in T.grid(512, 512, 512):  # type: ignore
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # type: ignore
            with T.init():
                C[vi, vj] = 0.0  # type: ignore
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=invalid-name, no-member


def _sch(decisions: List[List[int]]) -> Schedule:
    sch = Schedule(matmul, debug_mask="all")
    # pylint: disable=invalid-name
    d0, d1, d2 = decisions
    b0 = sch.get_block(name="C", func_name="main")
    root = sch.get_block(name="root", func_name="main")
    sch.get_consumers(block=b0)
    b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6, v7, v8 = sch.sample_perfect_tile(
        loop=l2,
        n=4,
        max_innermost_factor=64,
        decision=d0,
    )
    l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])
    v13, v14, v15, v16 = sch.sample_perfect_tile(
        loop=l3,
        n=4,
        max_innermost_factor=64,
        decision=d1,
    )
    l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])
    v21, v22 = sch.sample_perfect_tile(
        loop=l4,
        n=2,
        max_innermost_factor=64,
        decision=d2,
    )
    l23, l24 = sch.split(loop=l4, factors=[v21, v22])
    sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
    sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=True)
    v57 = sch.sample_categorical(
        candidates=[0, 16, 64, 512],
        probs=[0.25, 0.25, 0.25, 0.25],
        decision=0,
    )
    sch.annotate(block_or_loop=root, ann_key="meta_schedule.unroll_explicit", ann_val=v57)
    # pylint: enable=invalid-name
    return sch


def _make_mutator(target: Target) -> ms.Mutator:
    ctx = ms.TuneContext(
        mod=matmul,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateUnroll(): 1.0},
        ),
    )
    return list(ctx.space_generator.mutator_probs.keys())[0]


def test_mutate_unroll_matmul():
    mutator = _make_mutator(target=Target("llvm --num-cores=16"))
    sch = _sch(
        decisions=[
            [4, 32, 4, 1],
            [8, 4, 8, 2],
            [512, 1],
        ],
    )
    results = set()
    for _ in range(100):
        trace = mutator.apply(sch.trace)
        decision = trace.decisions[trace.insts[-2]]
        results.add(decision)
        if len(results) == 3:
            break
    assert len(results) == 3
    assert results == {1, 2, 3}


if __name__ == """__main__""":
    test_mutate_unroll_matmul()
