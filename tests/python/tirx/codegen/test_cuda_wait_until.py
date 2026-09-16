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
"""Codegen contract for the declared wait.

`wait_until` is the only operation of the set that survives. The four
direct forms were removed: they emitted exactly what their raw spellings do, so
they bought nothing at codegen, and a protocol's word is now recognized by the
address the wait names rather than by each access carrying an identity. A
publisher is spelled `red`/`atom`/`st` directly, as these tests do.

What the wait itself generates is a loop, and the shape of that loop is the
contract asserted here: it polls relaxed, closes an acquiring wait with one
`ld.acquire`, peels the first poll, and never lets the closing read reach the
caller's destination. The protocols exercised are the shapes real kernels use.
"""

import pytest

import tvm
from tvm.script import tirx as T


def build(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")
    return getattr(mod, "mod", mod).imports[0].inspect_source("")


def rendezvous(backoff_ns=None, ptx_type=None):
    """An N-way barrier on a monotone counter, as radix_topk_multi_cta writes it."""

    @T.prim_func
    def kernel(state: T.Buffer((1,), "int32"), participants: T.int32):
        T.device_entry()
        T.cta_id([2])
        lane = T.thread_id([32])
        spin = T.alloc_local((1,), "int32")
        phase = T.alloc_local((1,), "int32")
        if lane == 0:
            phase[0] = T.int32(0)
            spin[0] = T.int32(0)
            T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
            T.cuda.wait_until(
                spin[0],
                state.ptr_to([0]),
                predicate=lambda v: v >= (phase[0] + T.int32(1)) * participants,
                scope="gpu",
                ptx_type=ptx_type,
                backoff_ns=backoff_ns,
            )

    return kernel


def packed_contribution():
    """Counter in the high half, payload in the low half: DeepEP's notify slot."""

    @T.prim_func
    def kernel(slot: T.Buffer((1,), "uint64"), out: T.Buffer((1,), "uint64"), n: T.int32):
        T.device_entry()
        T.cta_id([2])
        lane = T.thread_id([32])
        observed = T.alloc_local((1,), "uint64")
        if lane == 0:
            T.ptx.red.relaxed.gpu.global_.add.u64(
                slot.ptr_to([0]),
                T.bitwise_or(T.shift_left(T.uint64(1), T.uint64(32)), T.uint64(7)),
            )
            observed[0] = T.uint64(0)
            T.cuda.wait_until(
                observed[0],
                slot.ptr_to([0]),
                predicate=lambda v: T.shift_right(v, T.uint64(32)) == T.Cast("uint64", n),
                scope="gpu",
            )
            out[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))

    return kernel


def _wait_macro(source):
    return next(
        line for line in source.splitlines() if line.startswith("#define") and "wait_until" in line
    )


# =============================================================================
# The emitted loop
# =============================================================================


def test_the_wait_is_a_loop_beside_a_raw_publisher():
    """The publisher is ordinary PTX; only the wait generates something new."""

    source = build(rendezvous())
    assert "red.release.gpu.global.add.s32" in source
    assert "while (!(" in source


def test_a_wait_loads_before_it_tests():
    """`do { ld } while (!done)`, the way every spin here is written by hand.

    A pre-tested loop would need the caller to seed the destination, and the
    only honest seed is another load of the same word -- one more unguaranteed
    read, in the kernel, for a checker to have an opinion about. Loading first
    needs none.
    """

    macro = _wait_macro(build(rendezvous()))
    assert "do { tvm_builtin_cuda_wait_until" in macro
    assert macro.index("_load(") < macro.index("while (!(")


def test_an_acquiring_wait_polls_relaxed_and_closes_with_one_acquire():
    """The poll carries no ordering; a single closing `ld.acquire` takes the edge.

    Measured against the acquiring poll over every benchmarkable kernel that
    owns a spin wait, interleaved, round 1 dropped: -1.04% on
    `sm100_fp8_fp4_mega_moe` (8 wait sites), -0.50% on `radix_topk_multi_cta`,
    and within 0.3% on the other three. Paying acquire semantics on every poll
    buys nothing -- only the last read decides what the waiter goes on to
    observe.
    """

    source = build(rendezvous())
    macro = _wait_macro(source)
    assert "ld.relaxed.gpu.global.s32" in source
    assert "ld.acquire.gpu.global.s32" in source
    # the loop polls relaxed, and the acquire is reached once, after it
    assert macro.count("_load(") == 2
    assert macro.count("_acquire(") == 1
    assert macro.index("while (!(") < macro.index("_acquire(")


def test_the_closing_acquire_cannot_overwrite_the_waited_value():
    """The edge read goes to a scratch, never to the caller's destination.

    It may observe a value later than the one the predicate accepted, and the
    predicates here are not all monotone -- `sm100_fp8_fp4_mega_moe`'s grid
    barrier tests a sign-bit flip and its ring waits test equality -- so letting
    it reach `dst` would hand the caller a value its own predicate rejects.

    The edge survives that: these words are published by `red`/`atom` release
    RMWs, so every contribution sits in one release sequence and an acquire
    reading any of them synchronizes with all the earlier ones.
    """

    macro = _wait_macro(build(rendezvous()))
    scratch = "__tirx_wait_edge"
    assert f"{scratch};" in macro
    assert f"_acquire({scratch}, (ptr))" in macro
    # every write to the caller's destination comes from a polling load
    assert "_acquire((dst)" not in macro


def test_every_wait_takes_the_edge():
    """There is one lowering, so a caller cannot ask for a wait worth less.

    A wait that consumes only its own exit value takes an edge it does not
    need. Both such waits in the kernel corpus were measured against this form
    and neither could tell the difference, so the mode they would have needed
    is not worth the thing it costs: a promise, made at the call site, that
    nothing the word guards is read afterwards -- which is a promise the call
    site cannot show.
    """

    macro = _wait_macro(build(rendezvous()))
    assert "_acquire(__tirx_wait_edge, (ptr))" in macro
    assert "ld.volatile" not in build(rendezvous())


def test_a_bit_typed_word_keeps_its_handwritten_ptx():
    """A kernel picks the PTX type per operation, not per word.

    `red.add` has no bit-typed form, while a load that only moves the value is
    ordinarily spelled `.b32`. Reproducing that split is what lets a migration
    leave the emitted PTX unchanged.
    """

    source = build(rendezvous(ptx_type="b32"))
    assert "red.release.gpu.global.add.s32" in source
    assert "ld.relaxed.gpu.global.b32" in source
    assert "ld.acquire.gpu.global.b32" in source
    # the default spelling must be gone, or the override did nothing
    assert "ld.acquire.gpu.global.s32" not in source


def test_a_packed_word_round_trips():
    source = build(packed_contribution())
    assert "red.relaxed.gpu.global.add.u64" in source
    assert "ld.relaxed.gpu.global.u64" in source


def test_a_backoff_sleeps_between_polls_and_never_around_them():
    """Sleep between polls, as the contended waits are written by hand: never
    before the first poll, never after the last.

    The first poll is peeled out of the loop, so the shape is
    `ld; if (!done) { while (1) { sleep; ld; if (done) break; } }`. That runs
    the same sequence a hand-written `while (1) { ld; if (done) break; sleep; }`
    runs -- load, test, sleep, load, test -- and the peel is what keeps the
    early exit a hand-written spin gets."""

    source = build(rendezvous(backoff_ns=40))
    macro = _wait_macro(source)
    assert "__nanosleep(backoff_ns)" in macro
    # The peeled poll runs before any sleep: nothing sleeps before polling once.
    assert macro.index("_load(") < macro.index("__nanosleep")
    # The peeled poll is guarded, so an already satisfied predicate never
    # reaches the loop at all.
    assert macro.index("_load(") < macro.index("if (!(predicate))")
    # Inside the loop the sleep precedes the poll it separates, and the break
    # follows that poll, so nothing sleeps after the last one.
    loop = macro[macro.index("while (1)") :]
    assert loop.index("__nanosleep") < loop.index("_load(")
    assert loop.index("_load(") < loop.index("if (predicate) break;")
    # The call site is what fixes the 40.
    assert "_backoff(" in source and ", 40)" in source

    # No backoff is the spelling a wait had before the field existed.
    plain = build(rendezvous())
    assert "__nanosleep" not in plain
    assert "_backoff" not in plain


def test_a_backoff_still_closes_the_acquire_after_the_last_poll():
    """The edge read belongs after the loop, never inside it."""

    macro = _wait_macro(build(rendezvous(backoff_ns=40)))
    assert macro.index("if (predicate) break;") < macro.index("_acquire(")


def test_a_wait_may_test_against_a_thread_local_scalar():
    """A barrier tests the counter against its loop-carried phase.

    The macro re-evaluates the predicate each iteration, and the loop body only
    loads, so the phase cannot move under the wait even though it is re-read.
    """

    source = build(rendezvous())
    assert "while (!(" in source
    assert "phase" in source


# =============================================================================
# What the wait refuses
# =============================================================================


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"order": "acquire"}, TypeError, "order"),
        ({"impl": "volatile"}, TypeError, "impl"),
        ({"scope": "warp"}, ValueError, "scope"),
    ],
)
def test_a_wait_rejects_attributes_it_cannot_mean(kwargs, error, message):
    """`order` and `impl` are gone, so a kernel still passing one is told.

    They named the two axes this wait was measured on. Both settled, and a
    silently accepted keyword would let a kernel written against the old
    spelling keep compiling while meaning something else.
    """

    from tvm.backend.cuda.op import cuda_wait_until

    with pytest.raises(error, match=message):
        cuda_wait_until(None, None, None, **kwargs)


def test_a_ptx_type_of_another_width_is_refused():
    with pytest.raises(Exception, match="64 bits but the sync word is 32 bits"):
        build(rendezvous(ptx_type="b64"))


def test_a_wide_word_cannot_be_waited_on():
    """A predicate tests one scalar, and a 16-byte exit value is not one.

    The wait refuses the width rather than testing half of it. The refusal lands
    while the function is traced, so the definition is what is guarded here.
    """

    with pytest.raises(Exception, match="does not take a 128-bit word"):

        @T.prim_func
        def kernel(response: T.Buffer((2,), "uint64")):
            T.device_entry()
            T.cta_id([1])
            lane = T.thread_id([32])
            seen = T.local_scalar("uint64")
            if lane == 0:
                seen = T.uint64(0)
                T.cuda.wait_until(
                    seen,
                    response.ptr_to([0]),
                    seen != T.uint64(0),
                    scope="gpu",
                    ptx_type="b128",
                )


def test_a_declared_word_is_global_and_says_where_a_shared_wait_belongs():
    """Shared memory is out of scope by design, not by omission.

    A protocol that waits within a CTA or a cluster has `mbarrier`, which is
    the hardware's primitive for it and which the checker models by generation.
    A polled flag in shared memory would be a worse spelling of the same thing,
    so the refusal points at the primitive that does belong there.
    """

    with pytest.raises(ValueError, match="mbarrier"):
        T.cuda.wait_until(None, None, None, scope="cta", space="shared", ptx_type="b32")


def test_the_four_direct_forms_are_gone():
    """Only the wait remains: a publisher is spelled in raw PTX.

    The removed forms emitted exactly what their raw spellings do, so they never
    changed codegen; the protocol's word is recognized by the address the wait
    names instead.
    """

    retired = (
        "atomic_ref_store",
        "atomic_ref_add",
        "atomic_ref_fetch_add",
        "atomic_ref_load",
        # The surviving wait kept the family's name until it was the only
        # member left; `atomic` claimed an atomicity its scoped loads never
        # had, and `ref` grouped a family of one.
        "atomic_ref_wait",
    )
    assert [name for name in retired if hasattr(T.cuda, name)] == []
