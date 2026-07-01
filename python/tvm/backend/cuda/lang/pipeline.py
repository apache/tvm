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
"""Reusable pipeline state and mbarrier helpers for SM100 kernels.

These classes emit TIR via @T.inline. Decorate with @T.meta_class so that
instances are automatically treated as meta values inside @T.prim_func.
"""

from tvm.script import tirx as T


@T.meta_class
class PipelineState:
    """Tracks stage and phase for a software-pipelined ring buffer.

    This class does not know anything about full/empty barriers. Use it when
    the kernel manually waits/signals barriers, or when the stage/phase drives
    a ring not wrapped in a ``Pipeline``.

    Parameters
    ----------
    depth : int
        Number of stages in the ring.
    phase : int, optional
        Initial phase. Omit when initialization should happen later.
    """

    def __init__(self, depth: int, phase=None):
        self.stage = T.local_scalar("int32")
        self.phase = T.local_scalar("int32")
        self.depth = depth
        if phase is not None:
            self.init(phase)

    @T.inline
    def init(self, phase):
        self.stage = 0
        self.phase = phase

    @T.inline
    def advance(self):
        if self.depth > 1:
            self.stage = self.stage + 1
            if self.stage == self.depth:
                self.stage = 0
                self.phase = self.phase ^ 1
        else:
            self.phase = self.phase ^ 1


@T.meta_class
class MBarrier:
    """Mbarrier wrapper with regular ``mbarrier.arrive``.

    Parameters
    ----------
    pool : SMEMPool
        Shared memory pool allocator.
    depth : int
        Number of barrier slots (one per pipeline stage).
    phase_offset : int
        XORed into the phase bit on every ``wait`` / ``arrive``.
    leader : Expr, optional
        Boolean predicate selecting the single thread that runs
        ``mbarrier.init``. Defaults to ``T.cuda.thread_rank() == 0`` --
        thread 0 of the enclosing CTA, which always picks exactly one
        thread regardless of which scope_id vars the caller declared.
        Override only when you want a different CTA-local thread to do
        the init.

        Note: the default deliberately avoids ``T.warp_id()`` /
        ``T.lane_id()``. Those introduce deferred ``cta->warp`` /
        ``warp->thread`` ScopeIdDefs that the verifier cannot pin down
        unless the kernel header declares the full warp/lane chain (e.g. a
        single-CTA DSMEM kernel that only declares ``thread_id``). It also
        avoids the synccheck false-deadlock on kernels that declare a
        second warp-scope id. The generated CUDA is equivalent.
    """

    def __init__(self, pool, depth, phase_offset=0, leader=None):
        self.buf = pool.alloc((depth,), "uint64", align=8)
        self.depth = depth
        self.phase_offset = phase_offset
        self.leader = leader if leader is not None else (T.cuda.thread_rank() == 0)

    @T.inline
    def init(self, count):
        if self.leader:
            for i in T.unroll(self.depth):
                T.ptx.mbarrier.init(self.buf.ptr_to([i]), count)

    @T.inline
    def wait(self, stage, phase):
        # Blocks: ``mbarrier.try_wait`` loops internally until the phase flips,
        # so this returns only once the barrier has completed.
        T.ptx.mbarrier.try_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset)

    @T.inline
    def arrive(self, stage, cta_id=None, pred=None, count=None):
        # Default: local-CTA arrive — emits the simple
        # ``mbarrier.arrive.shared.b64`` form. To arrive on a remote
        # CTA's mbarrier in a cluster kernel, callers must pass
        # ``cta_id=`` explicitly (e.g. ``bar.arrive(stage, cta_id=0)``)
        # or use ``MBarrier.remote_view(rank).arrive(stage)``. Defaulting
        # the cross-CTA path was both surprising (``bar.arrive(stage)``
        # silently ``mapa`` ed across the cluster) and a per-call cost
        # of ~3 PTX ops on every single-CTA kernel.
        #
        # ``count`` (cross-CTA path only) emits the explicit arrival-count
        # operand, i.e. ``mbarrier.arrive.shared::cluster.b64 _, [addr], count``.
        # When ``None`` the implicit count-of-1 form is emitted. Passing
        # ``count=1`` is semantically identical but spells the count explicitly.
        if cta_id is None:
            T.ptx.mbarrier.arrive(self.buf.ptr_to([stage]))
        else:
            actual_pred = True if pred is None else pred
            T.ptx.mbarrier.arrive(
                self.buf.ptr_to([stage]), cta_id=cta_id, pred=actual_pred, count=count
            )

    def ptr_to(self, idx):
        return self.buf.ptr_to(idx)

    def remote_view(self, rank):
        """Create a view of this barrier mapped to another CTA's shared memory.

        Arrive-only: the returned view is built with ``object.__new__`` and
        never copies ``self.leader``, so calling ``.init()`` on it would fail.
        Use it solely to ``arrive`` on a remote CTA's mbarrier.
        """
        from tvm.ir import PointerType, PrimType
        from tvm.tirx import Var as TIRVar

        expr = T.reinterpret("handle", T.ptx.map_shared_rank(self.buf.ptr_to([0]), rank))
        ptr = TIRVar("remote_mbar_ptr", PointerType(PrimType("uint64")))
        T.Bind(expr, var=ptr)
        buf = T.decl_buffer([self.depth], "uint64", data=ptr, scope="shared")
        remote = object.__new__(type(self))
        remote.buf = buf
        remote.depth = self.depth
        remote.phase_offset = self.phase_offset
        return remote


class TMABar(MBarrier):
    """Barrier signaled by TMA (mbarrier.arrive.expect_tx).

    When ``tx_count`` is None, falls back to a remote mbarrier.arrive
    (matching MBarrier.arrive defaults).
    """

    @T.inline
    def arrive(self, stage, tx_count=None, cta_id=None, pred=None):
        # NOTE: this arrive() kwarg set intentionally differs from
        # MBarrier.arrive (hardware necessity, LSP-incompatible by design).
        # ``tx_count``: TMA byte count for ``mbarrier.arrive.expect_tx``.
        # ``cta_id`` / ``pred``: forwarded to the underlying
        # ``mbarrier.arrive`` (cluster path) when set; otherwise the
        # arrive is local-CTA only. See ``MBarrier.arrive`` for the
        # full default-local rationale.
        if tx_count is not None:
            T.ptx.mbarrier.arrive.expect_tx(self.buf.ptr_to([stage]), tx_count)
        elif cta_id is None:
            T.ptx.mbarrier.arrive(self.buf.ptr_to([stage]))
        else:
            actual_pred = True if pred is None else pred
            T.ptx.mbarrier.arrive(self.buf.ptr_to([stage]), cta_id=cta_id, pred=actual_pred)


class TCGen05Bar(MBarrier):
    """Barrier signaled by ``tcgen05`` commit.

    The caller is responsible for ensuring only one thread issues the
    commit, e.g. by wrapping the call in ``if T.ptx.elect_sync():``.
    """

    @T.inline
    def arrive(self, stage, cta_group=1, cta_mask=None):
        # NOTE: this arrive() kwarg set intentionally differs from
        # MBarrier.arrive (hardware necessity, LSP-incompatible by design).
        if cta_mask is None and cta_group == 1:
            T.ptx.tcgen05.commit(self.buf.ptr_to([stage]))
        else:
            T.ptx.tcgen05.commit(self.buf.ptr_to([stage]), cta_group=cta_group, cta_mask=cta_mask)


# Barrier-type tags accepted by Pipeline's ``full=`` / ``empty=`` arguments.
_BAR_KINDS = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}


@T.meta_class
class Pipeline:
    """A full/empty mbarrier pair for a software-pipelined data flow.

    Pass barrier-type tags and ``Pipeline`` constructs and ``init``\\ s the
    barriers itself. Tags: ``"tma"`` (TMABar), ``"tcgen05"`` (TCGen05Bar),
    ``"mbar"`` (MBarrier). The barrier type and arrival count of each event
    stay explicit at the call site -- e.g. ``Pipeline(pool, n, full="tma",
    empty="tcgen05", init_empty=NUM_CONSUMER)``.

    Both signals are required: a ``Pipeline`` is a *pair*. For a one-way event
    (a pure "X happened" signal with no slot to recycle) use a bare barrier
    (``TMABar``/``TCGen05Bar``/``MBarrier``) directly -- it has no empty side.

    Parameters
    ----------
    pool : SMEMPool
        Shared memory pool allocator.
    stages : int
        Number of pipeline stages (barrier slots).
    full, empty : str
        Barrier-type tag for the full / empty signal (see above).
    init_full, init_empty : int
        Expected arrival count for the full / empty barrier.
    empty_phase_offset : int
        XORed into the empty barrier's phase bit on every wait / arrive.
    leader : Expr, optional
        Propagated to both barriers; defaults to thread 0 of the CTA.
    """

    def __init__(
        self,
        pool,
        stages,
        *,
        full,
        empty,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        self.stages = stages
        self.full = _BAR_KINDS[full](pool, stages, leader=leader)
        self.full.init(init_full)
        self.empty = _BAR_KINDS[empty](pool, stages, phase_offset=empty_phase_offset, leader=leader)
        self.empty.init(init_empty)
