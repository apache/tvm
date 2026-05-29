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

These classes emit TIR via @Tx.inline. Decorate with @Tx.meta_class so that
instances are automatically treated as meta values inside @Tx.prim_func.
"""

from tvm.script import tirx as Tx


@Tx.meta_class
class RingState:
    """Tracks stage and phase for a software-pipelined ring buffer.

    This class does not know anything about full/empty barriers. Use it when
    the kernel manually waits/signals barriers, or when the stage/phase drives
    a non-``Pipe`` ring.

    Parameters
    ----------
    depth : int
        Number of stages in the ring.
    phase : int, optional
        Initial phase. Omit when initialization should happen later.
    """

    def __init__(self, depth: int, phase=None):
        self.stage = Tx.local_scalar("int32")
        self.phase = Tx.local_scalar("int32")
        self.depth = depth
        if phase is not None:
            self.init(phase)

    @Tx.inline
    def init(self, phase):
        self.stage = 0
        self.phase = phase

    @Tx.inline
    def advance(self):
        if self.depth > 1:
            self.stage = self.stage + 1
            if self.stage == self.depth:
                self.stage = 0
                self.phase = self.phase ^ 1
        else:
            self.phase = self.phase ^ 1


@Tx.meta_class
class _PipeEndpoint:
    """Standard producer or consumer endpoint for a Pipe."""

    def __init__(self, pipe, is_producer):
        self.pipe = pipe
        self.is_producer = is_producer
        self.state = RingState(pipe.stages, 1 if is_producer else 0)

    @property
    def stage(self):
        return self.state.stage

    @property
    def phase(self):
        return self.state.phase

    @Tx.inline
    def wait(self):
        """Producer: wait for empty slot. Consumer: wait for full data."""
        if self.is_producer:
            self.pipe.empty.wait(self.stage, self.phase)
        else:
            self.pipe.full.wait(self.stage, self.phase)

    @Tx.inline
    def signal(self, **kwargs):
        """Producer: signal full. Consumer: signal empty."""
        if self.is_producer:
            self.pipe.full.arrive(self.stage, **kwargs)
        else:
            self.pipe.empty.arrive(self.stage, **kwargs)

    @Tx.inline
    def advance(self):
        """Move to the next pipeline stage."""
        self.state.advance()

    def snapshot(self):
        """Freeze current (stage, phase) for deferred use."""
        return (self.stage, self.phase)


@Tx.meta_class
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
    leader : PrimExpr, optional
        Boolean predicate selecting the single thread that runs
        ``mbarrier.init``. Defaults to ``Tx.cuda.thread_rank() == 0`` --
        thread 0 of the enclosing CTA, which always picks exactly one
        thread regardless of which scope_id vars the caller declared.
        Override only when you want a different CTA-local thread to do
        the init.
    """

    def __init__(self, pool, depth, phase_offset=0, leader=None):
        self.buf = pool.alloc((depth,), "uint64", align=8)
        self.depth = depth
        self.phase_offset = phase_offset
        self.leader = leader if leader is not None else (Tx.cuda.thread_rank() == 0)

    @Tx.inline
    def init(self, count):
        if self.leader:
            for i in Tx.unroll(self.depth):
                Tx.ptx.mbarrier.init(self.buf.ptr_to([i]), count)

    @Tx.inline
    def wait(self, stage, phase):
        Tx.ptx.mbarrier.try_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset)

    @Tx.inline
    def arrive(self, stage, cta_id=None, pred=None):
        # Default: local-CTA arrive — emits the simple
        # ``mbarrier.arrive.shared.b64`` form. To arrive on a remote
        # CTA's mbarrier in a cluster kernel, callers must pass
        # ``cta_id=`` explicitly (e.g. ``bar.arrive(stage, cta_id=0)``)
        # or use ``MBarrier.remote_view(rank).arrive(stage)``. Defaulting
        # the cross-CTA path was both surprising (``bar.arrive(stage)``
        # silently ``mapa`` ed across the cluster) and a per-call cost
        # of ~3 PTX ops on every single-CTA kernel.
        if cta_id is None:
            Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]))
        else:
            actual_pred = True if pred is None else pred
            Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]), cta_id=cta_id, pred=actual_pred)

    def ptr_to(self, idx):
        return self.buf.ptr_to(idx)

    def remote_view(self, rank):
        """Create a view of this barrier mapped to another CTA's shared memory."""
        from tvm.ir import PointerType, PrimType
        from tvm.tirx import Var as TIRVar

        expr = Tx.reinterpret("handle", Tx.ptx.map_shared_rank(self.buf.ptr_to([0]), rank))
        ptr = TIRVar("remote_mbar_ptr", PointerType(PrimType("uint64")))
        Tx.Bind(expr, var=ptr)
        buf = Tx.decl_buffer([self.depth], "uint64", data=ptr, scope="shared")
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

    @Tx.inline
    def arrive(self, stage, tx_count=None, cta_id=None, pred=None):
        # ``tx_count``: TMA byte count for ``mbarrier.arrive.expect_tx``.
        # ``cta_id`` / ``pred``: forwarded to the underlying
        # ``mbarrier.arrive`` (cluster path) when set; otherwise the
        # arrive is local-CTA only. See ``MBarrier.arrive`` for the
        # full default-local rationale.
        if tx_count is not None:
            Tx.ptx.mbarrier.arrive.expect_tx(self.buf.ptr_to([stage]), tx_count)
        elif cta_id is None:
            Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]))
        else:
            actual_pred = True if pred is None else pred
            Tx.ptx.mbarrier.arrive(self.buf.ptr_to([stage]), cta_id=cta_id, pred=actual_pred)


class TCGen05Bar(MBarrier):
    """Barrier signaled by ``tcgen05`` commit.

    The caller is responsible for ensuring only one thread issues the
    commit, e.g. by wrapping the call in ``if Tx.ptx.elect_sync():``.
    """

    @Tx.inline
    def arrive(self, stage, cta_group=1, cta_mask=None):
        if cta_mask is None and cta_group == 1:
            Tx.ptx.tcgen05.commit(self.buf.ptr_to([stage]))
        else:
            Tx.ptx.tcgen05.commit(self.buf.ptr_to([stage]), cta_group=cta_group, cta_mask=cta_mask)


@Tx.meta_class
class Pipe:
    """Full+empty barrier pair for a software-pipelined data flow.

    Wraps a full barrier (signaled when data is ready) and an optional
    empty barrier (signaled when a slot is consumed) into a single object.
    Provides factory methods for common barrier type combinations.

    Parameters
    ----------
    pool : SMEMPool
        Shared memory pool allocator.
    stages : int
        Number of pipeline stages (barrier slots).
    full_type : type
        Barrier class for the full signal (TMABar, TCGen05Bar, or MBarrier).
    empty_type : type or None
        Barrier class for the empty signal, or None for one-way pipes.
    init_full : int
        Expected arrival count for the full barrier.
    init_empty : int or None
        Expected arrival count for the empty barrier.
    leader : PrimExpr, optional
        Propagated to the underlying MBarrier / TMABar / TCGen05Bar.
        Defaults to ``Tx.cuda.thread_rank() == 0`` when omitted.
    """

    def __init__(
        self,
        pool,
        stages,
        *,
        full_type=MBarrier,
        empty_type=None,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        self.full = full_type(pool, stages, leader=leader)
        if empty_type is not None:
            self.empty = empty_type(pool, stages, phase_offset=empty_phase_offset, leader=leader)
        else:
            self.empty = None
        self.stages = stages
        self.full.init(init_full)
        if self.empty is not None:
            self.empty.init(init_empty)

    @classmethod
    def tma(cls, pool, stages, *, empty_count=1, empty_phase_offset=0, leader=None):
        """TMA -> consumer: full=TMABar, empty=TCGen05Bar."""
        return cls(
            pool,
            stages,
            full_type=TMABar,
            empty_type=TCGen05Bar,
            init_full=1,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            leader=leader,
        )

    @classmethod
    def tcgen05(cls, pool, stages, *, empty_count=None, empty_phase_offset=0, leader=None):
        """TCGen05 -> consumer: full=TCGen05Bar, empty=MBarrier (if empty_count given)."""
        return cls(
            pool,
            stages,
            full_type=TCGen05Bar,
            empty_type=MBarrier if empty_count is not None else None,
            init_full=1,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            leader=leader,
        )

    @classmethod
    def mbar(cls, pool, stages, *, full_count, empty_count=None, empty_phase_offset=0, leader=None):
        """Thread -> thread: full=MBarrier, empty=MBarrier (if empty_count given)."""
        return cls(
            pool,
            stages,
            full_type=MBarrier,
            empty_type=MBarrier if empty_count is not None else None,
            init_full=full_count,
            init_empty=empty_count,
            empty_phase_offset=empty_phase_offset,
            leader=leader,
        )

    def producer(self):
        """Create a standard producer endpoint for this pipe."""
        return _PipeEndpoint(self, is_producer=True)

    def consumer(self):
        """Create a standard consumer endpoint for this pipe."""
        return _PipeEndpoint(self, is_producer=False)
