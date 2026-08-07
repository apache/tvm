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
from tvm.tirx import IntImm as _IntImm


def _tcgen05_commit_is_unicast(cta_mask):
    """Trace-time choice of the commit form, as the legacy wrapper made it.

    A compile-time mask arriving at most one CTA means the mbar address
    already names the target, so the unicast line (no mask operand) is used.
    A runtime mask is fine -- ctaMask is a register operand -- but the choice
    of *form* cannot depend on it, so a runtime mask always multicasts.

    Lives outside the inline body on purpose: the inline evaluator neither
    short-circuits conditional expressions nor accepts a None binding, so this
    decision needs real Python semantics.
    """
    if cta_mask is None:
        return True
    if isinstance(cta_mask, _IntImm):
        cta_mask = cta_mask.value
    return isinstance(cta_mask, int) and bin(cta_mask).count("1") <= 1


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


def _map_addr_into_cta(ptr, rank):
    """The 32-bit shared-window address of ``ptr`` as another CTA sees it.

    ``mapa.u32`` maps window addresses, so this is one cvta plus one mapa --
    what the fused legacy wrapper emitted. ``mapa.u64`` would map the generic
    pointer instead and cost a 64-bit register to carry it.

    Plain Python rather than ``@T.inline``: the mapa call has to be handed to
    the enclosing frame explicitly or it is discarded, and the scratch it
    writes into has to be declared in this scope to bind into the PrimFunc.
    """
    mapped = T.alloc_local([1], "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(
            mapped[0], T.cuda.cvta_generic_to_shared(ptr), T.uint32(rank)
        )
    )
    return mapped[0]


def _map_buffer_into_cta(ptr, rank, depth):
    """A buffer view of ``ptr`` as another CTA sees it.

    A buffer needs a pointer to hang off, so this takes the ``mapa.u64`` route
    and reinterprets the result. Callers that only need an address should use
    ``_map_addr_into_cta``, which is a register cheaper.
    """
    from tvm.ir import PointerType, PrimType  # pylint: disable=import-outside-toplevel
    from tvm.tirx import Var as TIRVar  # pylint: disable=import-outside-toplevel

    ptr_ty = PointerType(PrimType("uint64"), "shared")
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    remote_ptr = TIRVar("remote_mbar_ptr", ptr_ty)
    T.Bind(T.reinterpret(ptr_ty, mapped[0]), var=remote_ptr)
    return T.decl_buffer([depth], "uint64", data=remote_ptr, scope="shared")


def _mbarrier_arrive_remote(bar, pred=None, count=None):
    """``mbarrier.arrive.shared::cluster.b64 _, [bar]{, count}``.

    ``count`` spells the arrival count explicitly; omitting it emits the
    implicit count-of-1 line, which is a distinct ISA syntax line rather than
    a default operand.
    """
    chain = T.ptx.mbarrier.arrive.shared__cluster.b64
    args = (bar,) if count is None else (bar, T.uint32(count))
    T.evaluate(chain(*args, pred=pred) if pred is not None else chain(*args))


def _mbarrier_arrive_expect_tx_remote(bar, tx_count, pred=None):
    """``mbarrier.arrive.expect_tx.shared::cluster.b64 _, [bar], txCount``."""
    chain = T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64
    args = (bar, T.uint32(tx_count))
    T.evaluate(chain(*args, pred=pred) if pred is not None else chain(*args))


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
        self._remote_cta_id = None
        self.depth = depth
        self.phase_offset = phase_offset
        self.leader = leader if leader is not None else (T.cuda.thread_rank() == 0)

    def init(self, count):
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be initialized")
        self._init(count)

    @T.inline
    def _init(self, count):
        if self.leader:
            for i in T.unroll(self.depth):
                T.ptx.mbarrier.init.shared.b64(self.buf.ptr_to([i]), T.uint32(count))

    def wait(self, stage, phase):
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be waited on")
        self._wait(stage, phase)

    @T.inline
    def _wait(self, stage, phase):
        # Blocks: ``mbarrier.try_wait`` loops internally until the phase flips,
        # so this returns only once the barrier has completed.
        T.cuda.mbarrier_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset)

    def arrive(self, stage, remote=None, pred=None, count=None):
        if self._remote_cta_id is not None:
            if remote is not None:
                raise ValueError("MBarrier.remote_view().arrive() cannot also specify remote")
            # ``self.buf`` is already the mapped view, so every arrive on it
            # reuses the one mapa the view did.
            _mbarrier_arrive_remote(self.buf.ptr_to([stage]), pred, count)
        elif remote is None:
            self._arrive(self.buf.ptr_to([stage]))
        else:
            # Split of the legacy fused wrapper: map the address into the
            # target CTA, then arrive on it. Plain Python so the mapa scratch
            # binds here; `pred` rides on the arrive alone, since computing a
            # remote address has no effect worth predicating.
            _mbarrier_arrive_remote(
                _map_addr_into_cta(self.buf.ptr_to([stage]), remote), pred, count
            )

    @T.inline
    def _arrive(self, bar):
        # Local-CTA arrive. To arrive on a remote CTA's mbarrier in a cluster
        # kernel, callers must pass ``remote=`` explicitly (e.g.
        # ``bar.arrive(stage, remote=0)``) or use
        # ``MBarrier.remote_view(rank).arrive(stage)``. Defaulting the
        # cross-CTA path was both surprising (``bar.arrive(stage)`` silently
        # ``mapa``ed across the cluster) and a per-call cost of ~3 PTX ops on
        # every single-CTA kernel.
        T.ptx.mbarrier.arrive.shared.b64(bar, T.uint32(1))

    def ptr_to(self, idx):
        return self.buf.ptr_to(idx)

    def remote_view(self, rank):
        """Create a view of this barrier mapped to another CTA's shared memory.

        The returned view retains the local barrier and target CTA so
        ``arrive`` emits the cluster form. Its mapped buffer remains available
        through ``ptr_to`` for operations that consume a remote shared-memory
        pointer. ``init`` and ``wait`` are local-only and reject remote views.
        """
        if self._remote_cta_id is not None:
            raise ValueError("MBarrier.remote_view() cannot be applied to a remote view")

        buf = _map_buffer_into_cta(self.buf.ptr_to([0]), rank, self.depth)
        remote = object.__new__(type(self))
        remote.buf = buf
        remote._remote_cta_id = rank
        remote.depth = self.depth
        remote.phase_offset = self.phase_offset
        return remote


class TMABar(MBarrier):
    """Barrier signaled by TMA (mbarrier.arrive.expect_tx).

    When ``tx_count`` is None, falls back to a remote mbarrier.arrive
    (matching MBarrier.arrive defaults).
    """

    def arrive(self, stage, tx_count=None, remote=None, pred=None):
        # NOTE: this arrive() kwarg set intentionally differs from
        # MBarrier.arrive (hardware necessity, LSP-incompatible by design).
        # ``tx_count``: TMA byte count for ``mbarrier.arrive.expect_tx``.
        # ``remote`` / ``pred``: forwarded to the underlying
        # ``mbarrier.arrive`` (cluster path) when set; otherwise the
        # arrive is local-CTA only. See ``MBarrier.arrive`` for the
        # full default-local rationale.
        if self._remote_cta_id is not None:
            if remote is not None:
                raise ValueError("TMABar.remote_view().arrive() cannot also specify remote")
            remote_bar = self.buf.ptr_to([stage])  # already mapped by the view
        elif remote is not None:
            remote_bar = _map_addr_into_cta(self.buf.ptr_to([stage]), remote)
        else:
            self._arrive_tma_local(self.buf.ptr_to([stage]), tx_count)
            return
        if tx_count is None:
            _mbarrier_arrive_remote(remote_bar, pred)
        else:
            _mbarrier_arrive_expect_tx_remote(remote_bar, tx_count, pred)

    @T.inline
    def _arrive_tma_local(self, bar, tx_count=None):
        if tx_count is None:
            T.ptx.mbarrier.arrive.shared.b64(bar, T.uint32(1))
        else:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(bar, T.uint32(tx_count))


class TCGen05Bar(MBarrier):
    """Barrier signaled by ``tcgen05`` commit.

    The caller is responsible for ensuring only one thread issues the
    commit, e.g. by wrapping the call in ``if T.cuda.elect_sync():`` or by
    passing ``pred=T.cuda.elect_sync()``. The ``pred`` form emits the
    predicated instruction (``@p tcgen05.commit``) instead of a branch and
    lets one elected leader predicate be shared across several commits.
    """

    @T.inline
    def arrive(self, stage, cta_group=1, cta_mask=None, pred=None):
        # NOTE: this arrive() kwarg set intentionally differs from
        # MBarrier.arrive (hardware necessity, LSP-incompatible by design).
        # The unicast/multicast split is decided at trace time, exactly as the
        # legacy wrapper did: a compile-time mask arriving <= 1 CTA means the
        # mbar address already names the target, so no mask operand. A runtime
        # mask is fine -- ctaMask is a register operand -- but the *choice* of
        # form cannot depend on it, so a runtime mask always multicasts.
        # ``pred`` rides the ptx keyword: the instruction is emitted
        # predicated (@p) rather than branched around.
        if _tcgen05_commit_is_unicast(cta_mask):
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}"
                    ".mbarrier::arrive::one.shared::cluster.b64"
                ](self.buf.ptr_to([stage]), pred=pred)
            )
        else:
            T.evaluate(
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}"
                    ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                ](self.buf.ptr_to([stage]), T.Cast("uint16", cta_mask), pred=pred)
            )


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
