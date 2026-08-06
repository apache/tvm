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
# pylint: disable=invalid-name
"""Synchronization primitives.

PTX side:
* ``bar.arrive`` / ``bar.sync`` — aligned named-barrier aliases
* ``barrier.sync`` — unaligned named barrier for divergent control flow
* ``fence{.sem}.scope`` / ``fence.proxy.async`` / ``fence.mbarrier_init``
* ``barrier.cluster.arrive`` / ``barrier.cluster.wait``
* ``mbarrier.try_wait``
* ``elect.sync``  — warp leader election
* warp-vote ``__any_sync``

CUDA-side helpers:
* ``__threadfence`` / ``__syncwarp`` / ``__syncthreads`` / ``__syncthreads_and|or``
* cooperative-groups grid sync
* cluster sync (open-coded ``barrier.cluster.arrive/wait`` pair)
* warpgroup sync (``bar.sync``)
"""

from tvm.tirx.operator.intrinsics._common import (
    CLUSTER_BARRIER_SEM,
)

from ._schema import device_intrinsic
from .utils import parse_str

# =============================================================================
# bar.arrive / bar.sync — aligned named-barrier aliases. 1 form each.
#   bar.sync   a, b ;
#   bar.arrive a, b ;
# barrier.sync — unaligned named barrier. 1 form.
#   barrier.sync a, b ;
# =============================================================================


# =============================================================================
# fence{.sem}.scope — 1 form (sem/scope are modifier values).


# =============================================================================
# fence.proxy.async{.<space>} — 1 form, optional .space modifier.


# =============================================================================
# fence.mbarrier_init.release.cluster — 1 form, no operands.
# =============================================================================


# =============================================================================
# barrier.cluster.arrive{.sem}{.aligned} — 1 form.
# =============================================================================
def _ptx_barrier_cluster_arrive(sem, aligned):
    sem = parse_str(sem)
    aligned = bool(int(aligned)) if hasattr(aligned, "value") else bool(aligned)
    assert sem in CLUSTER_BARRIER_SEM, (
        f"invalid cluster.arrive sem {sem!r}, expected one of {CLUSTER_BARRIER_SEM}"
    )
    sem_suffix = f".{sem}" if sem else ""
    aligned_suffix = ".aligned" if aligned else ""
    name_sem = "_" + sem.replace("::", "_").replace(".", "_") if sem else ""
    name_aligned = "_aligned" if aligned else ""
    return (
        f"tvm_builtin_ptx_barrier_cluster_arrive{name_sem}{name_aligned}",
        f'    asm volatile("barrier.cluster.arrive{sem_suffix}{aligned_suffix};" ::: "memory");',
    )


# =============================================================================
# barrier.cluster.wait{.acquire}{.aligned} — 1 form.
# =============================================================================
def _ptx_barrier_cluster_wait(acquire, aligned):
    acquire = bool(int(acquire)) if hasattr(acquire, "value") else bool(acquire)
    aligned = bool(int(aligned)) if hasattr(aligned, "value") else bool(aligned)
    acq_suffix = ".acquire" if acquire else ""
    aligned_suffix = ".aligned" if aligned else ""
    return (
        f"tvm_builtin_ptx_barrier_cluster_wait"
        f"{'_acquire' if acquire else ''}{'_aligned' if aligned else ''}",
        f'    asm volatile("barrier.cluster.wait{acq_suffix}{aligned_suffix};" ::: "memory");',
    )


# =============================================================================
# mbarrier.try_wait.parity.shared::cta.b64 — 1 form. Body wraps the asm in a
# label loop (TIRx convention; the magic ``ticks = 0x989680`` is the timeout
# hint in ns).
# =============================================================================
def _mbarrier_wait_parts(barrier, *_rest):
    """Dispatch on the barrier operand's dtype, as the retired op did.

    A ``uint32`` is already a shared-window address (the caller ran cvta once
    and carries offsets in integer space); converting it again would corrupt
    it, so the raw form binds it directly. Anything else is a generic pointer
    and gets the cvta here.
    """
    raw = str(getattr(barrier, "ty", "")) == "uint32"
    return (
        ("_raw_u32" if raw else ""),
        ("(unsigned int barrier, int phase)" if raw else "(void* barrier, int phase)"),
        (
            "    unsigned int barrier_addr_int = barrier;\n"
            if raw
            else "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        ),
    )


device_intrinsic(
    "cuda_mbarrier_wait",
    helper_name=lambda *a: f"tvm_builtin_cuda_mbarrier_wait{_mbarrier_wait_parts(*a)[0]}",
    c_signature=lambda *a: _mbarrier_wait_parts(*a)[1],
    body=lambda *a: (
        _mbarrier_wait_parts(*a)[2]
        + "    unsigned int ticks = 0x989680;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT:\\n"\n'
        '        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"\n'
        '        "@P1                       bra.uni DONE;\\n"\n'
        '        "bra.uni                   LAB_WAIT;\\n"\n'
        '        "DONE:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");'
    ),
)


# mbarrier.try_wait.parity.acquire.cluster — cluster-scope acquire wait used for
# cross-CTA barrier handshakes (e.g. the tmem-finished handoff).
device_intrinsic(
    "cuda_mbarrier_wait_acquire_cluster",
    c_signature="(void* barrier, int phase)",
    body=(
        "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT_AC:\\n"\n'
        '        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\\n"\n'
        '        "@P1                       bra.uni DONE_AC;\\n"\n'
        '        "bra.uni                   LAB_WAIT_AC;\\n"\n'
        '        "DONE_AC:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase) : "memory");'
    ),
)


# =============================================================================
# elect.sync — TIRx uses the CUDA builtin ``tvm_builtin_elect_one_sync()``
# helper (declared in the CUDA header tags), not direct PTX.
# =============================================================================
device_intrinsic(
    "cuda_elect_sync",
    helper_name="tvm_builtin_elect_one_sync_op",
    return_type="uint32_t",
    body="    return tvm_builtin_elect_one_sync();",
    extra_deps=("elect_one_sync",),
)


# =============================================================================
# __any_sync — warp-vote (pure CUDA helper).
# =============================================================================
device_intrinsic(
    "cuda_any_sync",
    c_signature="(unsigned mask, int pred)",
    body="    return __any_sync(mask, pred);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# CUDA-side sync helpers (zero-arg void unless noted).
# =============================================================================
device_intrinsic("cuda_thread_fence", body="    __threadfence();")
device_intrinsic("cuda_warp_sync", body="    __syncwarp();")
device_intrinsic("cuda_cta_sync", body="    __syncthreads();")
device_intrinsic(
    "cuda_grid_sync",
    body="    namespace cg = cooperative_groups;\n    cg::this_grid().sync();",
    extra_deps=("cooperative_groups",),
)
device_intrinsic(
    "cuda_cluster_sync",
    body=('    asm("barrier.cluster.arrive.aligned;");\n    asm("barrier.cluster.wait.aligned;");'),
)
device_intrinsic(
    "cuda_warpgroup_sync",
    c_signature="(int name_bar_id)",
    body='    asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));',
)
device_intrinsic(
    "cuda_syncthreads_and",
    c_signature="(int predicate)",
    body="    return __syncthreads_and(predicate);",
    return_type="int",
    tvm_return_type="int32",
)
device_intrinsic(
    "cuda_syncthreads_or",
    c_signature="(int predicate)",
    body="    return __syncthreads_or(predicate);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# Additional mbarrier, grid-sync, and warp collective helpers.
# =============================================================================


device_intrinsic(
    "cuda_ballot_sync",
    helper_name="tvm_builtin_ballot_sync",
    c_signature="(unsigned int mask, int pred)",
    return_type="unsigned int",
    body="    return __ballot_sync(mask, pred);",
)
device_intrinsic(
    "cuda_reduce_add_sync_u32",
    helper_name="tvm_builtin_reduce_add_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_add_sync(mask, value);",
)
device_intrinsic(
    "cuda_reduce_min_sync_u32",
    helper_name="tvm_builtin_reduce_min_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_min_sync(mask, value);",
)


# =============================================================================
# griddepcontrol.wait / griddepcontrol.launch_dependents (sm_90+)
# Programmatic Dependent Launch (PDL) synchronization. Both carry memory
# clobber to prevent CSE / cross-barrier reordering.
# =============================================================================
