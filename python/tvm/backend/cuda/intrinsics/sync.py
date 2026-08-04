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
* ``mbarrier.init`` / ``mbarrier.arrive[.expect_tx]`` (local + remote) / ``mbarrier.try_wait``
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
    FENCE_PROXY_ASYNC_SPACE,
    FENCE_SCOPE,
    FENCE_SEM,
    MBARRIER_ARRIVE_SCOPE,
    MBARRIER_ARRIVE_SEM,
    MBARRIER_ARRIVE_SPACE,
    MBARRIER_COMPLETE_TX_SCOPE,
    MBARRIER_COMPLETE_TX_SEM,
    MBARRIER_COMPLETE_TX_SPACE,
)

from ._schema import device_intrinsic
from .utils import parse_str


def _safe_attr(value):
    return parse_str(value).replace("::", "_").replace(".", "_")


def _as_bool(value) -> bool:
    return bool(int(value)) if hasattr(value, "value") else bool(value)


# =============================================================================
# bar.arrive / bar.sync — aligned named-barrier aliases. 1 form each.
#   bar.sync   a, b ;
#   bar.arrive a, b ;
# barrier.sync — unaligned named barrier. 1 form.
#   barrier.sync a, b ;
# =============================================================================
device_intrinsic(
    "ptx_bar_arrive",
    c_signature="(int name_bar_id, int thread_count)",
    body=(
        '    asm volatile("bar.arrive %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory");'
    ),
)
device_intrinsic(
    "ptx_bar_sync",
    c_signature="(int name_bar_id, int thread_count)",
    body=(
        '    asm volatile("bar.sync %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory");'
    ),
)
device_intrinsic(
    "ptx_barrier_sync",
    c_signature="(int name_bar_id, int thread_count)",
    body=(
        '    asm volatile("barrier.sync %0, %1;" : : '
        '"r"(name_bar_id), "r"(thread_count) : "memory");'
    ),
)


# =============================================================================
# fence{.sem}.scope — 1 form (sem/scope are modifier values).
# =============================================================================
def _ptx_fence(sem, scope):
    sem, scope = parse_str(sem), parse_str(scope)
    assert sem in FENCE_SEM, f"invalid fence sem {sem!r}, expected one of {FENCE_SEM}"
    assert scope in FENCE_SCOPE, f"invalid fence scope {scope!r}, expected one of {FENCE_SCOPE}"
    return (
        f"tvm_builtin_ptx_fence_{sem}_{scope}",
        f'    asm volatile("fence.{sem}.{scope};" ::: "memory");',
    )


device_intrinsic(
    "ptx_fence",
    n_attrs=2,
    helper_name=lambda sem, scope: _ptx_fence(sem, scope)[0],
    body=lambda sem, scope: _ptx_fence(sem, scope)[1],
)


# =============================================================================
# fence.proxy.async{.<space>} — 1 form, optional .space modifier.
# =============================================================================
def _ptx_fence_proxy_async(space):
    space = parse_str(space)
    assert space in FENCE_PROXY_ASYNC_SPACE, (
        f"invalid fence.proxy.async space {space!r}, expected one of {FENCE_PROXY_ASYNC_SPACE}"
    )
    suffix = f".{space}" if space else ""
    name_safe = "_" + space.replace("::", "_").replace(".", "_") if space else ""
    return (
        f"tvm_builtin_ptx_fence_proxy_async{name_safe}",
        f'    asm volatile("fence.proxy.async{suffix};" ::: "memory");',
    )


device_intrinsic(
    "ptx_fence_proxy_async",
    n_attrs=1,
    helper_name=lambda space: _ptx_fence_proxy_async(space)[0],
    body=lambda space: _ptx_fence_proxy_async(space)[1],
)


# =============================================================================
# fence.mbarrier_init.release.cluster — 1 form, no operands.
# =============================================================================
device_intrinsic(
    "ptx_fence_mbarrier_init",
    body='    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
)


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


device_intrinsic(
    "ptx_barrier_cluster_arrive",
    n_attrs=2,
    helper_name=lambda sem, aligned: _ptx_barrier_cluster_arrive(sem, aligned)[0],
    body=lambda sem, aligned: _ptx_barrier_cluster_arrive(sem, aligned)[1],
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


device_intrinsic(
    "ptx_barrier_cluster_wait",
    n_attrs=2,
    helper_name=lambda acquire, aligned: _ptx_barrier_cluster_wait(acquire, aligned)[0],
    body=lambda acquire, aligned: _ptx_barrier_cluster_wait(acquire, aligned)[1],
)


# =============================================================================
# clusterlaunchcontrol.try_cancel / query_cancel — Blackwell Cluster Launch
# Control (CLC) work-stealing, written from the PTX ISA spec (section
# "clusterlaunchcontrol", PTX ISA 8.6). try_cancel async-requests cancelling the
# next cluster's launch, writing a 16B response to smem + signalling mbar. query
# decodes the response: on success it extracts the cancelled cluster's first
# ctaid.x (via the get_first_ctaid::x form); a single uint32 is returned, with
# 0xFFFFFFFF as the "no work stolen" sentinel (a device helper returns one scalar).
# =============================================================================
device_intrinsic(
    "ptx_clc_try_cancel",
    c_signature="(void* handle, void* mbar)",
    body=(
        "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(handle);\n"
        "    unsigned int bar = (unsigned int)__cvta_generic_to_shared(mbar);\n"
        "    asm volatile(\n"
        '        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"\n'
        '        ".multicast::cluster::all.b128 [%0], [%1];\\n"\n'
        '        :: "r"(addr), "r"(bar) : "memory");'
    ),
)


def _ptx_clc_query_cancel_parts(use_ld_acquire):
    use_ld_acquire = (
        bool(int(use_ld_acquire)) if hasattr(use_ld_acquire, "value") else bool(use_ld_acquire)
    )
    name = f"tvm_builtin_ptx_clc_query_cancel{'_ld_acquire' if use_ld_acquire else ''}"
    load_instr = "ld.acquire.cta.shared.b128" if use_ld_acquire else "ld.shared.b128"
    body = (
        "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(handle);\n"
        "    unsigned int first_ctaid_x;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred canceled;\\n"\n'
        '        ".reg .b128 response;\\n"\n'
        f'        "{load_instr} response, [%1];\\n"\n'
        '        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\\n"\n'
        '        "mov.u32 %0, 0xffffffff;\\n"\n'
        '        "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128"\n'
        '        " %0, response;\\n"\n'
        '        "}\\n"\n'
        '        : "=r"(first_ctaid_x) : "r"(addr) : "memory");\n'
        '    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n'
        "    return first_ctaid_x;"
    )
    return name, body


device_intrinsic(
    "ptx_clc_query_cancel",
    n_attrs=1,
    helper_name=lambda *args: _ptx_clc_query_cancel_parts(args[-1])[0],
    c_signature="(void* handle)",
    return_type="uint32_t",
    tvm_return_type="uint32",
    body=lambda *args: _ptx_clc_query_cancel_parts(args[-1])[1],
)


# =============================================================================
# mbarrier.init.shared.b64 [addr], count ; — 1 form.
# =============================================================================
device_intrinsic(
    "ptx_mbarrier_init",
    c_signature="(void* barrier, int thread_count)",
    body=(
        "    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);\n"
        '    asm volatile("mbarrier.init.shared.b64 [%0], %1;"'
        ' : : "r"(barrier_addr), "r"(thread_count) : "memory");'
    ),
)


# =============================================================================
# mbarrier.arrive{.sem.scope}{.space}.b64 _, [addr]{, count};
# =============================================================================
def _check_mbarrier_arrive_attrs(sem, scope, space):
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    if (sem == "") != (scope == ""):
        raise ValueError("mbarrier.arrive sem and scope must be specified together")
    if sem not in MBARRIER_ARRIVE_SEM:
        raise ValueError(f"invalid mbarrier.arrive sem {sem!r}")
    if scope not in MBARRIER_ARRIVE_SCOPE:
        raise ValueError(f"invalid mbarrier.arrive scope {scope!r}")
    if space not in MBARRIER_ARRIVE_SPACE:
        raise ValueError(f"invalid mbarrier.arrive space {space!r}")
    return sem, scope, space


def _mbarrier_address_parts(barrier):
    """Return the CUDA parameter/address spelling for an mbarrier operand."""
    raw_address = str(barrier.ty) == "uint32"
    if raw_address:
        return "unsigned int", "", "barrier", "_raw_u32"
    return (
        "void*",
        "    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);\n",
        "barrier_addr",
        "",
    )


def _ptx_mbarrier_arrive_parts(*args):
    sem, scope, space, has_count, has_remote, has_pred = args[-6:]
    sem, scope, space = _check_mbarrier_arrive_attrs(sem, scope, space)
    has_count = _as_bool(has_count)
    has_remote = _as_bool(has_remote)
    has_pred = _as_bool(has_pred)
    if has_remote and space != "shared::cluster":
        raise ValueError("remote mbarrier.arrive requires space='shared::cluster'")

    barrier_type, address_decl, barrier_addr, raw_suffix = _mbarrier_address_parts(args[0])
    name = "tvm_builtin_ptx_mbarrier_arrive"
    if sem:
        name += f"_{_safe_attr(sem)}_{_safe_attr(scope)}"
    name += f"_{_safe_attr(space)}"
    if has_count:
        name += "_count"
    if has_remote:
        name += "_remote"
    if has_pred:
        name += "_pred"
    name += raw_suffix

    params = [f"{barrier_type} barrier"]
    arg_idx = 1
    if has_count:
        params.append("int count")
        count_idx = arg_idx
        arg_idx += 1
    if has_remote:
        params.append("int remote")
        remote_idx = arg_idx
        arg_idx += 1
    if has_pred:
        params.append("int pred")
        pred_idx = arg_idx

    instr_suffix = f".{sem}.{scope}" if sem else ""
    instr = f"mbarrier.arrive{instr_suffix}.{space}.b64"
    body = address_decl

    if has_remote or has_pred:
        body += "    asm volatile(\n"
        body += '        "{\\n"\n'
        if has_pred:
            body += '        ".reg .pred p;\\n"\n'
        if has_remote:
            body += '        ".reg .b32 remAddr32;\\n"\n'
        if has_pred:
            body += f'        "setp.ne.s32 p, %{pred_idx}, 0;\\n"\n'
        if has_remote:
            prefix = "@p " if has_pred else ""
            body += (
                f'        "{prefix}mapa.shared::cluster.u32  remAddr32, %0, %{remote_idx};\\n"\n'
            )
            addr = "remAddr32"
        else:
            addr = "%0"
        pred_prefix = "@p " if has_pred else ""
        count_suffix = f", %{count_idx}" if has_count else ""
        body += f'        "{pred_prefix}{instr}  _, [{addr}]{count_suffix};\\n"\n'
        body += '        "}\\n"\n'
        constraints = [f'"r"({barrier_addr})']
        if has_count:
            constraints.append('"r"(count)')
        if has_remote:
            constraints.append('"r"(remote)')
        if has_pred:
            constraints.append('"r"(pred)')
        body += f'        :: {", ".join(constraints)} : "memory");'
    else:
        count_suffix = ", %1" if has_count else ""
        constraints = f'"r"({barrier_addr})'
        if has_count:
            constraints += ', "r"(count)'
        body += f'    asm volatile("{instr} _, [%0]{count_suffix};"\n'
        body += f'                 :: {constraints} : "memory");'

    return name, f"({', '.join(params)})", body


device_intrinsic(
    "ptx_mbarrier_arrive",
    n_attrs=6,
    helper_name=lambda *a: _ptx_mbarrier_arrive_parts(*a)[0],
    c_signature=lambda *a: _ptx_mbarrier_arrive_parts(*a)[1],
    body=lambda *a: _ptx_mbarrier_arrive_parts(*a)[2],
)


# mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;
#   sem={.relaxed} scope={.cta,.cluster} space={.shared{::cta},.shared::cluster}
def _ptx_mbarrier_complete_tx_parts(*args):
    sem, scope, space, has_remote, has_pred = args[-5:]
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    has_remote = _as_bool(has_remote)
    has_pred = _as_bool(has_pred)
    if sem not in MBARRIER_COMPLETE_TX_SEM:
        raise ValueError(f"invalid mbarrier.complete_tx sem {sem!r}")
    if scope not in MBARRIER_COMPLETE_TX_SCOPE:
        raise ValueError(f"invalid mbarrier.complete_tx scope {scope!r}")
    if space not in MBARRIER_COMPLETE_TX_SPACE:
        raise ValueError(f"invalid mbarrier.complete_tx space {space!r}")
    if has_remote and space != "shared::cluster":
        raise ValueError("remote mbarrier.complete_tx requires space='shared::cluster'")

    name = (
        "tvm_builtin_ptx_mbarrier_complete_tx"
        f"_{_safe_attr(sem)}_{_safe_attr(scope)}_{_safe_attr(space)}"
    )
    if has_remote:
        name += "_remote"
    if has_pred:
        name += "_pred"

    params = ["void* barrier", "int tx_count"]
    if has_remote:
        params.append("int remote")
    if has_pred:
        params.append("int pred")

    instr = f"mbarrier.complete_tx.{sem}.{scope}.{space}.b64"
    body = "    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);\n"
    addr = "barrier_addr"
    if has_remote:
        body += (
            "    unsigned int remote_addr;\n"
            '    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"\n'
            '                 : "=r"(remote_addr) : "r"(barrier_addr), "r"(remote));\n'
        )
        addr = "remote_addr"
    if has_pred:
        body += (
            "    asm volatile(\n"
            '        "{\\n"\n'
            '        ".reg .pred p;\\n"\n'
            '        "setp.ne.s32 p, %2, 0;\\n"\n'
            f'        "@p {instr} [%0], %1;\\n"\n'
            '        "}\\n"\n'
            f'        :: "r"({addr}), "r"(tx_count), "r"(pred) : "memory");'
        )
    else:
        body += (
            f'    asm volatile("{instr} [%0], %1;"\n'
            f'                 :: "r"({addr}), "r"(tx_count) : "memory");'
        )

    return name, f"({', '.join(params)})", body


device_intrinsic(
    "ptx_mbarrier_complete_tx",
    n_attrs=5,
    helper_name=lambda *a: _ptx_mbarrier_complete_tx_parts(*a)[0],
    c_signature=lambda *a: _ptx_mbarrier_complete_tx_parts(*a)[1],
    body=lambda *a: _ptx_mbarrier_complete_tx_parts(*a)[2],
)


# mbarrier.arrive.expect_tx{.sem.scope}{.space}.b64 _, [addr], txCount;
def _ptx_mbarrier_arrive_expect_tx_parts(*args):
    sem, scope, space, has_remote, has_pred = args[-5:]
    sem, scope, space = _check_mbarrier_arrive_attrs(sem, scope, space)
    has_remote = _as_bool(has_remote)
    has_pred = _as_bool(has_pred)
    if has_remote and space != "shared::cluster":
        raise ValueError("remote mbarrier.arrive.expect_tx requires space='shared::cluster'")

    barrier_type, address_decl, barrier_addr, raw_suffix = _mbarrier_address_parts(args[0])
    name = "tvm_builtin_ptx_mbarrier_arrive_expect_tx"
    if sem:
        name += f"_{_safe_attr(sem)}_{_safe_attr(scope)}"
    name += f"_{_safe_attr(space)}"
    if has_remote:
        name += "_remote"
    if has_pred:
        name += "_pred"
    name += raw_suffix

    params = [f"{barrier_type} barrier", "int byte_count"]
    arg_idx = 2
    if has_remote:
        params.append("int remote")
        remote_idx = arg_idx
        arg_idx += 1
    if has_pred:
        params.append("int pred")
        pred_idx = arg_idx

    instr_suffix = f".{sem}.{scope}" if sem else ""
    instr = f"mbarrier.arrive.expect_tx{instr_suffix}.{space}.b64"
    body = address_decl

    if has_remote or has_pred:
        body += "    asm volatile(\n"
        body += '        "{\\n"\n'
        if has_pred:
            body += '        ".reg .pred p;\\n"\n'
        if has_remote:
            body += '        ".reg .b32 remAddr32;\\n"\n'
        if has_pred:
            body += f'        "setp.ne.s32 p, %{pred_idx}, 0;\\n"\n'
        if has_remote:
            prefix = "@p " if has_pred else ""
            body += (
                f'        "{prefix}mapa.shared::cluster.u32  remAddr32, %0, %{remote_idx};\\n"\n'
            )
            addr = "remAddr32"
        else:
            addr = "%0"
        pred_prefix = "@p " if has_pred else ""
        body += f'        "{pred_prefix}{instr}  _, [{addr}], %1;\\n"\n'
        body += '        "}\\n"\n'
        constraints = [f'"r"({barrier_addr})', '"r"(byte_count)']
        if has_remote:
            constraints.append('"r"(remote)')
        if has_pred:
            constraints.append('"r"(pred)')
        body += f'        :: {", ".join(constraints)} : "memory");'
    else:
        body += f'    asm volatile("{instr} _, [%0], %1;"\n'
        body += f'                 :: "r"({barrier_addr}), "r"(byte_count) : "memory");'

    return name, f"({', '.join(params)})", body


device_intrinsic(
    "ptx_mbarrier_arrive_expect_tx",
    n_attrs=5,
    helper_name=lambda *a: _ptx_mbarrier_arrive_expect_tx_parts(*a)[0],
    c_signature=lambda *a: _ptx_mbarrier_arrive_expect_tx_parts(*a)[1],
    body=lambda *a: _ptx_mbarrier_arrive_expect_tx_parts(*a)[2],
)


def _ptx_mbarrier_arrive_no_complete_parts(*args):
    space, has_pred = args[-2:]
    space = parse_str(space)
    has_pred = _as_bool(has_pred)
    if space not in ("shared", "shared::cta"):
        raise ValueError("mbarrier.arrive.noComplete space must be 'shared' or 'shared::cta'")

    name = f"tvm_builtin_ptx_mbarrier_arrive_no_complete_{_safe_attr(space)}"
    if has_pred:
        name += "_pred"

    params = ["void* barrier", "int count"]
    if has_pred:
        params.append("int pred")
    instr = f"mbarrier.arrive.noComplete.release.cta.{space}.b64"
    body = "    unsigned int barrier_addr = __cvta_generic_to_shared(barrier);\n"
    if has_pred:
        body += (
            "    asm volatile(\n"
            '        "{\\n"\n'
            '        ".reg .pred p;\\n"\n'
            '        "setp.ne.s32 p, %2, 0;\\n"\n'
            f'        "@p {instr} _, [%0], %1;\\n"\n'
            '        "}\\n"\n'
            '        :: "r"(barrier_addr), "r"(count), "r"(pred) : "memory");'
        )
    else:
        body += f'    asm volatile("{instr} _, [%0], %1;"\n'
        body += '                 :: "r"(barrier_addr), "r"(count) : "memory");'
    return name, f"({', '.join(params)})", body


device_intrinsic(
    "ptx_mbarrier_arrive_no_complete",
    n_attrs=2,
    helper_name=lambda *a: _ptx_mbarrier_arrive_no_complete_parts(*a)[0],
    c_signature=lambda *a: _ptx_mbarrier_arrive_no_complete_parts(*a)[1],
    body=lambda *a: _ptx_mbarrier_arrive_no_complete_parts(*a)[2],
)


# =============================================================================
# mbarrier.try_wait.parity.shared::cta.b64 — 1 form. Body wraps the asm in a
# label loop (TIRx convention; the magic ``ticks = 0x989680`` is the timeout
# hint in ns).
# =============================================================================
def _ptx_mbarrier_try_wait_parts(*args):
    barrier_type, address_decl, barrier_addr, raw_suffix = _mbarrier_address_parts(args[0])
    body = (
        address_decl + "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT:\\n"\n'
        '        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\\n"\n'
        '        "@P1                       bra.uni DONE;\\n"\n'
        '        "bra.uni                   LAB_WAIT;\\n"\n'
        '        "DONE:\\n"\n'
        '        "}\\n"\n'
        f'        :: "r"({barrier_addr}), "r"(phase) : "memory");'
    )
    return (
        f"tvm_builtin_ptx_mbarrier_try_wait{raw_suffix}",
        f"({barrier_type} barrier, int phase)",
        body,
    )


device_intrinsic(
    "ptx_mbarrier_try_wait",
    helper_name=lambda *a: _ptx_mbarrier_try_wait_parts(*a)[0],
    c_signature=lambda *a: _ptx_mbarrier_try_wait_parts(*a)[1],
    body=lambda *a: _ptx_mbarrier_try_wait_parts(*a)[2],
)


# mbarrier.try_wait.parity.acquire.cluster — cluster-scope acquire wait used for
# cross-CTA barrier handshakes (e.g. the tmem-finished handoff).
device_intrinsic(
    "ptx_mbarrier_try_wait_acquire_cluster",
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
# mbarrier.try_wait.parity — ONE-SHOT non-blocking variant. Returns true
# if the requested parity has already been reached, false otherwise.
# The TIRx-standard ``ptx_mbarrier_try_wait`` above wraps this in a
# label loop that retries until success; this one-shot form is the
# building block for bounded-retry debug waits (Nymph's
# ``debug_bounded_wait`` lowering mode wraps it in a Python-counted
# loop so the kernel cannot hang forever at a mis-protocoled wait).
# =============================================================================
device_intrinsic(
    "ptx_mbarrier_try_wait_once",
    c_signature="(void* barrier, int phase, int ticks)",
    return_type="uint32_t",
    body=(
        "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        "    unsigned int ticks_u = (unsigned int)ticks;\n"
        "    unsigned int result;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2, %3;\\n"\n'
        '        "selp.u32                  %0, 1, 0, P1;\\n"\n'
        '        "}\\n"\n'
        '        : "=r"(result) : "r"(barrier_addr_int), "r"(phase), "r"(ticks_u) : "memory");\n'
        "    return result;"
    ),
)


# =============================================================================
# elect.sync — TIRx uses the CUDA builtin ``tvm_builtin_elect_one_sync()``
# helper (declared in the CUDA header tags), not direct PTX.
# =============================================================================
device_intrinsic(
    "ptx_elect_sync",
    helper_name="tvm_builtin_elect_one_sync_op",
    return_type="uint32_t",
    body="    return tvm_builtin_elect_one_sync();",
    extra_deps=("elect_one_sync",),
)


# =============================================================================
# __any_sync — warp-vote (pure CUDA helper).
# =============================================================================
device_intrinsic(
    "ptx_any_sync",
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


# PTX mbarrier parity wait form:
#   mbarrier.test_wait.parity{.sem.scope}{.shared{::cta}}.b64 waitComplete, [addr], phaseParity;
def _mbarrier_test_wait_parity_parts(_barrier, _phase, sem, scope, space):
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    if sem and sem not in ("acquire", "relaxed"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity sem {sem!r}")
    if scope and scope not in ("cta", "cluster"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity scope {scope!r}")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity space {space!r}")
    sem_scope = f".{sem}.{scope}" if sem else ""
    name = (
        "tvm_builtin_ptx_mbarrier_test_wait_parity"
        f"{('_' + sem + '_' + scope) if sem else ''}_{space.replace('::', '_')}_b64"
    )
    body = (
        "    unsigned int ready = 0;\n"
        "    asm volatile(\n"
        '        "{\\n\\t"\n'
        '        ".reg .pred P1; \\n\\t"\n'
        f'        "mbarrier.test_wait.parity{sem_scope}.{space}.b64 P1, [%1], %2; \\n\\t"\n'
        '        "selp.b32 %0, 1, 0, P1; \\n\\t"\n'
        '        "}" : "=r"(ready) : "r"((unsigned int)__cvta_generic_to_shared(barrier)), '
        '"r"(phase) : "memory");\n'
        "    return ready;"
    )
    return name, body


device_intrinsic(
    "ptx_mbarrier_test_wait_parity",
    n_attrs=3,
    helper_name=lambda *a: _mbarrier_test_wait_parity_parts(*a)[0],
    c_signature="(void* barrier, int phase)",
    return_type="unsigned int",
    tvm_return_type="uint32",
    body=lambda *a: _mbarrier_test_wait_parity_parts(*a)[1],
)

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
device_intrinsic(
    "ptx_griddepcontrol_wait",
    body='    asm volatile("griddepcontrol.wait;" ::: "memory");',
)

device_intrinsic(
    "ptx_griddepcontrol_launch_dependents",
    body='    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");',
)
