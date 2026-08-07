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
"""Generated stub for T.ptx — do not edit.

Regenerate:
  python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi
"""

from typing import Any

class _Chain_add:
    """`add` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    bf16: _Chain_add
    bf16x2: _Chain_add
    f16: _Chain_add
    f16x2: _Chain_add
    f32: _Chain_add
    f32x2: _Chain_add
    f64: _Chain_add
    ftz: _Chain_add
    rm: _Chain_add
    rn: _Chain_add
    rp: _Chain_add
    rz: _Chain_add
    sat: _Chain_add
    def __call__(self, *args: Any) -> None: ...

class _Chain_atom:
    """`atom` — sem∈{relaxed,acquire,release,acq_rel} (opt); scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster} (opt);
    op∈{and,or,xor,add,inc,dec,min,max}; type∈{b32,b64,u32,u64,s32,s64,f32,f64} — op x type
    pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).      Normative source: ISA Table 35
    (atom) and Table 36 (red), which give the     pairing cell by cell. The `.type = {...}`
    line in the Syntax block is only     the union across ops, which is why it cannot be
    transcribed directly. Half-precision     types appear in ptxas' message but are excluded
    from this entry (they need     .noftz and a half carrier type).
    """

    acq_rel: _Chain_atom
    acquire: _Chain_atom
    add: _Chain_atom
    and_: _Chain_atom
    b32: _Chain_atom
    b64: _Chain_atom
    cluster: _Chain_atom
    cta: _Chain_atom
    dec: _Chain_atom
    f32: _Chain_atom
    f64: _Chain_atom
    global_: _Chain_atom
    gpu: _Chain_atom
    inc: _Chain_atom
    max: _Chain_atom
    min: _Chain_atom
    or_: _Chain_atom
    relaxed: _Chain_atom
    release: _Chain_atom
    s32: _Chain_atom
    s64: _Chain_atom
    shared: _Chain_atom
    shared__cluster: _Chain_atom
    shared__cta: _Chain_atom
    sys: _Chain_atom
    u32: _Chain_atom
    u64: _Chain_atom
    xor: _Chain_atom
    def __call__(self, d: Any, addr: Any, value: Any, *args: Any) -> None: ...

class _Chain_bar:
    """`bar` — 4 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (a); (a, b); (membermask)
    """

    arrive: _Chain_bar
    cta: _Chain_bar
    sync: _Chain_bar
    warp: _Chain_bar
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_barrier:
    """`barrier` — 5 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (a); (a, b); ()
    """

    acquire: _Chain_barrier
    aligned: _Chain_barrier
    arrive: _Chain_barrier
    cluster: _Chain_barrier
    cta: _Chain_barrier
    relaxed: _Chain_barrier
    release: _Chain_barrier
    sync: _Chain_barrier
    wait: _Chain_barrier
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_clusterlaunchcontrol:
    """`clusterlaunchcontrol` — 4 entries sharing this mnemonic; PTX puts their difference in
    the operand list, so the call selects one. Shapes: (addr, mbar); (p, response); (d,
    response); (d0, d1, d2, d3, response)
    """

    async_: _Chain_clusterlaunchcontrol
    b128: _Chain_clusterlaunchcontrol
    b32: _Chain_clusterlaunchcontrol
    get_first_ctaid: _Chain_clusterlaunchcontrol
    get_first_ctaid__x: _Chain_clusterlaunchcontrol
    get_first_ctaid__y: _Chain_clusterlaunchcontrol
    get_first_ctaid__z: _Chain_clusterlaunchcontrol
    is_canceled: _Chain_clusterlaunchcontrol
    mbarrier__complete_tx__bytes: _Chain_clusterlaunchcontrol
    multicast__cluster__all: _Chain_clusterlaunchcontrol
    pred: _Chain_clusterlaunchcontrol
    query_cancel: _Chain_clusterlaunchcontrol
    shared__cta: _Chain_clusterlaunchcontrol
    try_cancel: _Chain_clusterlaunchcontrol
    v4: _Chain_clusterlaunchcontrol
    def __call__(self, *args: Any) -> None: ...

class _Chain_cp:
    """`cp` — 19 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (addr); (); (group); (*__operands); (dst_mem, src_mem,
    size, mbar)
    """

    L2: _Chain_cp
    L2__128B: _Chain_cp
    L2__256B: _Chain_cp
    L2__64B: _Chain_cp
    L2__cache_hint: _Chain_cp
    add: _Chain_cp
    and_: _Chain_cp
    arrive: _Chain_cp
    async_: _Chain_cp
    b64: _Chain_cp
    bulk: _Chain_cp
    bulk_group: _Chain_cp
    ca: _Chain_cp
    cg: _Chain_cp
    commit_group: _Chain_cp
    cp_mask: _Chain_cp
    cta_group__1: _Chain_cp
    cta_group__2: _Chain_cp
    dec: _Chain_cp
    global_: _Chain_cp
    ignore_oob: _Chain_cp
    inc: _Chain_cp
    max: _Chain_cp
    mbarrier: _Chain_cp
    mbarrier__complete_tx__bytes: _Chain_cp
    min: _Chain_cp
    multicast__cluster: _Chain_cp
    noinc: _Chain_cp
    or_: _Chain_cp
    prefetch: _Chain_cp
    read: _Chain_cp
    reduce: _Chain_cp
    shared: _Chain_cp
    shared__cluster: _Chain_cp
    shared__cta: _Chain_cp
    tensor: _Chain_cp
    tile: _Chain_cp
    tile__gather4: _Chain_cp
    tile__scatter4: _Chain_cp
    wait_all: _Chain_cp
    wait_group: _Chain_cp
    xor: _Chain_cp
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_cvt:
    """`cvt` — 27 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a); (d, a, b); (d, a, b, rbits); (*__operands); (d,
    abef0, abef1, abef2, abef3, rbits)
    """

    bf16: _Chain_cvt
    bf16x2: _Chain_cvt
    e2m1x2: _Chain_cvt
    e2m1x4: _Chain_cvt
    e2m3x2: _Chain_cvt
    e2m3x4: _Chain_cvt
    e3m2x2: _Chain_cvt
    e3m2x4: _Chain_cvt
    e4m3x2: _Chain_cvt
    e4m3x4: _Chain_cvt
    e5m2x2: _Chain_cvt
    e5m2x4: _Chain_cvt
    f16: _Chain_cvt
    f16x2: _Chain_cvt
    f32: _Chain_cvt
    f64: _Chain_cvt
    ftz: _Chain_cvt
    relu: _Chain_cvt
    rm: _Chain_cvt
    rmi: _Chain_cvt
    rn: _Chain_cvt
    rna: _Chain_cvt
    rni: _Chain_cvt
    rp: _Chain_cvt
    rpi: _Chain_cvt
    rs: _Chain_cvt
    rz: _Chain_cvt
    rzi: _Chain_cvt
    s16: _Chain_cvt
    s2f6x2: _Chain_cvt
    s32: _Chain_cvt
    s64: _Chain_cvt
    s8: _Chain_cvt
    sat: _Chain_cvt
    satfinite: _Chain_cvt
    scaled__n2__ue8m0: _Chain_cvt
    tf32: _Chain_cvt
    u16: _Chain_cvt
    u32: _Chain_cvt
    u64: _Chain_cvt
    u8: _Chain_cvt
    ue8m0x2: _Chain_cvt
    def __call__(self, *args: Any) -> None: ...

class _Chain_cvta:
    """`cvta` — dir∈{to}; space∈{shared}; type∈{u64}"""

    shared: _Chain_cvta
    to: _Chain_cvta
    u64: _Chain_cvta
    def __call__(self, d: Any, ptr: Any, *args: Any) -> None: ...

class _Chain_ex2:
    """`ex2` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32}"""

    approx: _Chain_ex2
    f32: _Chain_ex2
    ftz: _Chain_ex2
    def __call__(self, d: Any, value: Any, *args: Any) -> None: ...

class _Chain_fence:
    """`fence` — 5 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (); (addr)
    """

    acq_rel: _Chain_fence
    acquire: _Chain_fence
    alias: _Chain_fence
    async_: _Chain_fence
    cluster: _Chain_fence
    cta: _Chain_fence
    global_: _Chain_fence
    gpu: _Chain_fence
    mbarrier_init: _Chain_fence
    proxy: _Chain_fence
    release: _Chain_fence
    sc: _Chain_fence
    shared__cluster: _Chain_fence
    shared__cta: _Chain_fence
    sys: _Chain_fence
    tensormap__generic: _Chain_fence
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_fma:
    """`fma` — rnd∈{rn,rz,rm,rp}; ftz∈{ftz} (opt); sat∈{sat} (opt); type∈{f32,f64,f32x2};
    srctype∈{f16,bf16} (opt) — Which qualifiers each add/sub/mul/fma syntax line allows (PTX
    ISA 9.7.3.{3,4,5,6}, 9.7.5).      Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 |
    op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64     Mixed-precision lines: op{.rnd}{.sat}.f32.atype
    (.atype = .f16 | .bf16)
    """

    bf16: _Chain_fma
    f16: _Chain_fma
    f32: _Chain_fma
    f32x2: _Chain_fma
    f64: _Chain_fma
    ftz: _Chain_fma
    rm: _Chain_fma
    rn: _Chain_fma
    rp: _Chain_fma
    rz: _Chain_fma
    sat: _Chain_fma
    def __call__(self, d: Any, a: Any, b: Any, c: Any, *args: Any) -> None: ...

class _Chain_fns:
    """`fns` — type∈{b32}"""

    b32: _Chain_fns
    def __call__(self, d: Any, mask: Any, base: Any, offset: Any, *args: Any) -> None: ...

class _Chain_griddepcontrol:
    """`griddepcontrol` — action∈{launch_dependents,wait}"""

    launch_dependents: _Chain_griddepcontrol
    wait: _Chain_griddepcontrol
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_ld:
    """`ld` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, addr); (*__operands)
    """

    L1__evict_first: _Chain_ld
    L1__evict_last: _Chain_ld
    L1__evict_normal: _Chain_ld
    L1__evict_unchanged: _Chain_ld
    L1__no_allocate: _Chain_ld
    L2__128B: _Chain_ld
    L2__256B: _Chain_ld
    L2__64B: _Chain_ld
    L2__evict_first: _Chain_ld
    L2__evict_last: _Chain_ld
    L2__evict_normal: _Chain_ld
    acquire: _Chain_ld
    b128: _Chain_ld
    b16: _Chain_ld
    b32: _Chain_ld
    b64: _Chain_ld
    b8: _Chain_ld
    ca: _Chain_ld
    cg: _Chain_ld
    cluster: _Chain_ld
    cs: _Chain_ld
    cta: _Chain_ld
    cv: _Chain_ld
    f32: _Chain_ld
    f64: _Chain_ld
    global_: _Chain_ld
    gpu: _Chain_ld
    local: _Chain_ld
    lu: _Chain_ld
    mmio: _Chain_ld
    nc: _Chain_ld
    relaxed: _Chain_ld
    s16: _Chain_ld
    s32: _Chain_ld
    s64: _Chain_ld
    s8: _Chain_ld
    shared: _Chain_ld
    shared__cluster: _Chain_ld
    shared__cta: _Chain_ld
    sys: _Chain_ld
    u16: _Chain_ld
    u32: _Chain_ld
    u64: _Chain_ld
    u8: _Chain_ld
    v2: _Chain_ld
    v4: _Chain_ld
    v8: _Chain_ld
    volatile: _Chain_ld
    weak: _Chain_ld
    def __call__(self, *args: Any) -> None: ...

class _Chain_ldmatrix:
    """`ldmatrix` — 3 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (*__operands)
    """

    aligned: _Chain_ldmatrix
    b16: _Chain_ldmatrix
    b4x16_p64: _Chain_ldmatrix
    b6x16_p32: _Chain_ldmatrix
    b8: _Chain_ldmatrix
    b8x16: _Chain_ldmatrix
    m16n16: _Chain_ldmatrix
    m8n16: _Chain_ldmatrix
    m8n8: _Chain_ldmatrix
    shared: _Chain_ldmatrix
    shared__cta: _Chain_ldmatrix
    sync: _Chain_ldmatrix
    trans: _Chain_ldmatrix
    x1: _Chain_ldmatrix
    x2: _Chain_ldmatrix
    x4: _Chain_ldmatrix
    def __call__(self, *args: Any) -> None: ...

class _Chain_mapa:
    """`mapa` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    shared__cluster: _Chain_mapa
    u32: _Chain_mapa
    u64: _Chain_mapa
    def __call__(self, *args: Any) -> None: ...

class _Chain_max:
    """`max` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b); (d, a, b, c)
    """

    NaN: _Chain_max
    abs: _Chain_max
    bf16: _Chain_max
    bf16x2: _Chain_max
    f16: _Chain_max
    f16x2: _Chain_max
    f32: _Chain_max
    f64: _Chain_max
    ftz: _Chain_max
    relu: _Chain_max
    s16: _Chain_max
    s16x2: _Chain_max
    s32: _Chain_max
    s64: _Chain_max
    u16: _Chain_max
    u16x2: _Chain_max
    u32: _Chain_max
    u64: _Chain_max
    xorsign: _Chain_max
    def __call__(self, *args: Any) -> None: ...

class _Chain_mbarrier:
    """`mbarrier` — 15 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (addr, count); (addr); (addr, tx_count); (state,
    addr, count); (count, state); (wait_complete, addr, phase); (wait_complete, addr, phase,
    time_hint)
    """

    acquire: _Chain_mbarrier
    arrive: _Chain_mbarrier
    arrive_drop: _Chain_mbarrier
    b64: _Chain_mbarrier
    cluster: _Chain_mbarrier
    complete_tx: _Chain_mbarrier
    cta: _Chain_mbarrier
    expect_tx: _Chain_mbarrier
    init: _Chain_mbarrier
    inval: _Chain_mbarrier
    noComplete: _Chain_mbarrier
    parity: _Chain_mbarrier
    pending_count: _Chain_mbarrier
    relaxed: _Chain_mbarrier
    release: _Chain_mbarrier
    shared: _Chain_mbarrier
    shared__cluster: _Chain_mbarrier
    shared__cta: _Chain_mbarrier
    test_wait: _Chain_mbarrier
    try_wait: _Chain_mbarrier
    def __call__(self, *args: Any) -> None: ...

class _Chain_min:
    """`min` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b); (d, a, b, c)
    """

    NaN: _Chain_min
    abs: _Chain_min
    bf16: _Chain_min
    bf16x2: _Chain_min
    f16: _Chain_min
    f16x2: _Chain_min
    f32: _Chain_min
    f64: _Chain_min
    ftz: _Chain_min
    relu: _Chain_min
    s16: _Chain_min
    s16x2: _Chain_min
    s32: _Chain_min
    s64: _Chain_min
    u16: _Chain_min
    u16x2: _Chain_min
    u32: _Chain_min
    u64: _Chain_min
    xorsign: _Chain_min
    def __call__(self, *args: Any) -> None: ...

class _Chain_mma:
    """`mma` — 12 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands)
    """

    aligned: _Chain_mma
    and_: _Chain_mma
    b1: _Chain_mma
    bf16: _Chain_mma
    col: _Chain_mma
    e4m3: _Chain_mma
    e5m2: _Chain_mma
    f16: _Chain_mma
    f32: _Chain_mma
    f64: _Chain_mma
    m16n8k128: _Chain_mma
    m16n8k16: _Chain_mma
    m16n8k256: _Chain_mma
    m16n8k32: _Chain_mma
    m16n8k4: _Chain_mma
    m16n8k64: _Chain_mma
    m16n8k8: _Chain_mma
    m8n8k128: _Chain_mma
    m8n8k16: _Chain_mma
    m8n8k32: _Chain_mma
    m8n8k4: _Chain_mma
    popc: _Chain_mma
    rm: _Chain_mma
    rn: _Chain_mma
    row: _Chain_mma
    rp: _Chain_mma
    rz: _Chain_mma
    s32: _Chain_mma
    s4: _Chain_mma
    s8: _Chain_mma
    satfinite: _Chain_mma
    sp: _Chain_mma
    sp__ordered_metadata: _Chain_mma
    sync: _Chain_mma
    tf32: _Chain_mma
    u4: _Chain_mma
    u8: _Chain_mma
    xor: _Chain_mma
    def __call__(self, *args: Any) -> None: ...

class _Chain_mov:
    """`mov` — 10 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a0, a1); (d0, d1, a); (d, a0, a1, a2, a3); (d0, d1,
    d2, d3, a)
    """

    b128: _Chain_mov
    b32: _Chain_mov
    b64: _Chain_mov
    def __call__(self, *args: Any) -> None: ...

class _Chain_mul:
    """`mul` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    bf16: _Chain_mul
    bf16x2: _Chain_mul
    f16: _Chain_mul
    f16x2: _Chain_mul
    f32: _Chain_mul
    f32x2: _Chain_mul
    f64: _Chain_mul
    ftz: _Chain_mul
    rm: _Chain_mul
    rn: _Chain_mul
    rp: _Chain_mul
    rz: _Chain_mul
    sat: _Chain_mul
    def __call__(self, *args: Any) -> None: ...

class _Chain_neg:
    """`neg` — ftz∈{ftz} (opt); type∈{f32,f64} — `neg{.ftz}.f32 d, a;` and `neg.f64 d, a;` (ISA
    9.7.3.10) -- the f64     line spells no `.ftz`.
    """

    f32: _Chain_neg
    f64: _Chain_neg
    ftz: _Chain_neg
    def __call__(self, d: Any, a: Any, *args: Any) -> None: ...

class _Chain_prefetch:
    """`prefetch` — space∈{global,local,const,param} (opt); level∈{L1,L2} (opt);
    evict∈{L2::evict_last,L2::evict_normal} (opt); tensormap∈{tensormap} (opt) — Each
    prefetch syntax line names exactly one target (PTX ISA 9.7.9.16).
    `.level::eviction_priority` stays bound to `.global` on purpose: its syntax     line is
    `prefetch.global.level::eviction_priority`, with `.global` written     in rather than
    the `{.ss}` that the `ld` lines carry. Generic addressing is     not offered there, so
    neither is it here.
    """

    L1: _Chain_prefetch
    L2: _Chain_prefetch
    L2__evict_last: _Chain_prefetch
    L2__evict_normal: _Chain_prefetch
    const: _Chain_prefetch
    global_: _Chain_prefetch
    local: _Chain_prefetch
    param: _Chain_prefetch
    tensormap: _Chain_prefetch
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_rcp:
    """`rcp` — mode∈{approx,rn,rz,rm,rp}; ftz∈{ftz} (opt); type∈{f32,f64} — This entry's
    rcp.approx is f32-only; .f64 is IEEE-rounded, no .ftz (PTX ISA 9.7.3.13).
    """

    approx: _Chain_rcp
    f32: _Chain_rcp
    f64: _Chain_rcp
    ftz: _Chain_rcp
    rm: _Chain_rcp
    rn: _Chain_rcp
    rp: _Chain_rcp
    rz: _Chain_rcp
    def __call__(self, d: Any, value: Any, *args: Any) -> None: ...

class _Chain_red:
    """`red` — sem∈{relaxed,release} (opt); scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster} (opt);
    op∈{and,or,xor,add,inc,dec,min,max}; type∈{b32,b64,u32,u64,s32,s64,f32,f64} — op x type
    pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).      Normative source: ISA Table 35
    (atom) and Table 36 (red), which give the     pairing cell by cell. The `.type = {...}`
    line in the Syntax block is only     the union across ops, which is why it cannot be
    transcribed directly. Half-precision     types appear in ptxas' message but are excluded
    from this entry (they need     .noftz and a half carrier type).
    """

    add: _Chain_red
    and_: _Chain_red
    b32: _Chain_red
    b64: _Chain_red
    cluster: _Chain_red
    cta: _Chain_red
    dec: _Chain_red
    f32: _Chain_red
    f64: _Chain_red
    global_: _Chain_red
    gpu: _Chain_red
    inc: _Chain_red
    max: _Chain_red
    min: _Chain_red
    or_: _Chain_red
    relaxed: _Chain_red
    release: _Chain_red
    s32: _Chain_red
    s64: _Chain_red
    shared: _Chain_red
    shared__cluster: _Chain_red
    shared__cta: _Chain_red
    sys: _Chain_red
    u32: _Chain_red
    u64: _Chain_red
    xor: _Chain_red
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_setmaxnreg:
    """`setmaxnreg` — action∈{inc,dec}; sync∈{sync}; aligned∈{aligned}; type∈{u32}"""

    aligned: _Chain_setmaxnreg
    dec: _Chain_setmaxnreg
    inc: _Chain_setmaxnreg
    sync: _Chain_setmaxnreg
    u32: _Chain_setmaxnreg
    def __call__(self, nreg: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_st:
    """`st` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (addr, value); (*__operands)
    """

    L1__evict_first: _Chain_st
    L1__evict_last: _Chain_st
    L1__evict_normal: _Chain_st
    L1__evict_unchanged: _Chain_st
    L1__no_allocate: _Chain_st
    L2__evict_first: _Chain_st
    L2__evict_last: _Chain_st
    L2__evict_normal: _Chain_st
    b128: _Chain_st
    b16: _Chain_st
    b32: _Chain_st
    b64: _Chain_st
    b8: _Chain_st
    cg: _Chain_st
    cluster: _Chain_st
    cs: _Chain_st
    cta: _Chain_st
    f32: _Chain_st
    f64: _Chain_st
    global_: _Chain_st
    gpu: _Chain_st
    local: _Chain_st
    mmio: _Chain_st
    relaxed: _Chain_st
    release: _Chain_st
    s16: _Chain_st
    s32: _Chain_st
    s64: _Chain_st
    s8: _Chain_st
    shared: _Chain_st
    shared__cluster: _Chain_st
    shared__cta: _Chain_st
    sys: _Chain_st
    u16: _Chain_st
    u32: _Chain_st
    u64: _Chain_st
    u8: _Chain_st
    v2: _Chain_st
    v4: _Chain_st
    v8: _Chain_st
    volatile: _Chain_st
    wb: _Chain_st
    weak: _Chain_st
    wt: _Chain_st
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_st_bulk:
    """`st_bulk` — weak∈{weak} (opt); space∈{shared::cta} (opt)"""

    shared__cta: _Chain_st_bulk
    weak: _Chain_st_bulk
    def __call__(self, addr: Any, size: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_stmatrix:
    """`stmatrix` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (*__operands)
    """

    aligned: _Chain_stmatrix
    b16: _Chain_stmatrix
    b8: _Chain_stmatrix
    m16n8: _Chain_stmatrix
    m8n8: _Chain_stmatrix
    shared: _Chain_stmatrix
    shared__cta: _Chain_stmatrix
    sync: _Chain_stmatrix
    trans: _Chain_stmatrix
    x1: _Chain_stmatrix
    x2: _Chain_stmatrix
    x4: _Chain_stmatrix
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_sub:
    """`sub` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    bf16: _Chain_sub
    bf16x2: _Chain_sub
    f16: _Chain_sub
    f16x2: _Chain_sub
    f32: _Chain_sub
    f32x2: _Chain_sub
    f64: _Chain_sub
    ftz: _Chain_sub
    rm: _Chain_sub
    rn: _Chain_sub
    rp: _Chain_sub
    rz: _Chain_sub
    sat: _Chain_sub
    def __call__(self, *args: Any) -> None: ...

class _Chain_tcgen05:
    """`tcgen05` — 18 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (dst, ncols); (taddr, ncols); (); (mbar); (mbar,
    mask); (*__operands); (taddr, s_desc); (d_tmem, a_desc, b_desc, idesc, enable_input_d,
    zero_col_mask); (d_tmem, a_tmem, b_desc, idesc, enable_input_d, zero_col_mask); (d_tmem,
    a_desc, b_desc, idesc, sfa_tmem, sfb_tmem, enable_input_d); (d_tmem, a_tmem, b_desc,
    idesc, sfa_tmem, sfb_tmem, enable_input_d)
    """

    aligned: _Chain_tcgen05
    alloc: _Chain_tcgen05
    b32: _Chain_tcgen05
    b4x16_p64: _Chain_tcgen05
    b64: _Chain_tcgen05
    b6x16_p32: _Chain_tcgen05
    b8x16: _Chain_tcgen05
    block_scale: _Chain_tcgen05
    commit: _Chain_tcgen05
    cp: _Chain_tcgen05
    cta_group__1: _Chain_tcgen05
    cta_group__2: _Chain_tcgen05
    dealloc: _Chain_tcgen05
    fence__after_thread_sync: _Chain_tcgen05
    fence__before_thread_sync: _Chain_tcgen05
    kind__f16: _Chain_tcgen05
    kind__f8f6f4: _Chain_tcgen05
    kind__i8: _Chain_tcgen05
    kind__mxf4: _Chain_tcgen05
    kind__mxf4nvf4: _Chain_tcgen05
    kind__mxf8f6f4: _Chain_tcgen05
    kind__tf32: _Chain_tcgen05
    ld: _Chain_tcgen05
    mbarrier__arrive__one: _Chain_tcgen05
    mma: _Chain_tcgen05
    multicast__cluster: _Chain_tcgen05
    pack__16b: _Chain_tcgen05
    relinquish_alloc_permit: _Chain_tcgen05
    scale_vec__1X: _Chain_tcgen05
    scale_vec__2X: _Chain_tcgen05
    scale_vec__4X: _Chain_tcgen05
    shared__cluster: _Chain_tcgen05
    shared__cta: _Chain_tcgen05
    st: _Chain_tcgen05
    sync: _Chain_tcgen05
    unpack__16b: _Chain_tcgen05
    wait__ld: _Chain_tcgen05
    wait__st: _Chain_tcgen05
    warpx2__01_23: _Chain_tcgen05
    warpx2__02_13: _Chain_tcgen05
    warpx4: _Chain_tcgen05
    ws: _Chain_tcgen05
    x1: _Chain_tcgen05
    x128: _Chain_tcgen05
    x16: _Chain_tcgen05
    x2: _Chain_tcgen05
    x32: _Chain_tcgen05
    x4: _Chain_tcgen05
    x64: _Chain_tcgen05
    x8: _Chain_tcgen05
    def __call__(self, *args: Any) -> None: ...

class _Chain_wgmma:
    """`wgmma` — 19 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (group); (); (*__operands)
    """

    aligned: _Chain_wgmma
    and_: _Chain_wgmma
    b1: _Chain_wgmma
    bf16: _Chain_wgmma
    commit_group: _Chain_wgmma
    e4m3: _Chain_wgmma
    e5m2: _Chain_wgmma
    f16: _Chain_wgmma
    f32: _Chain_wgmma
    fence: _Chain_wgmma
    m64n104k16: _Chain_wgmma
    m64n104k32: _Chain_wgmma
    m64n104k8: _Chain_wgmma
    m64n112k16: _Chain_wgmma
    m64n112k256: _Chain_wgmma
    m64n112k32: _Chain_wgmma
    m64n112k8: _Chain_wgmma
    m64n120k16: _Chain_wgmma
    m64n120k32: _Chain_wgmma
    m64n120k8: _Chain_wgmma
    m64n128k16: _Chain_wgmma
    m64n128k256: _Chain_wgmma
    m64n128k32: _Chain_wgmma
    m64n128k8: _Chain_wgmma
    m64n136k16: _Chain_wgmma
    m64n136k32: _Chain_wgmma
    m64n136k8: _Chain_wgmma
    m64n144k16: _Chain_wgmma
    m64n144k256: _Chain_wgmma
    m64n144k32: _Chain_wgmma
    m64n144k8: _Chain_wgmma
    m64n152k16: _Chain_wgmma
    m64n152k32: _Chain_wgmma
    m64n152k8: _Chain_wgmma
    m64n160k16: _Chain_wgmma
    m64n160k256: _Chain_wgmma
    m64n160k32: _Chain_wgmma
    m64n160k8: _Chain_wgmma
    m64n168k16: _Chain_wgmma
    m64n168k32: _Chain_wgmma
    m64n168k8: _Chain_wgmma
    m64n16k16: _Chain_wgmma
    m64n16k256: _Chain_wgmma
    m64n16k32: _Chain_wgmma
    m64n16k8: _Chain_wgmma
    m64n176k16: _Chain_wgmma
    m64n176k256: _Chain_wgmma
    m64n176k32: _Chain_wgmma
    m64n176k8: _Chain_wgmma
    m64n184k16: _Chain_wgmma
    m64n184k32: _Chain_wgmma
    m64n184k8: _Chain_wgmma
    m64n192k16: _Chain_wgmma
    m64n192k256: _Chain_wgmma
    m64n192k32: _Chain_wgmma
    m64n192k8: _Chain_wgmma
    m64n200k16: _Chain_wgmma
    m64n200k32: _Chain_wgmma
    m64n200k8: _Chain_wgmma
    m64n208k16: _Chain_wgmma
    m64n208k256: _Chain_wgmma
    m64n208k32: _Chain_wgmma
    m64n208k8: _Chain_wgmma
    m64n216k16: _Chain_wgmma
    m64n216k32: _Chain_wgmma
    m64n216k8: _Chain_wgmma
    m64n224k16: _Chain_wgmma
    m64n224k256: _Chain_wgmma
    m64n224k32: _Chain_wgmma
    m64n224k8: _Chain_wgmma
    m64n232k16: _Chain_wgmma
    m64n232k32: _Chain_wgmma
    m64n232k8: _Chain_wgmma
    m64n240k16: _Chain_wgmma
    m64n240k256: _Chain_wgmma
    m64n240k32: _Chain_wgmma
    m64n240k8: _Chain_wgmma
    m64n248k16: _Chain_wgmma
    m64n248k32: _Chain_wgmma
    m64n248k8: _Chain_wgmma
    m64n24k16: _Chain_wgmma
    m64n24k256: _Chain_wgmma
    m64n24k32: _Chain_wgmma
    m64n24k8: _Chain_wgmma
    m64n256k16: _Chain_wgmma
    m64n256k256: _Chain_wgmma
    m64n256k32: _Chain_wgmma
    m64n256k8: _Chain_wgmma
    m64n32k16: _Chain_wgmma
    m64n32k256: _Chain_wgmma
    m64n32k32: _Chain_wgmma
    m64n32k8: _Chain_wgmma
    m64n40k16: _Chain_wgmma
    m64n40k32: _Chain_wgmma
    m64n40k8: _Chain_wgmma
    m64n48k16: _Chain_wgmma
    m64n48k256: _Chain_wgmma
    m64n48k32: _Chain_wgmma
    m64n48k8: _Chain_wgmma
    m64n56k16: _Chain_wgmma
    m64n56k32: _Chain_wgmma
    m64n56k8: _Chain_wgmma
    m64n64k16: _Chain_wgmma
    m64n64k256: _Chain_wgmma
    m64n64k32: _Chain_wgmma
    m64n64k8: _Chain_wgmma
    m64n72k16: _Chain_wgmma
    m64n72k32: _Chain_wgmma
    m64n72k8: _Chain_wgmma
    m64n80k16: _Chain_wgmma
    m64n80k256: _Chain_wgmma
    m64n80k32: _Chain_wgmma
    m64n80k8: _Chain_wgmma
    m64n88k16: _Chain_wgmma
    m64n88k32: _Chain_wgmma
    m64n88k8: _Chain_wgmma
    m64n8k16: _Chain_wgmma
    m64n8k256: _Chain_wgmma
    m64n8k32: _Chain_wgmma
    m64n8k8: _Chain_wgmma
    m64n96k16: _Chain_wgmma
    m64n96k256: _Chain_wgmma
    m64n96k32: _Chain_wgmma
    m64n96k8: _Chain_wgmma
    mma_async: _Chain_wgmma
    popc: _Chain_wgmma
    s32: _Chain_wgmma
    s8: _Chain_wgmma
    satfinite: _Chain_wgmma
    sync: _Chain_wgmma
    tf32: _Chain_wgmma
    u8: _Chain_wgmma
    wait_group: _Chain_wgmma
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _PTX:
    add: _Chain_add
    atom: _Chain_atom
    bar: _Chain_bar
    barrier: _Chain_barrier
    clusterlaunchcontrol: _Chain_clusterlaunchcontrol
    cp: _Chain_cp
    cvt: _Chain_cvt
    cvta: _Chain_cvta
    ex2: _Chain_ex2
    fence: _Chain_fence
    fma: _Chain_fma
    fns: _Chain_fns
    griddepcontrol: _Chain_griddepcontrol
    ld: _Chain_ld
    ldmatrix: _Chain_ldmatrix
    mapa: _Chain_mapa
    max: _Chain_max
    mbarrier: _Chain_mbarrier
    min: _Chain_min
    mma: _Chain_mma
    mov: _Chain_mov
    mul: _Chain_mul
    neg: _Chain_neg
    prefetch: _Chain_prefetch
    rcp: _Chain_rcp
    red: _Chain_red
    setmaxnreg: _Chain_setmaxnreg
    st: _Chain_st
    st_bulk: _Chain_st_bulk
    stmatrix: _Chain_stmatrix
    sub: _Chain_sub
    tcgen05: _Chain_tcgen05
    wgmma: _Chain_wgmma
    def __getitem__(self, text: str) -> Any: ...

ptx: _PTX

# Every other tvm.script.tirx member stays dynamically typed, as before.
def __getattr__(name: str) -> Any: ...
