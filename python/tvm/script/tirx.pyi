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
  python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi
"""

from typing import Any

class _Chain_abs:
    """`abs` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a)
    """

    bf16: _Chain_abs
    bf16x2: _Chain_abs
    f16: _Chain_abs
    f16x2: _Chain_abs
    f32: _Chain_abs
    f64: _Chain_abs
    ftz: _Chain_abs
    s16: _Chain_abs
    s32: _Chain_abs
    s64: _Chain_abs
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_activemask:
    """`activemask` — type∈{b32}"""

    b32: _Chain_activemask
    def __call__(
        self,
        d: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_add:
    """`add` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
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
    s16: _Chain_add
    s16x2: _Chain_add
    s32: _Chain_add
    s64: _Chain_add
    sat: _Chain_add
    u16: _Chain_add
    u16x2: _Chain_add
    u32: _Chain_add
    u64: _Chain_add
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_and:
    """`and` — type∈{pred,b16,b32,b64}"""

    b16: _Chain_and
    b32: _Chain_and
    b64: _Chain_and
    pred: _Chain_and
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_applypriority:
    """`applypriority` — space∈{global} (opt); level∈{L2::evict_normal}"""

    L2__evict_normal: _Chain_applypriority
    global_: _Chain_applypriority
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_atom:
    """`atom` — 6 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands); (d, addr, compare, value)
    """

    L2__cache_hint: _Chain_atom
    acq_rel: _Chain_atom
    acquire: _Chain_atom
    add: _Chain_atom
    and_: _Chain_atom
    b128: _Chain_atom
    b16: _Chain_atom
    b32: _Chain_atom
    b64: _Chain_atom
    bf16: _Chain_atom
    bf16x2: _Chain_atom
    cas: _Chain_atom
    cluster: _Chain_atom
    cta: _Chain_atom
    dec: _Chain_atom
    exch: _Chain_atom
    f16: _Chain_atom
    f16x2: _Chain_atom
    f32: _Chain_atom
    f64: _Chain_atom
    global_: _Chain_atom
    gpu: _Chain_atom
    inc: _Chain_atom
    max: _Chain_atom
    min: _Chain_atom
    noftz: _Chain_atom
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
    v2: _Chain_atom
    v4: _Chain_atom
    v8: _Chain_atom
    xor: _Chain_atom
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_bar:
    """`bar` — 8 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (a); (a, b); (membermask); (d, a, c); (d, a, b, c)
    """

    and_: _Chain_bar
    arrive: _Chain_bar
    cta: _Chain_bar
    or_: _Chain_bar
    popc: _Chain_bar
    pred: _Chain_bar
    red: _Chain_bar
    sync: _Chain_bar
    u32: _Chain_bar
    warp: _Chain_bar
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_barrier:
    """`barrier` — 9 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (a); (a, b); (); (d, a, c); (d, a, b, c)
    """

    acquire: _Chain_barrier
    aligned: _Chain_barrier
    and_: _Chain_barrier
    arrive: _Chain_barrier
    cluster: _Chain_barrier
    cta: _Chain_barrier
    or_: _Chain_barrier
    popc: _Chain_barrier
    pred: _Chain_barrier
    red: _Chain_barrier
    relaxed: _Chain_barrier
    release: _Chain_barrier
    sync: _Chain_barrier
    u32: _Chain_barrier
    wait: _Chain_barrier
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_bfe:
    """`bfe` — type∈{u32,u64,s32,s64}"""

    s32: _Chain_bfe
    s64: _Chain_bfe
    u32: _Chain_bfe
    u64: _Chain_bfe
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_bfi:
    """`bfi` — type∈{b32,b64}"""

    b32: _Chain_bfi
    b64: _Chain_bfi
    def __call__(
        self,
        f: Any,
        a: Any,
        b: Any,
        c: Any,
        d: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_bfind:
    """`bfind` — shiftamt∈{shiftamt} (opt); type∈{u32,u64,s32,s64}"""

    s32: _Chain_bfind
    s64: _Chain_bfind
    shiftamt: _Chain_bfind
    u32: _Chain_bfind
    u64: _Chain_bfind
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_bmsk:
    """`bmsk` — mode∈{clamp,wrap}; type∈{b32}"""

    b32: _Chain_bmsk
    clamp: _Chain_bmsk
    wrap: _Chain_bmsk
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_brev:
    """`brev` — type∈{b32,b64}"""

    b32: _Chain_brev
    b64: _Chain_brev
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_clz:
    """`clz` — type∈{b32,b64}"""

    b32: _Chain_clz
    b64: _Chain_clz
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_cnot:
    """`cnot` — type∈{b16,b32,b64}"""

    b16: _Chain_cnot
    b32: _Chain_cnot
    b64: _Chain_cnot
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_copysign:
    """`copysign` — type∈{f32,f64}"""

    f32: _Chain_copysign
    f64: _Chain_copysign
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_cos:
    """`cos` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32}"""

    approx: _Chain_cos
    f32: _Chain_cos
    ftz: _Chain_cos
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_cp:
    """`cp` — 22 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands); (group); (); (dst_mem, src_mem, size,
    mbar); (addr)
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

class _Chain_createpolicy:
    """`createpolicy` — 4 entries sharing this mnemonic; PTX puts their difference in the
    operand list, so the call selects one. Shapes: (cache_policy, addr, primary_size,
    total_size); (cache_policy); (cache_policy, fraction); (cache_policy, access_property)
    """

    L2: _Chain_createpolicy
    L2__evict_first: _Chain_createpolicy
    L2__evict_last: _Chain_createpolicy
    L2__evict_normal: _Chain_createpolicy
    L2__evict_unchanged: _Chain_createpolicy
    b64: _Chain_createpolicy
    cvt: _Chain_createpolicy
    fractional: _Chain_createpolicy
    global_: _Chain_createpolicy
    range: _Chain_createpolicy
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_cvt_pack:
    """`cvt_pack` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a, b); (d, a, b, c)
    """

    b32: _Chain_cvt_pack
    s16: _Chain_cvt_pack
    s2: _Chain_cvt_pack
    s32: _Chain_cvt_pack
    s4: _Chain_cvt_pack
    s8: _Chain_cvt_pack
    sat: _Chain_cvt_pack
    u16: _Chain_cvt_pack
    u2: _Chain_cvt_pack
    u4: _Chain_cvt_pack
    u8: _Chain_cvt_pack
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_cvta:
    """`cvta` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, ptr); (d, a)
    """

    const: _Chain_cvta
    global_: _Chain_cvta
    local: _Chain_cvta
    param: _Chain_cvta
    param__entry: _Chain_cvta
    shared: _Chain_cvta
    shared__cluster: _Chain_cvta
    shared__cta: _Chain_cvta
    to: _Chain_cvta
    u64: _Chain_cvta
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_discard:
    """`discard` — space∈{global} (opt); level∈{L2}"""

    L2: _Chain_discard
    global_: _Chain_discard
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_div:
    """`div` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    approx: _Chain_div
    f32: _Chain_div
    f64: _Chain_div
    ftz: _Chain_div
    full: _Chain_div
    rm: _Chain_div
    rn: _Chain_div
    rp: _Chain_div
    rz: _Chain_div
    s16: _Chain_div
    s32: _Chain_div
    s64: _Chain_div
    u16: _Chain_div
    u32: _Chain_div
    u64: _Chain_div
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_dp2a:
    """`dp2a` — mode∈{lo,hi}; atype∈{u32,s32}; btype∈{u32,s32}"""

    hi: _Chain_dp2a
    lo: _Chain_dp2a
    s32: _Chain_dp2a
    u32: _Chain_dp2a
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_dp4a:
    """`dp4a` — atype∈{u32,s32}; btype∈{u32,s32}"""

    s32: _Chain_dp4a
    u32: _Chain_dp4a
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_elect_sync:
    """`elect_sync` — (no modifiers)"""

    def __call__(
        self,
        d: Any,
        p: Any,
        membermask: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_ex2:
    """`ex2` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, value)
    """

    approx: _Chain_ex2
    bf16: _Chain_ex2
    bf16x2: _Chain_ex2
    f16: _Chain_ex2
    f16x2: _Chain_ex2
    f32: _Chain_ex2
    ftz: _Chain_ex2
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    """`fma` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b, c)
    """

    bf16: _Chain_fma
    bf16x2: _Chain_fma
    f16: _Chain_fma
    f16x2: _Chain_fma
    f32: _Chain_fma
    f32x2: _Chain_fma
    f64: _Chain_fma
    ftz: _Chain_fma
    oob: _Chain_fma
    relu: _Chain_fma
    rm: _Chain_fma
    rn: _Chain_fma
    rp: _Chain_fma
    rz: _Chain_fma
    sat: _Chain_fma
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_fns:
    """`fns` — type∈{b32}"""

    b32: _Chain_fns
    def __call__(
        self,
        d: Any,
        mask: Any,
        base: Any,
        offset: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_getctarank:
    """`getctarank` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a)
    """

    shared__cluster: _Chain_getctarank
    u32: _Chain_getctarank
    u64: _Chain_getctarank
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_griddepcontrol:
    """`griddepcontrol` — action∈{launch_dependents,wait}"""

    launch_dependents: _Chain_griddepcontrol
    wait: _Chain_griddepcontrol
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_isspacep:
    """`isspacep` —
    space∈{const,global,local,shared,shared::cta,shared::cluster,param,param::entry}
    """

    const: _Chain_isspacep
    global_: _Chain_isspacep
    local: _Chain_isspacep
    param: _Chain_isspacep
    param__entry: _Chain_isspacep
    shared: _Chain_isspacep
    shared__cluster: _Chain_isspacep
    shared__cta: _Chain_isspacep
    def __call__(
        self,
        p: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_ld:
    """`ld` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands)
    """

    L1__evict_first: _Chain_ld
    L1__evict_last: _Chain_ld
    L1__evict_normal: _Chain_ld
    L1__evict_unchanged: _Chain_ld
    L1__no_allocate: _Chain_ld
    L2__128B: _Chain_ld
    L2__256B: _Chain_ld
    L2__64B: _Chain_ld
    L2__cache_hint: _Chain_ld
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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_ldu:
    """`ldu` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, addr); (*__operands)
    """

    b128: _Chain_ldu
    b16: _Chain_ldu
    b32: _Chain_ldu
    b64: _Chain_ldu
    b8: _Chain_ldu
    f32: _Chain_ldu
    f64: _Chain_ldu
    global_: _Chain_ldu
    s16: _Chain_ldu
    s32: _Chain_ldu
    s64: _Chain_ldu
    s8: _Chain_ldu
    u16: _Chain_ldu
    u32: _Chain_ldu
    u64: _Chain_ldu
    u8: _Chain_ldu
    v2: _Chain_ldu
    v4: _Chain_ldu
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_lg2:
    """`lg2` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32}"""

    approx: _Chain_lg2
    f32: _Chain_lg2
    ftz: _Chain_lg2
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_lop3:
    """`lop3` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b, c, immLut); (d, p, a, b, c, immLut, q)
    """

    and_: _Chain_lop3
    b32: _Chain_lop3
    or_: _Chain_lop3
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mad:
    """`mad` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b, c)
    """

    f32: _Chain_mad
    f64: _Chain_mad
    ftz: _Chain_mad
    hi: _Chain_mad
    lo: _Chain_mad
    rm: _Chain_mad
    rn: _Chain_mad
    rp: _Chain_mad
    rz: _Chain_mad
    s16: _Chain_mad
    s32: _Chain_mad
    s64: _Chain_mad
    sat: _Chain_mad
    u16: _Chain_mad
    u32: _Chain_mad
    u64: _Chain_mad
    wide: _Chain_mad
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mad24:
    """`mad24` — mode∈{hi,lo}; sat∈{sat} (opt); type∈{u32,s32} — `.sat` on the multiply-add
    lines: `.hi` mode, `.s32` type, nothing else. Both lines spell it as a syntax line of
    its own -- `mad.hi.sat.s32 d, a, b, c;` (ISA 9.7.1.4) and `mad24.hi.sat.s32 d, a, b, c;`
    (9.7.1.7) -- with the Notes repeating "Applies only to .s32 type in .hi mode".
    """

    hi: _Chain_mad24
    lo: _Chain_mad24
    s32: _Chain_mad24
    sat: _Chain_mad24
    u32: _Chain_mad24
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_mapa:
    """`mapa` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b)
    """

    shared__cluster: _Chain_mapa
    u32: _Chain_mapa
    u64: _Chain_mapa
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_match:
    """`match` — 3 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a, membermask); (d, p, a, membermask)
    """

    all: _Chain_match
    any: _Chain_match
    b32: _Chain_match
    b64: _Chain_match
    sync: _Chain_match
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mbarrier:
    """`mbarrier` — 25 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (addr, count); (addr); (addr, tx_count); (state,
    addr, count); (wait_complete, addr, phase); (wait_complete, addr, phase, time_hint);
    (state, addr); (wait_complete, addr, state); (wait_complete, addr, state, time_hint);
    (count, state)
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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mov:
    """`mov` — 11 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a0, a1); (d0, d1, a); (d, a0, a1, a2, a3); (d0, d1,
    d2, d3, a); (d, a)
    """

    b128: _Chain_mov
    b16: _Chain_mov
    b32: _Chain_mov
    b64: _Chain_mov
    f32: _Chain_mov
    f64: _Chain_mov
    pred: _Chain_mov
    s16: _Chain_mov
    s32: _Chain_mov
    s64: _Chain_mov
    u16: _Chain_mov
    u32: _Chain_mov
    u64: _Chain_mov
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mul:
    """`mul` — 4 entries sharing this mnemonic; PTX puts their difference in the operand list,
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
    hi: _Chain_mul
    lo: _Chain_mul
    rm: _Chain_mul
    rn: _Chain_mul
    rp: _Chain_mul
    rz: _Chain_mul
    s16: _Chain_mul
    s32: _Chain_mul
    s64: _Chain_mul
    sat: _Chain_mul
    u16: _Chain_mul
    u32: _Chain_mul
    u64: _Chain_mul
    wide: _Chain_mul
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_mul24:
    """`mul24` — mode∈{hi,lo}; type∈{u32,s32}"""

    hi: _Chain_mul24
    lo: _Chain_mul24
    s32: _Chain_mul24
    u32: _Chain_mul24
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_multimem_ld_reduce:
    """`multimem_ld_reduce` — 3 entries sharing this mnemonic; PTX puts their difference in the
    operand list, so the call selects one. Shapes: (d, addr); (*__operands)
    """

    acc__f32: _Chain_multimem_ld_reduce
    acquire: _Chain_multimem_ld_reduce
    add: _Chain_multimem_ld_reduce
    and_: _Chain_multimem_ld_reduce
    b32: _Chain_multimem_ld_reduce
    b64: _Chain_multimem_ld_reduce
    bf16: _Chain_multimem_ld_reduce
    bf16x2: _Chain_multimem_ld_reduce
    cluster: _Chain_multimem_ld_reduce
    cta: _Chain_multimem_ld_reduce
    f16: _Chain_multimem_ld_reduce
    f16x2: _Chain_multimem_ld_reduce
    f32: _Chain_multimem_ld_reduce
    f64: _Chain_multimem_ld_reduce
    global_: _Chain_multimem_ld_reduce
    gpu: _Chain_multimem_ld_reduce
    max: _Chain_multimem_ld_reduce
    min: _Chain_multimem_ld_reduce
    or_: _Chain_multimem_ld_reduce
    relaxed: _Chain_multimem_ld_reduce
    s32: _Chain_multimem_ld_reduce
    s64: _Chain_multimem_ld_reduce
    sys: _Chain_multimem_ld_reduce
    u32: _Chain_multimem_ld_reduce
    u64: _Chain_multimem_ld_reduce
    v2: _Chain_multimem_ld_reduce
    v4: _Chain_multimem_ld_reduce
    v8: _Chain_multimem_ld_reduce
    weak: _Chain_multimem_ld_reduce
    xor: _Chain_multimem_ld_reduce
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_multimem_red:
    """`multimem_red` — 3 entries sharing this mnemonic; PTX puts their difference in the
    operand list, so the call selects one. Shapes: (addr, b); (*__operands)
    """

    add: _Chain_multimem_red
    and_: _Chain_multimem_red
    b32: _Chain_multimem_red
    b64: _Chain_multimem_red
    bf16: _Chain_multimem_red
    bf16x2: _Chain_multimem_red
    cluster: _Chain_multimem_red
    cta: _Chain_multimem_red
    f16: _Chain_multimem_red
    f16x2: _Chain_multimem_red
    f32: _Chain_multimem_red
    f64: _Chain_multimem_red
    global_: _Chain_multimem_red
    gpu: _Chain_multimem_red
    max: _Chain_multimem_red
    min: _Chain_multimem_red
    or_: _Chain_multimem_red
    relaxed: _Chain_multimem_red
    release: _Chain_multimem_red
    s32: _Chain_multimem_red
    s64: _Chain_multimem_red
    sys: _Chain_multimem_red
    u32: _Chain_multimem_red
    u64: _Chain_multimem_red
    v2: _Chain_multimem_red
    v4: _Chain_multimem_red
    v8: _Chain_multimem_red
    xor: _Chain_multimem_red
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_multimem_st:
    """`multimem_st` — 3 entries sharing this mnemonic; PTX puts their difference in the
    operand list, so the call selects one. Shapes: (addr, b); (*__operands)
    """

    b32: _Chain_multimem_st
    b64: _Chain_multimem_st
    bf16: _Chain_multimem_st
    bf16x2: _Chain_multimem_st
    cluster: _Chain_multimem_st
    cta: _Chain_multimem_st
    f16: _Chain_multimem_st
    f16x2: _Chain_multimem_st
    f32: _Chain_multimem_st
    f64: _Chain_multimem_st
    global_: _Chain_multimem_st
    gpu: _Chain_multimem_st
    relaxed: _Chain_multimem_st
    release: _Chain_multimem_st
    s32: _Chain_multimem_st
    s64: _Chain_multimem_st
    sys: _Chain_multimem_st
    u32: _Chain_multimem_st
    u64: _Chain_multimem_st
    v2: _Chain_multimem_st
    v4: _Chain_multimem_st
    v8: _Chain_multimem_st
    weak: _Chain_multimem_st
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_neg:
    """`neg` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a)
    """

    bf16: _Chain_neg
    bf16x2: _Chain_neg
    f16: _Chain_neg
    f16x2: _Chain_neg
    f32: _Chain_neg
    f64: _Chain_neg
    ftz: _Chain_neg
    s16: _Chain_neg
    s32: _Chain_neg
    s64: _Chain_neg
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_not:
    """`not` — type∈{pred,b16,b32,b64}"""

    b16: _Chain_not
    b32: _Chain_not
    b64: _Chain_not
    pred: _Chain_not
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_or:
    """`or` — type∈{pred,b16,b32,b64}"""

    b16: _Chain_or
    b32: _Chain_or
    b64: _Chain_or
    pred: _Chain_or
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_popc:
    """`popc` — type∈{b32,b64}"""

    b32: _Chain_popc
    b64: _Chain_popc
    def __call__(
        self,
        d: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_prefetch:
    """`prefetch` — space∈{global,local,const,param} (opt); level∈{L1,L2} (opt);
    evict∈{L2::evict_last,L2::evict_normal} (opt); tensormap∈{tensormap} (opt) — Each
    prefetch syntax line names exactly one target (PTX ISA 9.7.9.16).
    `.level::eviction_priority` stays bound to `.global` on purpose: its syntax line is
    `prefetch.global.level::eviction_priority`, with `.global` written in rather than the
    `{.ss}` that the `ld` lines carry. Generic addressing is not offered there, so neither
    is it here.
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

class _Chain_prefetchu:
    """`prefetchu` — level∈{L1}"""

    L1: _Chain_prefetchu
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_prmt:
    """`prmt` — type∈{b32}; mode∈{f4e,b4e,rc8,ecl,ecr,rc16} (opt)"""

    b32: _Chain_prmt
    b4e: _Chain_prmt
    ecl: _Chain_prmt
    ecr: _Chain_prmt
    f4e: _Chain_prmt
    rc16: _Chain_prmt
    rc8: _Chain_prmt
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_rcp:
    """`rcp` — mode∈{approx,rn,rz,rm,rp}; ftz∈{ftz} (opt); type∈{f32,f64} — rcp's four syntax
    lines, across two ISA subsections. rcp.approx{.ftz}.f32 d, a; rcp.rnd{.ftz}.f32 d, a;
    (9.7.3.13) rcp.rnd.f64 d, a; rcp.approx.ftz.f64 d, a; (9.7.3.14) The ISA gives the last
    one a subsection of its own because it is a different *computation* -- a gross
    approximation off the top 20 mantissa bits, with its own corner-case table -- but its
    syntax is one more cell of this grid, and the shape (`d, a`) is unchanged. So it lives
    here, with the mandatory `.ftz` of its syntax line enforced below rather than by a
    second entry that would render identically.
    """

    approx: _Chain_rcp
    f32: _Chain_rcp
    f64: _Chain_rcp
    ftz: _Chain_rcp
    rm: _Chain_rcp
    rn: _Chain_rcp
    rp: _Chain_rcp
    rz: _Chain_rcp
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_red:
    """`red` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands)
    """

    L2__cache_hint: _Chain_red
    add: _Chain_red
    and_: _Chain_red
    b32: _Chain_red
    b64: _Chain_red
    bf16: _Chain_red
    bf16x2: _Chain_red
    cluster: _Chain_red
    cta: _Chain_red
    dec: _Chain_red
    f16: _Chain_red
    f16x2: _Chain_red
    f32: _Chain_red
    f64: _Chain_red
    global_: _Chain_red
    gpu: _Chain_red
    inc: _Chain_red
    max: _Chain_red
    min: _Chain_red
    noftz: _Chain_red
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
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_red_async:
    """`red_async` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (addr, value, mbar); (addr, value)
    """

    add: _Chain_red_async
    and_: _Chain_red_async
    b32: _Chain_red_async
    cluster: _Chain_red_async
    dec: _Chain_red_async
    global_: _Chain_red_async
    gpu: _Chain_red_async
    inc: _Chain_red_async
    max: _Chain_red_async
    mbarrier__complete_tx__bytes: _Chain_red_async
    min: _Chain_red_async
    mmio: _Chain_red_async
    or_: _Chain_red_async
    relaxed: _Chain_red_async
    release: _Chain_red_async
    s32: _Chain_red_async
    s64: _Chain_red_async
    shared__cluster: _Chain_red_async
    sys: _Chain_red_async
    u32: _Chain_red_async
    u64: _Chain_red_async
    xor: _Chain_red_async
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_redux_sync:
    """`redux_sync` — 3 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a, membermask)
    """

    NaN: _Chain_redux_sync
    abs: _Chain_redux_sync
    add: _Chain_redux_sync
    and_: _Chain_redux_sync
    b32: _Chain_redux_sync
    f32: _Chain_redux_sync
    max: _Chain_redux_sync
    min: _Chain_redux_sync
    or_: _Chain_redux_sync
    s32: _Chain_redux_sync
    u32: _Chain_redux_sync
    xor: _Chain_redux_sync
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_rem:
    """`rem` — type∈{u16,u32,u64,s16,s32,s64}"""

    s16: _Chain_rem
    s32: _Chain_rem
    s64: _Chain_rem
    u16: _Chain_rem
    u32: _Chain_rem
    u64: _Chain_rem
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_rsqrt:
    """`rsqrt` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32,f64}"""

    approx: _Chain_rsqrt
    f32: _Chain_rsqrt
    f64: _Chain_rsqrt
    ftz: _Chain_rsqrt
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_sad:
    """`sad` — type∈{u16,u32,u64,s16,s32,s64}"""

    s16: _Chain_sad
    s32: _Chain_sad
    s64: _Chain_sad
    u16: _Chain_sad
    u32: _Chain_sad
    u64: _Chain_sad
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_selp:
    """`selp` — type∈{b16,b32,b64,u16,u32,u64,s16,s32,s64,f32,f64}"""

    b16: _Chain_selp
    b32: _Chain_selp
    b64: _Chain_selp
    f32: _Chain_selp
    f64: _Chain_selp
    s16: _Chain_selp
    s32: _Chain_selp
    s64: _Chain_selp
    u16: _Chain_selp
    u32: _Chain_selp
    u64: _Chain_selp
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_set:
    """`set` — 4 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a, b); (d, a, b, c)
    """

    and_: _Chain_set
    b16: _Chain_set
    b32: _Chain_set
    b64: _Chain_set
    bf16: _Chain_set
    bf16x2: _Chain_set
    eq: _Chain_set
    equ: _Chain_set
    f16: _Chain_set
    f16x2: _Chain_set
    f32: _Chain_set
    f64: _Chain_set
    ftz: _Chain_set
    ge: _Chain_set
    geu: _Chain_set
    gt: _Chain_set
    gtu: _Chain_set
    hi: _Chain_set
    hs: _Chain_set
    le: _Chain_set
    leu: _Chain_set
    lo: _Chain_set
    ls: _Chain_set
    lt: _Chain_set
    ltu: _Chain_set
    nan: _Chain_set
    ne: _Chain_set
    neu: _Chain_set
    num: _Chain_set
    or_: _Chain_set
    s16: _Chain_set
    s32: _Chain_set
    s64: _Chain_set
    u16: _Chain_set
    u32: _Chain_set
    u64: _Chain_set
    xor: _Chain_set
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_setmaxnreg:
    """`setmaxnreg` — action∈{inc,dec}; sync∈{sync}; aligned∈{aligned}; type∈{u32}"""

    aligned: _Chain_setmaxnreg
    dec: _Chain_setmaxnreg
    inc: _Chain_setmaxnreg
    sync: _Chain_setmaxnreg
    u32: _Chain_setmaxnreg
    def __call__(self, nreg: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_setp:
    """`setp` — 8 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (p, a, b); (p, q, a, b); (p, a, b, c); (p, q, a, b, c)
    """

    and_: _Chain_setp
    b16: _Chain_setp
    b32: _Chain_setp
    b64: _Chain_setp
    bf16: _Chain_setp
    bf16x2: _Chain_setp
    eq: _Chain_setp
    equ: _Chain_setp
    f16: _Chain_setp
    f16x2: _Chain_setp
    f32: _Chain_setp
    f64: _Chain_setp
    ftz: _Chain_setp
    ge: _Chain_setp
    geu: _Chain_setp
    gt: _Chain_setp
    gtu: _Chain_setp
    hi: _Chain_setp
    hs: _Chain_setp
    le: _Chain_setp
    leu: _Chain_setp
    lo: _Chain_setp
    ls: _Chain_setp
    lt: _Chain_setp
    ltu: _Chain_setp
    nan: _Chain_setp
    ne: _Chain_setp
    neu: _Chain_setp
    num: _Chain_setp
    or_: _Chain_setp
    s16: _Chain_setp
    s32: _Chain_setp
    s64: _Chain_setp
    u16: _Chain_setp
    u32: _Chain_setp
    u64: _Chain_setp
    xor: _Chain_setp
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_shf:
    """`shf` — dir∈{l,r}; mode∈{clamp,wrap}; type∈{b32}"""

    b32: _Chain_shf
    clamp: _Chain_shf
    l: _Chain_shf
    r: _Chain_shf
    wrap: _Chain_shf
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_shfl_sync:
    """`shfl_sync` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a, b, c, membermask); (d, p, a, b, c,
    membermask)
    """

    b32: _Chain_shfl_sync
    bfly: _Chain_shfl_sync
    down: _Chain_shfl_sync
    idx: _Chain_shfl_sync
    up: _Chain_shfl_sync
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_shl:
    """`shl` — type∈{b16,b32,b64}"""

    b16: _Chain_shl
    b32: _Chain_shl
    b64: _Chain_shl
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_shr:
    """`shr` — type∈{b16,b32,b64,u16,u32,u64,s16,s32,s64}"""

    b16: _Chain_shr
    b32: _Chain_shr
    b64: _Chain_shr
    s16: _Chain_shr
    s32: _Chain_shr
    s64: _Chain_shr
    u16: _Chain_shr
    u32: _Chain_shr
    u64: _Chain_shr
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_sin:
    """`sin` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32}"""

    approx: _Chain_sin
    f32: _Chain_sin
    ftz: _Chain_sin
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_slct:
    """`slct` — ftz∈{ftz} (opt); dtype∈{b16,b32,b64,u16,u32,u64,s16,s32,s64,f32,f64};
    ctype∈{s32,f32} — slct's two lines (ISA 9.7.6.4), which differ only in the selector
    type. slct.dtype.s32 d, a, b, c; slct{.ftz}.dtype.f32 d, a, b, c; `.ftz` is spelled on
    the .f32 selector line alone -- there is nothing to flush when the sign being tested is
    an integer's.
    """

    b16: _Chain_slct
    b32: _Chain_slct
    b64: _Chain_slct
    f32: _Chain_slct
    f64: _Chain_slct
    ftz: _Chain_slct
    s16: _Chain_slct
    s32: _Chain_slct
    s64: _Chain_slct
    u16: _Chain_slct
    u32: _Chain_slct
    u64: _Chain_slct
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        c: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_sqrt:
    """`sqrt` — mode∈{approx,rn,rz,rm,rp}; ftz∈{ftz} (opt); type∈{f32,f64} — sqrt's three lines
    (PTX ISA 9.7.3.15). sqrt.approx{.ftz}.f32 d, a; sqrt.rnd{.ftz}.f32 d, a; sqrt.rnd.f64 d,
    a; Unlike rcp, there is no f64 approximation at any spelling -- 9.7.3.15 is the whole of
    sqrt, and it offers `.approx` on the .f32 line only.
    """

    approx: _Chain_sqrt
    f32: _Chain_sqrt
    f64: _Chain_sqrt
    ftz: _Chain_sqrt
    rm: _Chain_sqrt
    rn: _Chain_sqrt
    rp: _Chain_sqrt
    rz: _Chain_sqrt
    def __call__(
        self,
        d: Any,
        value: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_st:
    """`st` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (*__operands)
    """

    L1__evict_first: _Chain_st
    L1__evict_last: _Chain_st
    L1__evict_normal: _Chain_st
    L1__evict_unchanged: _Chain_st
    L1__no_allocate: _Chain_st
    L2__cache_hint: _Chain_st
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

class _Chain_st_async:
    """`st_async` — 3 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (addr, b, mbar); (*__operands); (addr, b)
    """

    b128: _Chain_st_async
    b16: _Chain_st_async
    b32: _Chain_st_async
    b64: _Chain_st_async
    f32: _Chain_st_async
    f64: _Chain_st_async
    global_: _Chain_st_async
    gpu: _Chain_st_async
    mbarrier__complete_tx__bytes: _Chain_st_async
    mmio: _Chain_st_async
    release: _Chain_st_async
    s16: _Chain_st_async
    s32: _Chain_st_async
    s64: _Chain_st_async
    shared__cluster: _Chain_st_async
    sys: _Chain_st_async
    u16: _Chain_st_async
    u32: _Chain_st_async
    u64: _Chain_st_async
    v2: _Chain_st_async
    v4: _Chain_st_async
    weak: _Chain_st_async
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
    """`sub` — 3 entries sharing this mnemonic; PTX puts their difference in the operand list,
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
    s16: _Chain_sub
    s32: _Chain_sub
    s64: _Chain_sub
    sat: _Chain_sub
    u16: _Chain_sub
    u32: _Chain_sub
    u64: _Chain_sub
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_szext:
    """`szext` — mode∈{clamp,wrap}; type∈{u32,s32}"""

    clamp: _Chain_szext
    s32: _Chain_szext
    u32: _Chain_szext
    wrap: _Chain_szext
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_tanh:
    """`tanh` — 2 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, value)
    """

    approx: _Chain_tanh
    bf16: _Chain_tanh
    bf16x2: _Chain_tanh
    f16: _Chain_tanh
    f16x2: _Chain_tanh
    f32: _Chain_tanh
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_tcgen05:
    """`tcgen05` — 20 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (dst, ncols); (taddr, ncols); (); (*__operands);
    (taddr, s_desc); (d_tmem, a_desc, b_desc, idesc, sfa_tmem, sfb_tmem, enable_input_d);
    (d_tmem, a_tmem, b_desc, idesc, sfa_tmem, sfb_tmem, enable_input_d); (d_tmem, a_desc,
    b_desc, idesc, enable_input_d, zero_col_mask); (d_tmem, a_tmem, b_desc, idesc,
    enable_input_d, zero_col_mask); (mbar); (mbar, mask)
    """

    aligned: _Chain_tcgen05
    alloc: _Chain_tcgen05
    b32: _Chain_tcgen05
    b4x16_p64: _Chain_tcgen05
    b64: _Chain_tcgen05
    b6x16_p32: _Chain_tcgen05
    b8x16: _Chain_tcgen05
    block16: _Chain_tcgen05
    block32: _Chain_tcgen05
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
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_tensormap_cp_fenceproxy:
    """`tensormap_cp_fenceproxy` — dst∈{global}; src∈{shared::cta}; proxy∈{tensormap::generic};
    sem∈{release}; scope∈{cta,cluster,gpu,sys}; sync∈{sync}; aligned∈{aligned}
    """

    aligned: _Chain_tensormap_cp_fenceproxy
    cluster: _Chain_tensormap_cp_fenceproxy
    cta: _Chain_tensormap_cp_fenceproxy
    global_: _Chain_tensormap_cp_fenceproxy
    gpu: _Chain_tensormap_cp_fenceproxy
    release: _Chain_tensormap_cp_fenceproxy
    shared__cta: _Chain_tensormap_cp_fenceproxy
    sync: _Chain_tensormap_cp_fenceproxy
    sys: _Chain_tensormap_cp_fenceproxy
    tensormap__generic: _Chain_tensormap_cp_fenceproxy
    def __call__(self, dst_mem: Any, src_mem: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_tensormap_replace:
    """`tensormap_replace` — 8 entries sharing this mnemonic; PTX puts their difference in the
    operand list, so the call selects one. Shapes: (addr, new_val); (addr, ord, new_val)
    """

    b1024: _Chain_tensormap_replace
    b32: _Chain_tensormap_replace
    b64: _Chain_tensormap_replace
    box_dim: _Chain_tensormap_replace
    element_stride: _Chain_tensormap_replace
    elemtype: _Chain_tensormap_replace
    fill_mode: _Chain_tensormap_replace
    global_: _Chain_tensormap_replace
    global_address: _Chain_tensormap_replace
    global_dim: _Chain_tensormap_replace
    global_stride: _Chain_tensormap_replace
    interleave_layout: _Chain_tensormap_replace
    rank: _Chain_tensormap_replace
    shared__cta: _Chain_tensormap_replace
    swizzle_mode: _Chain_tensormap_replace
    tile: _Chain_tensormap_replace
    def __call__(self, *args: Any, pred: Any = None) -> None: ...

class _Chain_testp:
    """`testp` — op∈{finite,infinite,number,notanumber,normal,subnormal}; type∈{f32,f64}"""

    f32: _Chain_testp
    f64: _Chain_testp
    finite: _Chain_testp
    infinite: _Chain_testp
    normal: _Chain_testp
    notanumber: _Chain_testp
    number: _Chain_testp
    subnormal: _Chain_testp
    def __call__(
        self,
        p: Any,
        a: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _Chain_vote_sync:
    """`vote_sync` — 2 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (d, a, membermask)
    """

    all: _Chain_vote_sync
    any: _Chain_vote_sync
    b32: _Chain_vote_sync
    ballot: _Chain_vote_sync
    pred: _Chain_vote_sync
    uni: _Chain_vote_sync
    def __call__(self, *args: Any, pred: Any = None, preserve_dst: bool = False) -> None: ...

class _Chain_wgmma:
    """`wgmma` — 19 entries sharing this mnemonic; PTX puts their difference in the operand
    list, so the call selects one. Shapes: (*__operands); (); (group)
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

class _Chain_xor:
    """`xor` — type∈{pred,b16,b32,b64}"""

    b16: _Chain_xor
    b32: _Chain_xor
    b64: _Chain_xor
    pred: _Chain_xor
    def __call__(
        self,
        d: Any,
        a: Any,
        b: Any,
        *args: Any,
        pred: Any = None,
        preserve_dst: bool = False,
    ) -> None: ...

class _PTX:
    abs: _Chain_abs
    activemask: _Chain_activemask
    add: _Chain_add
    and_: _Chain_and
    applypriority: _Chain_applypriority
    atom: _Chain_atom
    bar: _Chain_bar
    barrier: _Chain_barrier
    bfe: _Chain_bfe
    bfi: _Chain_bfi
    bfind: _Chain_bfind
    bmsk: _Chain_bmsk
    brev: _Chain_brev
    clusterlaunchcontrol: _Chain_clusterlaunchcontrol
    clz: _Chain_clz
    cnot: _Chain_cnot
    copysign: _Chain_copysign
    cos: _Chain_cos
    cp: _Chain_cp
    createpolicy: _Chain_createpolicy
    cvt: _Chain_cvt
    cvt_pack: _Chain_cvt_pack
    cvta: _Chain_cvta
    discard: _Chain_discard
    div: _Chain_div
    dp2a: _Chain_dp2a
    dp4a: _Chain_dp4a
    elect_sync: _Chain_elect_sync
    ex2: _Chain_ex2
    fence: _Chain_fence
    fma: _Chain_fma
    fns: _Chain_fns
    getctarank: _Chain_getctarank
    griddepcontrol: _Chain_griddepcontrol
    isspacep: _Chain_isspacep
    ld: _Chain_ld
    ldmatrix: _Chain_ldmatrix
    ldu: _Chain_ldu
    lg2: _Chain_lg2
    lop3: _Chain_lop3
    mad: _Chain_mad
    mad24: _Chain_mad24
    mapa: _Chain_mapa
    match: _Chain_match
    max: _Chain_max
    mbarrier: _Chain_mbarrier
    min: _Chain_min
    mma: _Chain_mma
    mov: _Chain_mov
    mul: _Chain_mul
    mul24: _Chain_mul24
    multimem_ld_reduce: _Chain_multimem_ld_reduce
    multimem_red: _Chain_multimem_red
    multimem_st: _Chain_multimem_st
    neg: _Chain_neg
    not_: _Chain_not
    or_: _Chain_or
    popc: _Chain_popc
    prefetch: _Chain_prefetch
    prefetchu: _Chain_prefetchu
    prmt: _Chain_prmt
    rcp: _Chain_rcp
    red: _Chain_red
    red_async: _Chain_red_async
    redux_sync: _Chain_redux_sync
    rem: _Chain_rem
    rsqrt: _Chain_rsqrt
    sad: _Chain_sad
    selp: _Chain_selp
    set: _Chain_set
    setmaxnreg: _Chain_setmaxnreg
    setp: _Chain_setp
    shf: _Chain_shf
    shfl_sync: _Chain_shfl_sync
    shl: _Chain_shl
    shr: _Chain_shr
    sin: _Chain_sin
    slct: _Chain_slct
    sqrt: _Chain_sqrt
    st: _Chain_st
    st_async: _Chain_st_async
    st_bulk: _Chain_st_bulk
    stmatrix: _Chain_stmatrix
    sub: _Chain_sub
    szext: _Chain_szext
    tanh: _Chain_tanh
    tcgen05: _Chain_tcgen05
    tensormap_cp_fenceproxy: _Chain_tensormap_cp_fenceproxy
    tensormap_replace: _Chain_tensormap_replace
    testp: _Chain_testp
    vote_sync: _Chain_vote_sync
    wgmma: _Chain_wgmma
    xor: _Chain_xor
    def addr(self, base: Any, byte_offset: Any) -> Any: ...
    def __getitem__(self, text: str) -> Any: ...

ptx: _PTX

# Every other tvm.script.tirx member stays dynamically typed, as before.
def __getattr__(name: str) -> Any: ...
