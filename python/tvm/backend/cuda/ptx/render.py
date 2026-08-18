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
"""Pure CUDA-helper rendering for the ptx dialect.

tvm-free (imports only :mod:`.table`), so it is shared by the codegen engine
and by :mod:`.gen_helpers`, which dumps the generated helpers for humans to
inspect without compiling a kernel.
"""

import collections
import itertools
from typing import NamedTuple

from .table import (
    InstructionEntry,
    canonical_dtypes,
    imm_slots,
    lanes_of,
    mods,
    operand_space,
    operand_type,
)


class CBinding(NamedTuple):
    """How one TVM dtype crosses the C / inline-asm boundary.

    ``carrier`` is the C type actually bound to the asm register. When it equals
    ``c_type`` the value binds its register directly; otherwise it rides a
    differently-typed register and converts at each boundary, because 8-bit
    values have no constraint letter of their own and __half/__nv_bfloat16
    cannot bind one at all (nvcc: "more than one conversion function applies").

    Every conversion here is a bit pun, never an arithmetic cast. That is the
    whole point of the dtype axis: handing a float to a uint32_t parameter is a
    *numeric* conversion and emits `cvt.rzi.u32.f32`, silently changing the
    value. ``suffix`` is the token that names this dtype in a helper name.
    """

    c_type: str
    constraint: str
    carrier: str
    to_carrier: str  # format string, applied on the way into the asm
    from_carrier: str  # format string, applied on the way out
    suffix: str


# Named fields, not a positional tuple: this is the same table shape whose
# modifier-token twin needed `tokens_for`, and a new column would otherwise
# shift every unpacking site.
C_BINDING = {
    "uint8": CBinding("uint8_t", "h", "uint16_t", "(uint16_t){}", "(uint8_t){}", "u8"),
    "int8": CBinding("int8_t", "h", "int16_t", "(int16_t){}", "(int8_t){}", "s8"),
    "uint16": CBinding("uint16_t", "h", "uint16_t", "{}", "{}", "u16"),
    "int16": CBinding("int16_t", "h", "int16_t", "{}", "{}", "s16"),
    "float16": CBinding(
        "__half", "h", "uint16_t", "__half_as_ushort({})", "__ushort_as_half({})", "f16"
    ),
    "bfloat16": CBinding(
        "__nv_bfloat16",
        "h",
        "uint16_t",
        "__bfloat16_as_ushort({})",
        "__ushort_as_bfloat16({})",
        "bf16",
    ),
    "uint32": CBinding("uint32_t", "r", "uint32_t", "{}", "{}", "u32"),
    "int32": CBinding("int32_t", "r", "int32_t", "{}", "{}", "s32"),
    "float32": CBinding("float", "f", "float", "{}", "{}", "f32"),
    "uint64": CBinding("uint64_t", "l", "uint64_t", "{}", "{}", "u64"),
    "int64": CBinding("int64_t", "l", "int64_t", "{}", "{}", "s64"),
    "float64": CBinding("double", "d", "double", "{}", "{}", "f64"),
    # 128-bit: the "q" constraint needs __int128 support in the host compiler.
    "uint128": CBinding("__uint128_t", "q", "__uint128_t", "{}", "{}", "u128"),
    "int128": CBinding("__int128_t", "q", "__int128_t", "{}", "{}", "s128"),
}


class Bridge(NamedTuple):
    """How one PTX register class that inline asm cannot bind is reached.

    ``C_BINDING`` above answers "what C type carries this value"; this answers
    the other half, "what does the instruction actually name". Two ISA types
    need it, for the same reason from opposite ends of the constraint
    alphabet: ``.pred`` has no letter at all, and ``.b8`` is below the
    narrowest one (``"h"``). Both are reached the way hand-written PTX reaches
    them -- declare a register of the real class inside the asm block, and
    convert between it and the bound carrier.

    ``read``/``write`` are the local register's name (``{slot}`` = the operand
    name, ``{n}`` = how many of this direction the block already declared).
    ``into`` runs before the instruction, ``out_of`` after it; both take
    ``{reg}`` and ``{idx}``. ``rw`` fires both, in that order.

    These conversions are the asm block's sanctioned exceptions to the
    single-instruction invariant -- ``@p``'s own setp was the first of them.
    Each one is a *measured* fact, not a derivation: ptxas rejects the obvious
    alternatives (a wider register for a ``.b8`` operand; ``mov`` as the b8
    bridge), so a new row here means a new ptxas probe.
    """

    reg_class: str
    read: str
    write: str
    into: str
    out_of: str


BRIDGE = {
    # `.pred`: setp on the way in, selp materializing 0/1 on the way out --
    # the same conversion `@p` performs for its own guard predicate.
    "pred": Bridge(
        ".pred", "ps{n}", "pd{n}", "setp.ne.b32 {reg}, %{idx}, 0;", "selp.b32 %{idx}, 1, 0, {reg};"
    ),
    # `.e2m1x2`: the ISA types this operand .b8 (9.7.9.22:92 for the
    # destination, :101 for the source). Its general prose says a wider
    # register may be used and names no exception for e2m1x2 (:476-486), but
    # the toolchain disagrees, so the width here is measured, not read:
    # ptxas 13.2 at -arch=sm_100a answers "Arguments mismatch for instruction
    # 'cvt'" for BOTH carrier widths ("h" and "r") in BOTH directions, across
    # cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2} and
    # cvt.rn.{f16x2,bf16x2}.e2m1x2 -- ten probes, ten rejections.
    #
    # `mov` is not the bridge either: mov.b16 / mov.u16 between a `.reg .b8`
    # and a 16-bit register are the same "Arguments mismatch" (same probe).
    # `cvt.u8.u16` / `cvt.u16.u8` assemble, which settles the question the
    # deleted legacy implementation raised -- its idiom was the correct
    # bridge, not a defect to be cleaned up. The C side needs no help: the
    # uint8 row of C_BINDING already carries the uint16 and its casts.
    "e2m1x2": Bridge(
        ".b8", "raw_{slot}", "raw_{slot}", "cvt.u8.u16 {reg}, %{idx};", "cvt.u16.u8 %{idx}, {reg};"
    ),
}


def _normalize_addr_offsets(entry: InstructionEntry, addr_offsets) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalize ``(logical_address_slot, byte_offset)`` metadata."""
    address_slots = tuple(slot for slot in entry.operands if slot.kind == "addr")
    normalized = {}
    for logical_slot, offset in addr_offsets or ():
        if isinstance(logical_slot, bool) or not isinstance(logical_slot, int):
            raise ValueError(f"{entry.name}: address slot index must be an integer")
        if not 0 <= logical_slot < len(address_slots):
            raise ValueError(f"{entry.name}: no address slot {logical_slot}")
        slot = address_slots[logical_slot]
        if not slot.allow_imm_offset:
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' does not support an immediate offset"
            )
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError(f"{entry.name}: address byte offset must be an integer")
        if not -(1 << 31) <= offset <= (1 << 31) - 1:
            raise ValueError(f"{entry.name}: address byte offset {offset} is outside int32 range")
        if logical_slot in normalized:
            raise ValueError(f"{entry.name}: duplicate address slot {logical_slot}")
        if offset:
            normalized[logical_slot] = offset
    return tuple(sorted(normalized.items()))


def _helper_name(
    entry: InstructionEntry,
    written,
    imms,
    dtypes,
    canonical,
    mod_map,
    sinks=(),
    addr_offsets=(),
) -> str:
    """The helper's C identifier: the instruction's ISA identity, plus a
    signature discriminator only when it is no longer enough.

    The opcode alone stopped being enough once an operand could take several
    dtypes; a non-canonical choice then names *every* typed operand,
    positionally, because naming only the ones that changed collides whenever
    two operands swap which of them is non-canonical (atom's d and b do
    exactly that).

    The name keys off the table name rather than the mnemonic. For every
    single-shape family the two agree (st.bulk normalizes to st_bulk anyway);
    it matters only where several entries share a mnemonic because PTX puts
    their difference in the operand list, as `mov`'s pack/unpack shapes do.
    "m" for minus: wgmma's imm-scale-a/b take the value -1, and "-" cannot
    appear in a C identifier.
    """
    # Only the operands that are actually there. A length function may resolve
    # to zero -- that is how a bracketed-optional operand disappears when its
    # qualifier is absent -- and an operand with no C parameter cannot be part
    # of the signature this discriminator exists to name. Counting it anyway
    # would rename every helper of an entry the moment such an operand were
    # added to it, present or not.
    present = [
        (dtype, canon)
        for slot, dtype, canon in zip(entry.typed_operands, dtypes, canonical, strict=True)
        if lanes_of(slot, mod_map)
    ]
    # A sunk lane has no C parameter, so the mask is part of the signature and
    # has to be in the name. Empty mask -> nothing appended, which is what
    # keeps every pre-existing helper name exactly as it was.
    sunk = [
        f"sink_{name}" + "".join(str(lane) for _, lane in sorted(group))
        for name, group in itertools.groupby(sorted(sinks), key=lambda pair: pair[0])
    ]
    offset_suffixes = [
        f"addr{logical_slot}_{'p' if offset > 0 else 'm'}{abs(offset)}"
        for logical_slot, offset in addr_offsets
    ]
    isa_name = [entry.name, *written, *(imms or ()), *offset_suffixes, *sunk]
    discriminator = (
        []
        if all(dtype == canon for dtype, canon in present)
        else [C_BINDING[dtype].suffix for dtype, _ in present]
    )
    return "tvm_builtin_ptx_" + "_".join([*isa_name, *discriminator]).replace("::", "__").replace(
        ".", "_"
    ).replace("-", "m")


def render_variant(
    entry: InstructionEntry,
    tokens,
    predicated=False,
    dtypes=None,
    imms=None,
    sinks=frozenset(),
    preserve_dst=False,
    addr_offsets=(),
):
    """Render one variant: ``(opcode, helper_name, helper_source)``.

    Every helper is ``void`` and its C parameter list is the PTX operand list
    in order, so the generated call reads like the PTX text it wraps.
    Destinations (``rw="w"``) and accumulators (``rw="rw"``, read and
    written in place via a "+" constraint) are taken by reference; the caller passes a
    writable lvalue.

    ``dtypes`` picks one TVM dtype per typed operand (see ``table.dtype_combos``);
    None means the canonical choice, which is what every non-bit-typed operand
    has anyway. A non-canonical choice appends the dtypes to the helper name, so
    the names a table without bit-typed operands would produce are untouched.

    ``imms`` picks one value per caller-chosen immediate (see
    ``table.imm_slots``); the value is part of the instruction's identity, so
    it lands in the asm text and in the helper name, and the helper has no C
    parameter for it.

    ``predicated`` is a framework-level axis (never in the table): the helper
    gains a trailing ``uint32_t __pred`` operand, and the instruction is
    guarded with ``.reg .pred p; setp.ne.b32 p, %N, 0; @p ...``. A written
    destination uses the normal write-only binding by default, which states
    that the caller will not consume the inactive value before selecting or
    guarding it. ``preserve_dst`` instead binds read-write and retains the old
    value on the inactive path.
    """
    mod_map = mods(entry, tokens)
    addr_offsets = _normalize_addr_offsets(entry, addr_offsets)
    addr_offset_of = dict(addr_offsets)
    written = [tok for tok in tokens if tok]
    opcode = ".".join([entry.ptx_name, *written])
    canonical = canonical_dtypes(entry, tokens)
    if dtypes is None:
        dtypes = canonical
    if entry.raw_render is not None:
        # A hand-written body (see InstructionEntry.raw_render). The name is
        # derived the same way so dispatch, stubs and certification do not
        # need to know the difference.
        assert not predicated, f"{opcode}: raw entries have no @p twin"
        helper = _helper_name(entry, written, imms, dtypes, canonical, mod_map, sinks, addr_offsets)
        return opcode, helper, entry.raw_render(entry, opcode, helper, tokens, tuple(dtypes))
    # A helper name is the instruction's ISA identity plus, only when it is no
    # longer enough, a signature discriminator. The opcode alone stopped being
    # enough once an operand could take several dtypes; a non-canonical choice
    # then names *every* typed operand, positionally, because naming only the
    # ones that changed collides whenever two operands swap which of them is
    # non-canonical (atom's d and b do exactly that).
    imm_of = dict(zip(imm_slots(entry), imms or (), strict=True))
    helper = _helper_name(entry, written, imms, dtypes, canonical, mod_map, sinks, addr_offsets)
    assert not preserve_dst or entry.has_dst, "preserve_dst requires a written destination"
    if predicated:
        if entry.has_dst:
            helper += "_pred_keep" if preserve_dst else "_pred_undef"
        else:
            helper += "_pred"
    else:
        assert not preserve_dst, "preserve_dst requires predication"

    params, inputs, outputs, rendered = [], [], [], []
    pre, post = [], []  # C-side carrier declarations / bit puns
    # Bridge pieces, all inside the asm block: declarations of the register
    # classes inline asm cannot bind, the conversions in (before the
    # instruction) and out (after it). See `BRIDGE`.
    bridge_decls, asm_pre, asm_post = [], [], []
    bridge_counts: dict[str, int] = collections.defaultdict(int)
    dtype_of = dict(zip(entry.typed_operands, dtypes, strict=True))
    idx = 0
    logical_addr_slot = 0
    for slot in entry.operands:
        pname = f"__{slot.name}"
        if slot.kind == "imm":
            # An immediate lives in the instruction text, never a C parameter.
            # A literal is table-owned; a choices/open value arrives through
            # `imms` and is baked here (and into the helper name).
            rendered.append((slot, slot.literal if slot.literal is not None else imm_of[slot]))
            continue
        regs = []
        n_lanes = lanes_of(slot, mod_map)
        if n_lanes == 0:
            # A length function may return zero: that is how an operand the
            # syntax line brackets as optional disappears when its modifier is
            # absent. It takes no argument and leaves no text behind.
            continue
        # ptxas wants the braces even when a vector resolves to one element
        # ("Vector of size 1 is expected"), so vector-ness is declared, not
        # derived from the resolved count. A length function defaults to
        # vector, which is what every register group wants; an optional scalar
        # says otherwise.
        is_group = (
            slot.vector if slot.vector is not None else (callable(slot.lanes) or slot.lanes > 1)
        )
        for lane in range(n_lanes):
            # One operand, `lanes` registers: PTX writes the group in the
            # operand list, so a lane is a C parameter but not an operand.
            lname = f"{pname}{lane}" if is_group else pname
            if (slot.name, lane) in sinks:
                # `_`: the ISA's "this element is discarded". No C parameter,
                # no constraint, no `%k` -- the instruction names the symbol
                # directly, which is why the mask is part of the signature.
                regs.append("_")
                continue
            if slot.kind == "addr":
                # A tmem address rides the same 32-bit carrier a shared window
                # address does; only its provenance differs.
                if operand_space(slot, mod_map).startswith(("shared", "tmem")):
                    params.append(f"uint32_t {lname}")
                    inputs.append(f'"r"({lname})')
                else:
                    params.append(f"const void* {lname}")
                    inputs.append(f'"l"({lname})')
            elif slot.kind == "ptr":
                params.append(f"const void* {lname}")
                inputs.append(f'"l"({lname})')
            elif slot.rw == "rw" or (preserve_dst and slot.rw == "w"):
                # A register the instruction both reads and writes -- an
                # in-place accumulator. "+" tells the compiler the prior value
                # is live, which "=" would declare dead; it is also what makes
                # @p safe here (a false predicate leaves the old value intact).
                cb = C_BINDING[dtype_of[slot]]
                if cb.carrier != cb.c_type:
                    # A carrier round-trips through a differently-typed local,
                    # which has no coherent read-modify-write story; the dtypes
                    # that need one are refused rather than half-supported.
                    raise ValueError(
                        f"{entry.name}: read-write destination '{slot.name}' cannot take dtype "
                        f"{dtype_of[slot]} (it binds through a carrier)"
                    )
                params.append(f"{cb.c_type}& {lname}")
                outputs.append(f'"+{cb.constraint}"({lname})')
            elif slot.rw == "w":
                cb = C_BINDING[dtype_of[slot]]
                c_ty, constraint, carrier = cb.c_type, cb.constraint, cb.carrier
                params.append(f"{c_ty}& {lname}")
                if carrier == c_ty:
                    outputs.append(f'"={constraint}"({lname})')
                else:
                    # The value cannot bind this register class directly (8-bit
                    # has no constraint letter; __half binds none at all), so the
                    # asm writes a carrier local that is bit-punned back out.
                    reg = f"{lname}_reg"
                    pre.append(f"{carrier} {reg};")
                    outputs.append(f'"={constraint}"({reg})')
                    post.append(f"{lname} = {cb.from_carrier.format(reg)};")
            else:  # rw == "r", an ordinary register input
                cb = C_BINDING[dtype_of[slot]]
                params.append(f"{cb.c_type} {lname}")
                inputs.append(f'"{cb.constraint}"({cb.to_carrier.format(lname)})')
            bridge = BRIDGE.get(operand_type(slot, mod_map)) if slot.kind == "reg" else None
            if bridge is None:
                if slot.kind == "addr" and slot.bracket is None:
                    offset = addr_offset_of.get(logical_addr_slot)
                    regs.append(f"[%{idx}{f'+{offset}' if offset is not None else ''}]")
                else:
                    regs.append(f"%{idx}")
            else:
                # The C side above bound the carrier; the instruction names a
                # block-local register of the class the ISA actually asks for,
                # and the conversions move the value between the two.
                template = bridge.read if slot.rw == "r" else bridge.write
                reg = template.format(slot=slot.name, n=bridge_counts[template])
                bridge_counts[template] += 1
                bridge_decls.append(f".reg {bridge.reg_class} {reg};")
                if slot.rw in ("r", "rw") or (preserve_dst and slot.rw == "w"):
                    asm_pre.append(bridge.into.format(reg=reg, idx=idx))
                if slot.rw in ("w", "rw"):
                    asm_post.append(bridge.out_of.format(reg=reg, idx=idx))
                regs.append(reg)
            idx += 1
        rendered.append((slot, "{" + ", ".join(regs) + "}" if is_group else regs[0]))
        if slot.kind == "addr":
            logical_addr_slot += 1

    # Adjacent slots naming the same `pipe` are one operand written `p|q`
    # (setp's two predicate destinations). Merged first, and into a plain text
    # entry carrying the group's bracket, so the bracket pass below sees one
    # operand where the table declared two.
    piped = []
    for key, members in itertools.groupby(rendered, key=lambda pair: pair[0].pipe):
        members = list(members)
        if key is None:
            piped.extend((slot.bracket, text) for slot, text in members)
        else:
            slot = members[0][0]
            piped.append((slot.bracket, "|".join(text for _, text in members)))

    # Adjacent slots naming the same `bracket` are one composite memory operand:
    # `[tensorMap, {c0, c1}]` is a single PTX operand whose members keep their
    # own registers and constraints.
    ptx_operands = []
    for key, members in itertools.groupby(piped, key=lambda pair: pair[0]):
        texts = [text for _, text in members]
        if key is None:
            ptx_operands.extend(texts)
        else:
            ptx_operands.append(f"[{', '.join(texts)}]")

    instr = f"{opcode} {', '.join(ptx_operands)};" if ptx_operands else f"{opcode};"
    if predicated:
        params.append("uint32_t __pred")
        inputs.append('"r"(__pred)')
        bridge_decls.append(".reg .pred p;")
        asm_pre.append(f"setp.ne.b32 p, %{idx}, 0;")
        instr = f"@p {instr}"
    if bridge_decls:
        # One block: predicate declarations, the setp conversions in, the
        # instruction, the selp conversions out.
        asm_text = "{ " + " ".join([*bridge_decls, *asm_pre, instr, *asm_post]) + " }"
    else:
        asm_text = instr
    # Two independent C-level properties, deliberately not conflated:
    #
    #   asm volatile  - may nvcc reorder or drop the emitted asm?
    #   "memory"      - does the instruction touch memory?
    #
    # `volatile` is stated per entry so each instruction keeps the barrier its
    # legacy helper had. The clobber is derived instead: an instruction with no
    # memory operand cannot clobber memory, and claiming it does is a needless
    # optimization barrier around register-only instructions like ex2.
    volatile = " volatile" if entry.asm_volatile else ""
    # Two ways to clobber memory: name an address, or order everyone else's
    # accesses (a fence names nothing but must not let loads/stores move past).
    # A tmem address does not count: Tensor Memory is not C-visible memory, so
    # the compiler has nothing to spill/reload around the instruction, and the
    # legacy tcgen05 ld/st/mma helpers never claimed the clobber -- for mma,
    # issued per K-step in the hottest loop, the needless barrier is measurable.
    # Ordering against tmem consumers is the job of tcgen05.fence/commit/wait.
    touches_memory = (
        any(s.kind == "addr" and operand_space(s, mod_map) != "tmem" for s in entry.operands)
        or entry.orders_memory
    )
    clobber = ' : "memory"' if touches_memory else ""
    asm_line = f'asm{volatile}("{asm_text}" : {", ".join(outputs)} : {", ".join(inputs)}{clobber});'
    body = "\n".join(f"  {line}" for line in [*pre, asm_line, *post])
    source = f"__forceinline__ __device__ void {helper}({', '.join(params)}) {{\n{body}\n}}\n"
    return opcode, helper, source
