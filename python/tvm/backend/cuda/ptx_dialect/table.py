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
"""Instruction table for the ``T.ptxd`` table-driven PTX dialect prototype.

Pure data + pure functions: this module deliberately imports nothing from
``tvm`` so the thin generators (``gen_stubs``, ``gen_coverage``,
``gen_helpers``) can load it standalone.

The converged :class:`InstructionEntry` design:

- ``name`` is a single identifier-safe token. Multi-token PTX mnemonics
  (``cvta.to.shared``, ``prefetch.global.L2``) are expressed as a family
  plus single-choice modifier slots — PTX itself treats those dots as
  modifiers, and it keeps the namespace machinery trivial.
- ``slots`` declares each modifier position's domain (name, tokens,
  optional). This drives attribute-chain resolution, stub generation, and
  variant enumeration.
- ``check`` is the single cross-slot constraint mechanism: a plain, pure
  Python function ``mod_map -> error-string | None`` (mod_map maps every
  slot name to its token, ``""`` = omitted). It runs at trace time to
  reject illegal combinations with a readable message, and as a filter in
  :func:`variants` — so exhaustive nvcc gating still works. Give it a
  one-line docstring; the generators surface it as documentation.
- One entry = one *syntax shape*: fixed operand list and result structure.
  Variants that change either (e.g. vector destinations) become separate
  entries, each declaring only the slots it uses.
- Predication (``@p``) is framework-level: every instruction without a
  destination accepts ``pred=`` at the call site; entries never mention it.

Calling convention: PTX has no defining form — a register is declared first
and instructions then write into it — so a ptxd call mirrors the PTX text
exactly. Destinations are ordinary operands (``role="dst"``, in PTX operand
order), every helper is ``void``, and every call is a statement::

    acc: T.float32                        # .reg .f32 acc;
    T.ptxd.add.rn.f32(acc, x, acc)        # add.rn.f32 acc, x, acc;
"""

import functools
import itertools
import keyword
import re
from collections.abc import Callable
from dataclasses import dataclass

# PTX operand type token -> the TVM dtypes it accepts, canonical first.
#
# A *bit-size* type names a width, not an interpretation -- ISA 5.2: "The
# bit-size type is compatible with any fundamental type having the same size."
# So `.bN` accepts every TVM dtype of that width and each gets its own helper
# (see `operand_dtypes`). The canonical one is listed first: a variant using
# only canonical dtypes keeps the helper name it had before the axis existed.
#
# A concretely typed operand (.u32/.s32/.f32/...) names an interpretation, so
# it accepts that one dtype -- substituting another would be a semantic change
# rather than a relabelling.
PTX_TYPE_DTYPES = {
    "b8": ("uint8", "int8"),
    "b16": ("uint16", "int16", "float16", "bfloat16"),
    "b32": ("uint32", "int32", "float32"),
    "b64": ("uint64", "int64", "float64"),
    "b128": ("uint128", "int128"),
    "u8": ("uint8",),
    "s8": ("int8",),
    "u16": ("uint16",),
    "s16": ("int16",),
    "u32": ("uint32",),
    "s32": ("int32",),
    "u64": ("uint64",),
    "s64": ("int64",),
    "f32": ("float32",),
    "f64": ("float64",),
    # `.f32x2` operands "have .b64 type" (ISA 9.7.3.3): a pair of packed floats
    # in one 64-bit register. It names a packed layout whose container happens
    # to be .b64, not a bit container, so it is not widened.
    "f32x2": ("uint64",),
    # cvt's narrow packed formats, per ISA 9.7.9.22's operand table. Each names
    # a lane layout, not a container: two (or four) sub-byte or 8-bit elements
    # ride one integer register, exactly as `.f16x2` does. The carrier width is
    # what the register must be, so it is the dtype the operand binds.
    "e4m3x2": ("uint16",),
    "e5m2x2": ("uint16",),
    "e2m3x2": ("uint16",),
    "e3m2x2": ("uint16",),
    "ue8m0x2": ("uint16",),
    "s2f6x2": ("uint16",),
    "e2m1x4": ("uint16",),
    "e4m3x4": ("uint32",),
    "e5m2x4": ("uint32",),
    "e2m3x4": ("uint32",),
    "e3m2x4": ("uint32",),
    "tf32": ("uint32",),
    # `.e2m1x2` is the one that does not fit: the ISA types it .b8, and ptxas
    # rejects a wider register in that position, so it can only be reached
    # through a raw entry that stages it in a block-local `.reg .b8`. The token
    # is here so those entries can name it; the uint8 is the C-boundary
    # contract, not the register the instruction sees.
    "e2m1x2": ("uint8",),
    # Mixed-precision sources live in a plain 16-bit register -- the ISA's own
    # example declares them `.reg .b16`.
    "f16": ("uint16",),
    "bf16": ("uint16",),
    # Packed SIMD types, same reasoning as `.f32x2`: the token names a layout,
    # its container is one ordinary register of the matching width.
    "u16x2": ("uint32",),
    "s16x2": ("uint32",),
    "u8x4": ("uint32",),
    "s8x4": ("uint32",),
    "f16x2": ("uint32",),
    "bf16x2": ("uint32",),
}


def escape_token(token: str) -> str:
    """PTX token -> Python attribute name (``::`` -> ``__``, keyword -> trailing ``_``)."""
    token = token.replace("::", "__")
    if keyword.iskeyword(token):
        token += "_"
    return token


def unescape_token(token: str) -> str:
    """Python attribute name -> PTX token. Inverse of :func:`escape_token`."""
    token = token.replace("__", "::")
    if token.endswith("_"):
        token = token[:-1]
    return token


@dataclass(frozen=True)
class ModifierSlot:
    """One modifier position of an instruction family, in asm render order."""

    name: str
    choices: tuple[str, ...]
    optional: bool = False  # optional => omitted token is simply not rendered


LanesFn = Callable[[dict], int]  # modifier map -> registers in this operand's group


@dataclass(frozen=True)
class OperandSlot:
    """One operand of an instruction family, in PTX operand order.

    role:
      - ``"addr"``  memory operand, rendered ``[%k]``; state space comes from
        ``space`` (fixed per operand, for instructions whose operands live in
        different spaces like cp.async.bulk) or else the entry's ``space``
        modifier slot. Shared-space addresses are auto-coerced (generic
        pointer -> cvta) or accepted as raw ``uint32``.
      - ``"ptr"``   raw pointer value, rendered ``%k`` (e.g. ``cvta`` input).
      - ``"value"`` register operand typed by ``dtype`` (fixed per operand)
        or else the entry's ``type`` modifier slot.
      - ``"imm"``   an operand that lives in the instruction *text*, not in a
        register. With ``literal`` the ISA fixes its value (st.bulk's initval
        "must be zero"): no C parameter, no call argument. With ``choices``
        the caller picks the value -- a compile-time constant, validated
        against the closed set at trace time and baked into the text, with
        one helper generated per value (the way `cp.async.wait_group N` or
        `setmaxnreg`'s nreg exist only as integer literals in the ISA).
      - ``"dst"``   a destination the instruction writes, typed like
        ``"value"``. The helper takes it as a C++ reference and the caller
        passes a writable lvalue (a scalar or a buffer element), mirroring
        PTX's own "declare the register, then name it as an operand".
      - ``"acc"``   a register the instruction reads AND writes -- an in-place
        accumulator, bound "+" where a dst binds "=". Takes an lvalue like a
        dst; unlike a dst it does not block ``@p``, because "+" keeps the old
        value live under a false predicate.
      - ``"pred_dst"`` a ``.pred`` result. Inline asm has no constraint letter
        for predicate registers, so the boundary conversion lives inside the
        block: the instruction writes a predicate register and a trailing
        ``selp.b32`` materializes it as 0/1 into a "=r" uint32 the caller
        receives through a reference parameter, exactly like a dst (and it
        gates ``@p`` like one).
      - ``"pred_src"`` a ``.pred`` argument. The mirror conversion: a leading
        ``setp.ne.b32`` turns a "r"-bound uint32 into the predicate register
        the instruction reads. These are the block's second and third
        boundary-conversion exceptions -- ``@p``'s own setp was the first.

    ``lanes`` > 1 makes the operand a brace-enclosed register group. PTX writes
    the group in the operand list (``mov.b64 d, {lo, hi}``), so the group is
    part of the *shape*, never of the dotted modifier text.
    """

    name: str
    role: str
    space: str | None = None
    dtype: str | None = None
    # role="imm" is a value in the instruction *text* (never a C parameter),
    # in one of three states, by who owns the value:
    #   literal set   -- the ISA fixed it; invisible to programs.
    #   choices set   -- the caller picks from this closed set; every value is
    #                    a certified variant.
    #   neither set   -- OPEN: the caller passes any compile-time integer
    #                    constant; the table declares no domain because the ISA
    #                    declares none (tcgen05.ld/st's immHalfSplitoff).
    #                    Certification can only sample an open domain -- see
    #                    imm_combos.
    literal: str | None = None
    choices: tuple[str, ...] | None = None
    # A PTX register group: `{%k, %k+1, ...}`. The operand takes `lanes` call
    # arguments and renders as one brace-enclosed vector expression. 1 = a plain
    # scalar operand, which is every operand of every other family.
    #
    # A callable makes the group length a function of the modifier map -- the
    # ISA states these lengths as formulas over the modifiers ("r is a register
    # vector of length shape * num / 32b"), and the modifiers are a closed set,
    # so every resulting length is still enumerable and certifiable. Same
    # contract as `check`: total over every combination `variants()` generates.
    lanes: int | LanesFn = 1
    # Whether PTX writes this operand as a brace-enclosed vector. Defaults to
    # "it has more than one register", which a length function cannot answer
    # ahead of time -- an operand whose length varies between 0 and 1 is a
    # bracketed-optional scalar, not a vector.
    vector: bool | None = None
    # A composite memory operand: adjacent slots naming the same bracket render
    # inside one pair of square brackets, each keeping its own C parameters and
    # constraints. This is how TMA spells its address -- `[tensorMap,
    # tensorCoords]` is a single PTX operand holding a 64-bit pointer and an
    # .s32 coordinate vector that have no single-register encoding.
    bracket: str | None = None


CheckFn = Callable[[dict], str | None]


@dataclass(frozen=True)
class InstructionEntry:
    """One PTX instruction family (one syntax shape).
    ``orders_memory`` marks an instruction that constrains the order of *other*
    threads' memory accesses without naming any address itself -- fences,
    barriers, async-group waits. The ``"memory"`` clobber is otherwise derived
    from "does this entry have an ``addr`` operand", which is the right rule for
    instructions that touch memory but cannot see a fence: ``asm volatile``
    alone only pins the asm block, it does not stop the compiler moving ordinary
    loads and stores across it.
    """

    name: str  # table key, e.g. "cvta", "st_bulk"; the surface name is `family`
    operands: tuple[OperandSlot, ...]
    slots: tuple[ModifierSlot, ...] = ()
    check: CheckFn | None = None  # cross-slot validation, mod_map -> error | None
    # Whether the emitted inline asm carries `volatile`. This is purely a
    # C-level optimization barrier — it never changes *which* PTX instruction
    # is emitted, only whether nvcc may common up or drop identical calls.
    # Every op registers as kOpaque (a void call has to survive RemoveNoOp),
    # so the IR shares nothing either way; this flag exists only to preserve
    # each instruction's established barrier byte-for-byte across migration.
    asm_volatile: bool = True
    orders_memory: bool = False
    # The PTX mnemonic, when it is not spellable as a Python identifier.
    # `st.bulk` is one instruction whose name contains a dot, so the table key
    # (st_bulk) and the emitted mnemonic (st.bulk) have to differ.
    mnemonic: str | None = None
    # The -arch every variant of this family must be certified at: the maximum
    # floor over its variants, not the minimum. Certifying below a variant's
    # floor makes ptxas report legal forms as illegal, and those verdicts would
    # then get baked into a check() and silently delete coverage.
    cert_arch: str | None = None
    # An escape hatch, not a mechanism: a family whose helper body this table
    # cannot derive writes it out by hand.
    # `raw_render(entry, opcode, helper, tokens, dtypes)` returns the helper
    # source. It is handed the opcode and helper name `render_variant` has
    # already derived, rather than deriving them again -- a second formula for
    # the helper name is a call to a function that does not exist, and no
    # assertion can be relied on to catch it. Everything else (variant
    # enumeration, certification, dispatch, stubs) is unchanged, so a raw entry
    # is still enumerable and still proven against ptxas.
    #
    # It exists for one shape: an operand the ISA types as .b8. Inline asm has
    # no 8-bit register constraint, ptxas rejects a wider carrier in those
    # positions, and the value therefore has to be staged through a block-local
    # `.reg .b8` -- several statements, which the single-instruction invariant
    # otherwise forbids. Every raw entry must be listed in that test's
    # exemption set, so adding one is never silent.
    raw_render: Callable[["InstructionEntry", str, str, tuple, tuple], str] | None = None

    @property
    def op_name(self) -> str:
        return f"tirx.ptxd.{self.name}"

    @property
    def ptx_name(self) -> str:
        return self.mnemonic or self.name

    @property
    def family(self) -> str:
        """The attribute users type: ``T.ptxd.<family>``.

        The mnemonic with dots folded to underscores. Equal to ``name`` for
        every single-shape family (``st.bulk`` -> ``st_bulk``); the shared
        surface name where several entries differ only in operand shape, as
        all the ``mov_*`` entries do.
        """
        return self.ptx_name.replace(".", "_")

    @functools.cached_property
    def typed_operands(self) -> tuple[OperandSlot, ...]:
        """The operands carrying a dtype, in order: a dtype tuple aligns with these."""
        return tuple(s for s in self.operands if s.role in ("value", "dst", "acc"))

    @property
    def has_dst(self) -> bool:
        """Whether the instruction writes a destination operand.

        Gates ``@p``: a false predicate leaves destinations unwritten, and the
        ``"="`` output constraint tells nvcc the prior value is dead, so a
        predicated destination silently loses it. An accumulator (``role="acc"``)
        binds "+" instead, which keeps the old value live -- so it does not
        count here and @p remains available on it. A pred_dst is written
        through "=" the same way a dst is, so it counts.
        """
        return any(slot.role in ("dst", "pred_dst") for slot in self.operands)


def mods(entry: InstructionEntry, tokens) -> dict:
    """The canonical modifier map: every slot name -> token, ``""`` = omitted."""
    return {slot.name: tok or "" for slot, tok in zip(entry.slots, tokens)}


def lanes_of(slot: OperandSlot, mod_map: dict) -> int:
    """How many registers this operand's group occupies under these modifiers."""
    return slot.lanes(mod_map) if callable(slot.lanes) else slot.lanes


def operand_layout(entry, mod_map: dict) -> tuple[tuple[OperandSlot, int, int], ...]:
    """``(slot, first_arg_index, lanes)`` per operand the caller supplies.

    One row per *operand*, not per argument: a register group is one operand
    occupying N registers, so its dtype and its coercion are decided once for
    the whole group. Depends on the modifiers because a group's length may --
    the modifier set is closed, so there is one layout per token combination,
    memoized below.
    """
    return _operand_layout(entry, tuple(mod_map.values()))


@functools.cache
def _operand_layout(entry, mod_values):
    mod_map = dict(zip((s.name for s in entry.slots), mod_values))
    rows, i = [], 0
    for slot in entry.operands:
        if slot.role == "imm" and slot.literal is not None:
            # Table-owned value: no call argument. choices/open imms DO occupy
            # a call-argument position (the engine reads and bakes them).
            continue
        n = lanes_of(slot, mod_map)
        rows.append((slot, i, n))
        i += n
    return tuple(rows)


def tokens_for(entry: InstructionEntry, **by_name) -> tuple[str, ...]:
    """Modifier tokens in slot order, named instead of positional.

    The inverse of :func:`mods`. A positional tuple silently shifts every token
    when a slot is inserted -- the exact edit this table invites -- so anything
    naming a specific variant (tests, tools) should name its slots.
    """
    unknown = set(by_name) - {slot.name for slot in entry.slots}
    if unknown:
        raise ValueError(f"{entry.name}: no modifier slot named {sorted(unknown)}")
    out = []
    for slot in entry.slots:
        token = by_name.get(slot.name, "")
        if token and token not in slot.choices:
            raise ValueError(f"{entry.name}.{slot.name}: {token!r} not in {slot.choices}")
        if not token and not slot.optional:
            raise ValueError(f"{entry.name}.{slot.name} is required")
        out.append(token)
    out = tuple(out)
    # Slot-level legality is not the whole rule: `check` rejects combinations
    # whose tokens are individually fine (ld.volatile takes no scope). Without
    # this a golden test could pin a variant the dialect refuses to emit.
    if entry.check is not None:
        error = entry.check(mods(entry, out))
        if error:
            raise ValueError(f"{entry.name}: {error}")
    return out


def operand_type(slot: OperandSlot, mod_map: dict) -> str:
    """The PTX type token of one operand.

    ``OperandSlot.dtype`` either names a modifier slot or is a literal type
    token; ``None`` means the entry's ``type`` slot. Naming an *optional* slot
    that was omitted falls back to ``type``, which is how the mixed-precision
    forms are typed: ``add.rn.f32.bf16``'s ``a`` is the ``.bf16`` source, while
    plain ``add.rn.f32``'s ``a`` is just ``.f32``.
    """
    key = slot.dtype or "type"
    if key in mod_map:
        return mod_map[key] or mod_map["type"]
    return key


def operand_space(slot: OperandSlot, mod_map: dict) -> str:
    """The state space of one ``addr`` operand.

    Fixed per operand for instructions whose operands live in different spaces
    (cp.async.bulk), else the entry's ``space`` modifier slot. This is the one
    definition: an address operand's C carrier (32-bit shared window vs generic
    pointer) hangs off it, so a family whose space slot is named anything else
    silently renders every shared form as a generic pointer.
    """
    return slot.space or mod_map.get("space", "")


def operand_dtypes(slot: OperandSlot, mod_map: dict) -> tuple[str, ...]:
    """The TVM dtypes one operand accepts, canonical first (see PTX_TYPE_DTYPES)."""
    return PTX_TYPE_DTYPES[operand_type(slot, mod_map)]


def canonical_dtypes(entry: InstructionEntry, tokens) -> tuple[str, ...]:
    """The canonical dtype of each typed operand, in operand order.

    This is `dtype_combos(...)[0]`, but as the per-operand fact it actually is:
    "canonical" belongs to an operand, not to the product of all of them.
    """
    mod_map = mods(entry, tokens)
    return tuple(operand_dtypes(s, mod_map)[0] for s in entry.typed_operands)


def dtype_combos(entry: InstructionEntry, tokens) -> tuple[tuple[str, ...], ...]:
    """Every dtype assignment for one modifier combination, canonical first.

    One dtype per *operand*, never per lane: a register group is one operand
    that occupies N registers, and ISA 6.4.3 calls a brace list "similarly typed
    scalars". Operands multiply, so an instruction with two bit-typed operands
    has the product of their choices.
    """
    mod_map = mods(entry, tokens)
    axes = [operand_dtypes(s, mod_map) for s in entry.typed_operands]
    return tuple(itertools.product(*axes)) if axes else ((),)


def renderings(entry: InstructionEntry):
    """Every ``(tokens, dtypes, predicated)`` this entry renders to.

    The product of the four independent axes -- modifiers, operand dtypes,
    caller immediates, predication. Defined once so a new axis lands in one
    place instead of in every consumer (the certification tiers, the helper
    dump, the stub writer).
    """
    for tokens in variants(entry):
        for dtypes in dtype_combos(entry, tokens):
            for imms in imm_combos(entry):
                for predicated in pred_forms(entry):
                    yield tokens, dtypes, predicated, imms


def imm_slots(entry: InstructionEntry) -> tuple[OperandSlot, ...]:
    """The caller-passed immediates (choices or open), in operand order; an imm
    tuple aligns with these. Literal imms are table-owned and not in it."""
    return tuple(s for s in entry.operands if s.role == "imm" and s.literal is None)


def imm_combos(
    entry: InstructionEntry, open_samples: tuple[str, ...] = ("0",)
) -> tuple[tuple[str, ...], ...]:
    """Every caller-immediate assignment the enumeration walks.

    A `choices` slot enumerates its whole closed set -- each value is a
    variant, all pre-certified. An OPEN slot has no domain to enumerate, so it
    enumerates at ``open_samples`` instead. The samples belong to the
    enumerating facility (certification, snapshots, stubs), not to the
    instruction: at a call site the caller passes whatever constant it wants,
    and certification has only proven the *shape* assembles, at these samples.
    """
    axes = [s.choices if s.choices is not None else open_samples for s in imm_slots(entry)]
    return tuple(itertools.product(*axes)) if axes else ((),)


def pred_forms(entry: InstructionEntry) -> tuple[bool, ...]:
    """Which ``predicated`` renderings exist: both, unless the entry has a destination."""
    return (False,) if entry.has_dst else (False, True)


@functools.cache
def variants(entry: InstructionEntry) -> tuple:
    """Every legal modifier combination: the slot product filtered by ``check``.

    Cached: for wide entries like ld the raw product is in the millions.
    """
    axes = [(*slot.choices, "") if slot.optional else slot.choices for slot in entry.slots]
    return tuple(
        combo
        for combo in itertools.product(*axes)
        if entry.check is None or not entry.check(mods(entry, combo))
    )


def _check_ld(m):
    """Scalar ld grammar per PTX ISA 9.7.9.8 (ld) and 9.7.9.9 (ld.global.nc)."""
    sem, scope, ss = m["sem"], m["scope"], m["space"]
    mmio, cop, nc = m.get("mmio", ""), m["cop"], m["nc"]
    prefetch = m["prefetch"]
    # One eviction qualifier for grammar purposes. The ISA never spells the L1
    # and L2 priorities apart -- every line carrying either carries both, in
    # that order -- so every exclusion below covers both, and the 256-bit
    # entries need not restate any of them. `.mmio` and `.l2ev` are read with
    # a default because the vector entries declare no `mmio` slot and only the
    # 256-bit ones declare `l2ev`.
    l1ev = m["l1ev"] or m.get("l2ev", "")
    # "ld.relaxed.scope / ld.acquire.scope" — scope is mandatory there and
    # invalid on the weak/volatile forms (syntax lines).
    if sem in ("relaxed", "acquire") and not scope:
        return f"ld.{sem} requires a scope (cta/cluster/gpu/sys)"
    if sem in ("", "weak", "volatile") and scope:
        return "only ld.relaxed/ld.acquire take a scope"
    if mmio:
        # "ld.mmio.sem.sys{.global}": "Only .sys thread scope is valid";
        # global or generic addressing only. The ISA also allows .acquire
        # (PTX ISA 9.3+), but the current toolchain assembles 9.2 and ptxas
        # rejects it — widen when the toolchain catches up.
        if sem != "relaxed":
            return "ld.mmio requires .relaxed"
        if scope != "sys":
            return "only the sys scope is valid for ld.mmio"
        if ss not in ("", "global"):
            return "ld.mmio may only be used with .global or generic addressing"
        if cop or nc or l1ev or prefetch:
            return "ld.mmio takes no cache qualifiers"
    if sem in ("relaxed", "acquire") and not mmio:
        # "May be used with .global, .shared spaces, or generic addressing.
        # Cache operations are not allowed."
        if ss == "local":
            return f"ld.{sem} is not valid on .local"
        if cop:
            return f"cache operations are not allowed with ld.{sem}"
        if nc:
            return "ld.global.nc has no memory-synchronization forms"
    if sem == "volatile" and (cop or l1ev or nc):
        # "ld.volatile{.ss}{.level::prefetch_size}" — prefetch is its only
        # cache qualifier; allowed spaces global/shared/local/generic.
        return "ld.volatile only takes the prefetch_size cache qualifier"
    if cop and l1ev:
        # cop and eviction_priority appear on separate syntax lines.
        return "cache operators and eviction priorities are mutually exclusive"
    if nc:
        # "ld.global{.cop}.nc": .global only, no sem/scope, cop in {ca,cg,cs}.
        if ss != "global":
            return "ld.global.nc requires the .global state space"
        if sem:
            return "ld.global.nc has no sem qualifier"
        if cop in ("lu", "cv"):
            return "ld.global.nc cache operators are limited to ca/cg/cs"
    if prefetch and ss not in ("", "global"):
        # "may only be used with .global state space and generic addressing"
        return "prefetch_size may only be used with .global or generic addressing"
    if l1ev and ss not in ("", "global"):
        # ptxas: "Modifier '.evict_*' cannot be applied to '<ss>' space" —
        # implicit in the ISA prose, enforced by the assembler.
        return "eviction priorities apply only to .global or generic addressing"
    return None


# The vector entries declare no `mmio` slot -- it is the one qualifier with no
# `{.vec}` position, since PTX ISA 9.7.9.8 spells it
# `ld.mmio.sem.sys{.global}.type  d, [a];` and 9.7.9.11 spells it
# `st.mmio.sem.sys{.global}.type         [a], b;`, neither carrying a `{.vec}`.
# `.l2ev` is declared only by the 256-bit entries. Both are read with a default
# in the scalar checks, so a vector modifier map goes straight in.
#
# `.scope` needs no such treatment: every vector entry declares it, because the
# `.relaxed`/`.acquire`/`.release` syntax lines do carry `{.vec}` and spell
# `.scope` as mandatory on it.


def _check_ld_vec(m):
    return _check_vec128(m) or _check_ld(m)


def _check_st_vec(m):
    return _check_vec128(m) or _check_st(m)


def _check_ld_vec256(m):
    """The 256-bit ld lines -- the only ld entry with a `.level2::eviction_priority`.

    PTX ISA 9.7.9.8 spells the L2 priority only where the L1 priority already
    is, and on no line that carries `.cop` or `.volatile`. The three lines that
    settle it, wrapped here but otherwise verbatim:

        ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type
            d, [a]{.unified}{, cache_policy};

        ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}
           {.level::cache_hint}{.level::prefetch_size}{.vec}.type
            d, [a]{.unified}{, cache_policy};

        ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];

    with "The .weak, .volatile, .relaxed and .acquire qualifiers are mutually
    exclusive", so the third line is the only one `.volatile` can be on.

    9.7.9.9 splits `ld.global.nc` the same way -- `ld.global{.cop}.nc{...}`
    against `ld.global.nc{.level1::eviction_priority}{.level2::eviction_priority}`
    -- so `.nc` with an L2 priority is on a syntax line while `.nc` with a
    `.cop` and one is not. The two priorities are grammatically joined: every
    exclusion _check_ld applies to `l1ev` applies to `l2ev` too. That is not
    restated here -- `_check_ld` reads the two as one eviction qualifier, so
    the rule holds by construction and a future L1 rule covers L2 for free.

    ptxas is no authority here. It rejects the L1 spellings ("Modifier
    '.evict_first' cannot be combined with modifier '.cg'", and the same with
    '.volatile') but silently assembles the identical L2 ones.
    """
    return _check_vec256(m) or _check_ld(m)


def _check_st_vec256(m):
    """The 256-bit st lines -- the only st entry with a `.level2::eviction_priority`.

    Same structure as `_check_ld_vec256`, from PTX ISA 9.7.9.11: the L2
    priority shares its lines with the L1 one and appears on neither the `.cop`
    line nor the `.volatile` line, which spells no cache qualifier at all.

        st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type   [a], b{, cache_policy};

        st{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}
           {.level::cache_hint}{.vec}.type                    [a], b{, cache_policy};

        st.volatile{.ss}{.vec}.type                           [a], b;

    The third line is the whole argument for the volatile case: it spells no
    eviction position at all, and "The .weak, .volatile, .relaxed and .release
    qualifiers are mutually exclusive" leaves `.volatile` no other line. Do not
    argue it from the prose ".volatile: ... Cache operations are not allowed."
    instead -- the ISA says exactly the same of `.relaxed` and `.release`, whose
    lines *do* carry both eviction priorities, so "cache operations" there means
    `.cop`. So `l2ev` falls under the existing "st.volatile takes no cache
    qualifiers" rule, and under the cop/eviction exclusion, exactly as `l1ev`
    does -- `_check_st` reads the two as one eviction qualifier, so neither
    rule is restated here. ptxas accepts both L2 pairings while rejecting their
    L1 twins, so it cannot be used to justify keeping them.
    """
    return _check_vec256(m) or _check_st(m)


def _check_st(m):
    """Scalar st grammar per PTX ISA 9.7.9.11 (the mirror of _check_ld)."""
    sem, scope, ss = m["sem"], m["scope"], m["space"]
    mmio, cop = m.get("mmio", ""), m["cop"]
    # One eviction qualifier: see the note in `_check_ld`.
    l1ev = m["l1ev"] or m.get("l2ev", "")
    # "st.relaxed.scope / st.release.scope" -- scope is mandatory there and
    # invalid on the weak/volatile forms (syntax lines).
    if sem in ("relaxed", "release") and not scope:
        return f"st.{sem} requires a scope (cta/cluster/gpu/sys)"
    if sem in ("", "weak", "volatile") and scope:
        return "only st.relaxed/st.release take a scope"
    if mmio:
        # "st.mmio.sem.sys{.global}": "Only .sys thread scope is valid for the
        # st.mmio operation." .release with .mmio arrives in PTX ISA 9.3; the
        # toolchain here assembles 9.2 and ptxas rejects it, so keep .relaxed
        # only and widen when the toolchain catches up.
        if sem != "relaxed":
            return "st.mmio requires .relaxed"
        if scope != "sys":
            return "only the sys scope is valid for st.mmio"
        if ss not in ("", "global"):
            return "st.mmio may only be used with .global or generic addressing"
        if cop or l1ev:
            return "st.mmio takes no cache qualifiers"
    if sem in ("relaxed", "release") and not mmio:
        # ".relaxed and .release: May be used with .global, .shared spaces or
        # with generic addressing... Cache operations are not allowed."
        if ss == "local":
            return f"st.{sem} is not valid on .local"
        if cop:
            return f"cache operations are not allowed with st.{sem}"
    if sem == "volatile" and (cop or l1ev):
        # "st.volatile{.ss}{.vec}.type" -- no cache qualifiers at all.
        return "st.volatile takes no cache qualifiers"
    if cop and l1ev:
        # cop and eviction_priority appear on separate syntax lines.
        return "cache operators and eviction priorities are mutually exclusive"
    if l1ev and ss not in ("", "global"):
        # ptxas: "Modifier '.evict_*' cannot be applied to '<ss>' space"
        return "eviction priorities apply only to .global or generic addressing"
    return None


def _check_rcp(m):
    """This entry's rcp.approx is f32-only; .f64 is IEEE-rounded, no .ftz (PTX ISA 9.7.3.13)."""
    # Syntax lines: rcp.approx{.ftz}.f32 / rcp.rnd{.ftz}.f32 / rcp.rnd.f64
    if m["mode"] == "approx" and m["type"] != "f32":
        return (
            "rcp.approx.f64 is a separate syntax line (PTX ISA 9.7.3.14, where .ftz is "
            "mandatory) and is not registered here"
        )
    if m["type"] == "f64":
        if m["mode"] == "approx":
            return "rcp.f64 requires an IEEE rounding mode (.rn/.rz/.rm/.rp)"
        if m["ftz"]:
            return "rcp.rnd.f64 takes no .ftz"
    return None


# The .type tokens of the integer min/max lines (PTX ISA 9.7.1.13/9.7.1.14).
# `.relu` is only on the signed line (type2); the unsigned/other line takes none.
# NOT REGISTERED: `.u8x4` and `{.relu}.s8x4`, which the ISA supports only on
# "sm_120f or higher in the same family" -- outside the architectures this
# dialect certifies, and ptxas rejects them at sm_100.
_MINMAX_INT_PLAIN = ("u16", "u32", "u64", "u16x2", "s16", "s64")
_MINMAX_INT_RELU = ("s16x2", "s32")
# The floating-point and half-precision lines (9.7.3.11/9.7.3.12, 9.7.4.7/9.7.4.8).
_MINMAX_FP = ("f32", "f64", "f16", "f16x2", "bf16", "bf16x2")


def _check_mbarrier_sem_scope(m):
    """`.sem` and `.scope` are one qualifier pair: both or neither.

    Stated in so many words by every mbarrier section that has the pair --
    "Qualifiers .sem and .scope must be specified together." (ISA 9.7.14.16.14
    expect_tx, .15 complete_tx, .16 arrive, .17 arrive_drop, .19 test_wait /
    try_wait). The rule is what the sections say, not how they spell it: some
    lines write the pair as one `{.sem.scope}` group and others as two
    `{.sem}{.scope}` groups, and `mbarrier.arrive.noComplete` fixes it to
    `{.release.cta}` outright.
    """
    if bool(m.get("sem", "")) != bool(m.get("scope", "")):
        return ".sem and .scope go together: write both or neither"
    return None


def _check_fence_proxy(m):
    """`.proxykind = { .alias, .async, .async.global, .async.shared::{cta,cluster} }`.

    Modelled as proxykind + an optional space so the surface can walk it one
    attribute at a time; only `.async` carries a state space (ISA 9.7.14.4).
    """
    if m["space"] and m["proxykind"] != "async":
        return f"fence.proxy.{m['proxykind']} takes no state space"
    return None


def _check_minmax(m):
    """Which qualifiers each min/max syntax line allows.

    Integer lines (9.7.1.13/14):  op.type1 d,a,b   |  op{.relu}.type2 d,a,b
    Floating lines (9.7.3.11/12): op{.ftz}{.NaN}{.xorsign.abs}.f32 d,a,b
                                  op.f64 d,a,b
    Half lines (9.7.4.7/8):       op{.ftz}{.NaN}{.xorsign.abs}.f16{x2} d,a,b
                                  op{.NaN}{.xorsign.abs}.bf16{x2} d,a,b
    `.xorsign` and `.abs` are one paired qualifier `{.xorsign.abs}` -- two slots
    only because the surface reaches them one attribute at a time.
    """
    ty = m["type"]
    ftz, nan, xorsign, abs_, relu = (
        m.get("ftz", ""),
        m.get("nan", ""),
        m.get("xorsign", ""),
        m.get("abs", ""),
        m.get("relu", ""),
    )
    is_int = ty in _MINMAX_INT_PLAIN or ty in _MINMAX_INT_RELU
    if is_int:
        if ftz or nan or xorsign or abs_:
            return f".{ty} is an integer line and takes no .ftz/.NaN/.xorsign.abs"
        if relu and ty not in _MINMAX_INT_RELU:
            return f".relu is only on {'/'.join(_MINMAX_INT_RELU)}, not .{ty}"
        return None
    if relu:
        return f".relu is an integer-line qualifier, not valid on .{ty}"
    if ty == "f64" and (ftz or nan or xorsign or abs_):
        return "the .f64 line is the bare form: no .ftz/.NaN/.xorsign.abs"
    if ty.startswith("bf16") and ftz:
        return f".{ty} takes no .ftz"
    if bool(xorsign) != bool(abs_):
        return ".xorsign and .abs are one paired qualifier: write both or neither"
    return None


def _check_minmax3(m):
    """The three-source line: `op{.ftz}{.NaN}{.abs}.f32 d, a, b, c` (min 9.7.3.11, max 9.7.3.12).

    `.abs` here is standalone -- unlike the two-source line, which pairs it with
    `.xorsign`.
    """
    return None


def _check_half_arith(m):
    """The half-precision add/sub/mul lines, ISA 9.7.4.{1,2,3}.

        sub{.rnd}{.ftz}{.sat}.f16   d, a, b;
        sub{.rnd}{.ftz}{.sat}.f16x2 d, a, b;
        sub{.rnd}.bf16   d, a, b;
        sub{.rnd}.bf16x2 d, a, b;

    Two divergences from the single/double lines, both visible above: `.rnd`
    is `{.rn}` alone rather than _FRND, and the bf16 lines carry neither
    `.ftz` nor `.sat`.
    """
    if m["type"].startswith("bf16") and (m["ftz"] or m["sat"]):
        return f".{m['type']} takes no .ftz or .sat"
    return None


def _check_neg(m):
    """`neg{.ftz}.f32 d, a;` and `neg.f64 d, a;` (ISA 9.7.3.10) -- the f64
    line spells no `.ftz`."""
    if m["ftz"] and m["type"] != "f32":
        return ".ftz appears only on the .f32 line"
    return None


def _check_farith(m):
    """Which qualifiers each add/sub/mul/fma syntax line allows (PTX ISA 9.7.3.{3,4,5,6}, 9.7.5).

    Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 | op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64
    Mixed-precision lines: op{.rnd}{.sat}.f32.atype  (.atype = .f16 | .bf16)
    """
    ty, src = m["type"], m.get("srctype", "")
    if src:
        if ty != "f32":
            return f"mixed-precision .{src} source only exists on the .f32 line"
        if m.get("ftz"):
            return "the mixed-precision line takes no .ftz"
        return None
    if m.get("ftz") and ty == "f64":
        return "only the .f32/.f32x2 lines take .ftz"
    if m.get("sat") and ty != "f32":
        return "only the .f32 line takes .sat"
    return None


def _check_prefetch(m):
    """Each prefetch syntax line names exactly one target (PTX ISA 9.7.9.16).

    `.level::eviction_priority` stays bound to `.global` on purpose: its syntax
    line is `prefetch.global.level::eviction_priority`, with `.global` written
    in rather than the `{.ss}` that the `ld` lines carry. Generic addressing is
    not offered there, so neither is it here.
    """
    level, evict, tmap = m["level"], m["evict"], m["tensormap"]
    space = m["space"]
    if sum(bool(x) for x in (level, evict, tmap)) != 1:
        return "exactly one of .level, .level::eviction_priority or .tensormap"
    if tmap:
        # "prefetch{.tensormap_space}.tensormap", .tensormap_space = .const/.param
        if space not in ("", "const", "param"):
            return ".tensormap takes .const or .param (or generic addressing)"
    else:
        # "prefetch{.space}.level", .space = .global/.local
        if space in ("const", "param"):
            return ".const/.param are only valid with .tensormap"
        if evict and space != "global":
            return "eviction priority requires .global"
    return None


_RED_SEM = ("relaxed", "release")
_ATOM_SEM = ("relaxed", "acquire", "release", "acq_rel")
_ATOM_SCOPES = ("cta", "cluster", "gpu", "sys")
_ATOM_SPACES = ("global", "shared", "shared::cta", "shared::cluster")
_ATOM_OPS = ("and", "or", "xor", "add", "inc", "dec", "min", "max")
_ATOM_TYPES = ("b32", "b64", "u32", "u64", "s32", "s64", "f32", "f64")


def _check_atomic(m):
    """op x type pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).

    Normative source: ISA Table 35 (atom) and Table 36 (red), which give the
    pairing cell by cell. The `.type = {...}` line in the Syntax block is only
    the union across ops, which is why it cannot be transcribed directly. Half-precision
    types appear in ptxas' message but are excluded from this entry (they need
    .noftz and a half carrier type).
    """
    op, ty = m["op"], m["type"]
    allowed = {
        "and": ("b32", "b64"),
        "or": ("b32", "b64"),
        "xor": ("b32", "b64"),
        "inc": ("u32",),
        "dec": ("u32",),
        "min": ("u32", "s32", "u64", "s64"),
        "max": ("u32", "s32", "u64", "s64"),
        "add": ("u32", "s32", "u64", "f32", "f64"),
    }[op]
    if ty not in allowed:
        return f".{op} requires {' or '.join('.' + t for t in allowed)}"
    return None


_LD_TYPES = (
    "b8", "u8", "s8", "b16", "u16", "s16", "b32", "u32",
    "s32", "b64", "u64", "s64", "f32", "f64", "b128",
)  # fmt: skip
# ISA 5.4.2 caps a vector register at 128 bits, so .v2.b128 does not exist.
_LD_VEC_TYPES = tuple(t for t in _LD_TYPES if t != "b128")
_L1_EVICT = (
    "L1::evict_normal",
    "L1::evict_unchanged",
    "L1::evict_first",
    "L1::evict_last",
    "L1::no_allocate",
)
# ptxas rejects ".L2::evict_unchanged" on ld/st ("Illegal modifier"), though
# the ISA lists it among the eviction priorities.
_L2_EVICT = ("L2::evict_first", "L2::evict_last", "L2::evict_normal")
_BITS32 = ("b32", "u32", "s32", "f32")
_BITS64 = ("b64", "u64", "s64", "f64")


def _vec_lanes(m):
    # ISA 9.7.9.8/9.7.9.11: the destination/source is a brace-enclosed vector
    # of `.vec` registers.
    return int(m["vec"][1:])


def _check_vec128(m):
    """The .vec lines up to 128 bits wide."""
    if m["vec"] == "v4" and m["type"] in _BITS64:
        # 256 bits wide: a separate entry, with its own sm_100 floor.
        return "a 64-bit .v4 is a 256-bit access -- use the 256-bit entry"
    return None


def _check_vec256(m):
    """The two 256-bit .vec lines, which the ISA spells out individually."""
    vec, ty, ss = m["vec"], m["type"], m["space"]
    if not ((vec == "v8" and ty in _BITS32) or (vec == "v4" and ty in _BITS64)):
        return "the 256-bit lines are .v8 with a 32-bit type or .v4 with a 64-bit type"
    if ss not in ("", "global"):
        # Both lines: "State space is .global or generic addressing where the
        # address points to .global".
        return "a 256-bit access takes .global or generic addressing"
    return None


_FRND = ("rn", "rz", "rm", "rp")  # .rnd on the floating-point arithmetic lines


def _matrix_num_lanes(m):
    # ISA 9.7.15.5.16 (stmatrix): "a brace-enclosed vector expression consisting
    # of 1, 2, or 4 32-bit registers as per the value of .num" -- no shape term,
    # unlike ldmatrix's .m16n16 doubling.
    return int(m["num"][1:])


def _ldmatrix_lanes(m):
    # ISA 9.7.15.5.15: "a brace-enclosed vector expression consisting of 1, 2,
    # or 4 32-bit registers as per the value of .num" -- and, for shape 16x16,
    # "two destination registers r0 and r1 of type .b32 must be specified" per
    # matrix, so .m16n16 doubles the count (ptxas: "Vector of size 2 is
    # expected"). That doubling is also why .m16n16 caps .num at .x2.
    return int(m["num"][1:]) * (2 if m["shape"] == "m16n16" else 1)


def _check_ldmatrix_b8fmt(m):
    # Line 2 (`.m8n16`) spells no `.trans`; line 3 (`.m16n16`) spells it
    # mandatorily. "When .shape is .m16n16, only .x1 and .x2 are valid".
    if m["shape"] == "m8n16" and m["trans"]:
        return "the .m8n16 line takes no .trans"
    if m["shape"] == "m16n16":
        if not m["trans"]:
            return "the .m16n16 decompression line requires .trans"
        if m["num"] == "x4":
            return "shape m16n16 supports only .x1 and .x2"
    return None


def _tcgen05_ldst_lanes(m):
    # ISA 9.7.17.8.3 Table 52 / 9.7.17.8.4 Table 53: the register vector holds
    # `.num` x (shape width / 32b) registers, capped at 128.
    per_num = {"16x64b": 1, "32x32b": 1, "16x128b": 2, "16x256b": 4, "16x32bx2": 1}
    return int(m["num"][1:]) * per_num[m["shape"]]


def _check_tcgen05_ldst(m):
    """The Table 52/53 rows marked NA -- the products that exceed 128 registers."""
    if _tcgen05_ldst_lanes(m) > 128:
        return f"shape {m['shape']} caps .num where the vector would exceed 128 registers"
    return None


def _check_tcgen05_cp(m):
    """The shape <-> multicast pairings ISA 9.7.17.9.2 states, and the fmt pair.

    ".64x128b requires .warpx2::02_13 or .warpx2::01_23" and ".32x128b
    requires .warpx4"; the wider shapes copy to all warps and take no
    multicast. Decompression is stated as a pair of qualifiers, so one
    without the other is not a syntax line.
    """
    shape, multicast = m["shape"], m["multicast"]
    if shape == "64x128b":
        if multicast not in ("warpx2::02_13", "warpx2::01_23"):
            return "shape 64x128b requires multicast warpx2::02_13 or warpx2::01_23"
    elif shape == "32x128b":
        if multicast != "warpx4":
            return "shape 32x128b requires multicast warpx4"
    elif multicast:
        return f"shape {shape} takes no multicast"
    if bool(m["dst_fmt"]) != bool(m["src_fmt"]):
        return "dst_fmt and src_fmt are specified together"
    return None


# mma fragment sizes, per the Matrix Fragments tables of ISA 9.7.15.5.1-13.
# Each is `rows * cols * bits / threads / 32`, the register count a thread holds
# of an MxN (or MxK / KxN) tile -- the ISA states the tables, this states the
# rule they follow. .m8n8k4 with .f16 multiplicands is the one shape a warp
# runs as four independent 8-thread MMAs, so its A/B fragments divide by 8.
_MMA_BITS = {
    "f16": 16, "bf16": 16, "tf32": 32, "f32": 32, "f64": 64, "s32": 32,
    "u8": 8, "s8": 8, "u4": 4, "s4": 4, "b1": 1,
    "e4m3": 8, "e5m2": 8, "e3m2": 6, "e2m3": 6, "e2m1": 4,
}  # fmt: skip


def _mma_shape(m):
    mm, nn, kk = re.match(r"m(\d+)n(\d+)k(\d+)", m["shape"]).groups()
    return int(mm), int(nn), int(kk)


def _mma_regs(dtype, rows, cols, threads, reg_bits=32):
    return max(1, rows * cols * _MMA_BITS[dtype] // threads // reg_bits)


def _mma_threads(m):
    """Threads sharing one tile: 8 on the .f16 .m8n8k4 line, 32 everywhere else.

    ISA 9.7.15.5.14: "A warp executing mma.sync.m8n8k4 instruction computes 4
    matrix multiply and accumulate operations. Rest of the mma.sync operations
    compute a single matrix mutliply and accumulate operation per warp." Four
    operations to a warp is 8 threads each, so a thread's fragment of that line
    is an eighth of the tile rather than a thirty-second -- ptxas agrees
    (d=4, a=2, b=2 for .f16.f16.f16.f16; anything else is "Arguments mismatch").

    The division by 8 belongs to that line alone. The .f64 .m8n8k4 line has its
    own fragment section, ISA 9.7.15.5.2, which opens "A warp executing
    mma.m8n8k4 with .f64 floating point type will compute an MMA operation of
    shape .m8n8k4" -- one operation, the whole warp -- and tabulates A and B as
    "A vector expression containing a single .f64 register" and C/D as "A
    vector expression containing of two .f64 registers", i.e. a = b = 1 and
    c = d = 2. That is the 32-thread division. Applying the 8-thread one to
    .f64 is what produced the retracted "ptxas rejects .m8n8k4" note on
    `mma_f64` below.
    """
    # Named positively, as the ISA scopes it: the .f16 m8n8k4 line. Writing it
    # as "m8n8k4 and not .f64" would be the same thing only by accident of
    # which entries reach m8n8k4 today, and would silently return 8 for any
    # other type that line ever admits -- the wrong-fragment-length mistake
    # this function's own history records.
    return 8 if m["shape"] == "m8n8k4" and m["atype"] == "f16" else 32


def _mma_lanes(which):
    """Registers in one of the four operand groups, as a function of the modifiers."""

    def lanes(m):
        mm, nn, kk = _mma_shape(m)
        threads = _mma_threads(m)
        # f64 fragments live in .f64 registers, everything else in .b32.
        reg_bits = 64 if m["dtype"] == "f64" else 32
        if which == "d":
            return _mma_regs(m["dtype"], mm, nn, threads, reg_bits)
        if which == "c":
            return _mma_regs(m["ctype"], mm, nn, threads, reg_bits)
        if which == "a":
            return _mma_regs(m["atype"], mm, kk, threads, reg_bits)
        return _mma_regs(m["btype"], kk, nn, threads, reg_bits)

    return lanes


def _mma_sp_lanes(which):
    """Like `_mma_lanes`, but A holds half of K -- the structured-sparse half.

    ISA 9.7.15.6: "For an MxNxK sparse mma.sp{::ordered_metadata} operation,
    the MxK matrix A is packed into MxK/2 elements" -- two of every four along
    K, so the A fragment of an MxK sparse line is the dense fragment of MxK/2.
    The other three groups are unchanged.
    """

    def lanes(m):
        mm, nn, kk = _mma_shape(m)
        threads = 32  # every sparse line is m16n8, never the 8-thread m8n8k4
        if which == "d":
            return _mma_regs(m["dtype"], mm, nn, threads)
        if which == "c":
            return _mma_regs(m["ctype"], mm, nn, threads)
        if which == "a":
            return _mma_regs(m["atype"], mm, kk // 2, threads)
        return _mma_regs(m["btype"], kk, nn, threads)

    return lanes


# cvt's generic scalar line takes the ISA's twelve-type dtype/atype list.
_CVT_BASIC = (
    "u8", "u16", "u32", "u64",
    "s8", "s16", "s32", "s64",
    "bf16", "f16", "f32", "f64",
)  # fmt: skip

# (significand bits, exponent bits) -- what "loss of precision" is measured on.
_CVT_FP_FORMAT = {"f16": (11, 5), "bf16": (8, 8), "f32": (24, 8), "f64": (53, 11)}
_CVT_INT_BITS = {"u8": 8, "s8": 8, "u16": 16, "s16": 16, "u32": 32, "s32": 32, "u64": 64, "s64": 64}
_CVT_IRND = ("rni", "rzi", "rmi", "rpi")
_CVT_FRND = ("rn", "rz", "rm", "rp")


def _present_lanes(slot: str) -> LanesFn:
    """A bracketed-optional operand: one register when its qualifier is written.

    The ISA spells several of these `{, operand}` against a `{.qualifier}`, and
    says so in the same words each time -- ISA 9.7.9.22:180-182 for cvt's
    scale-factor is typical: "Operand scale-factor and qualifier
    .scaled::n2::ue8m0 must be used together." `lanes=0` is what makes the
    operand vanish from the helper signature (see render.operand_layout), so
    one factory keeps that contract in one place instead of once per family.
    """
    return lambda m: 1 if m[slot] else 0


# `{, scale-factor}` exists exactly when `.scaled::n2::ue8m0` is written. ISA
# 9.7.9.22:180-182: "Optional qualifier .scaled::n2::ue8m0 specifies that the
# instruction uses packed scale-factor with 2 scale values of ue8m0 type.
# Operand scale-factor and qualifier .scaled::n2::ue8m0 must be used together."
_cvt_scale_lanes = _present_lanes("scaled")


def _cvt_f4x2_raw(entry, opcode: str, helper: str, tokens, dtypes) -> str:
    """The hand-written helper body for one `.e2m1x2` line (Hatch A).

    `render.render_variant` hands over the entry plus the opcode and helper
    name it has already derived, so nothing about this entry's identity is
    restated here: the operands come from `entry.typed_operands`, which is the
    order `canonical_dtypes` reports, and which operands exist comes from
    `lanes_of` -- the same rule the derived path applies, so `{, scale-factor}`
    appears exactly when `.scaled::n2::ue8m0` is written without this function
    knowing that qualifier's name.

    These four lines are the only client of `raw_render`, for the reason that
    field documents: exactly one of their operands is `.b8`, and the value has
    to be staged through a block-local `.reg .b8`. What varies between them is
    only which end that operand sits on, so both shapes live in this one
    builder:

        dst  `{ .reg .b8 raw_d; <cvt> raw_d, %1[, %2]; cvt.u16.u8 %0, raw_d; }`
        src  `{ .reg .b8 raw_a; cvt.u8.u16 raw_a, %1; <cvt> %0, raw_a[, %2]; }`

    Every other piece -- carrier locals, constraint letters, the `(uint8_t)`
    truncation at the C boundary -- is taken from `render.C_BINDING`, the same
    table the derived path reads, so a raw helper differs from a derived one in
    the asm text and nowhere else. The import is deferred because `.render`
    imports this module.
    """
    from .render import C_BINDING

    mod_map = mods(entry, tokens)
    operands = [
        (slot.name, dtype)
        for slot, dtype in zip(entry.typed_operands, dtypes, strict=True)
        if lanes_of(slot, mod_map)
    ]
    (dname, ddtype), srcs = operands[0], operands[1:]
    params, inputs, texts = [], [], []
    for index, (name, dtype) in enumerate(srcs, start=1):
        cb = C_BINDING[dtype]
        params.append(f"{cb.c_type} __{name}")
        inputs.append(f'"{cb.constraint}"({cb.to_carrier.format(f"__{name}")})')
        texts.append(f"%{index}")
    dcb = C_BINDING[ddtype]
    params.insert(0, f"{dcb.c_type}& __{dname}")
    volatile = " volatile" if entry.asm_volatile else ""
    pre, post = [], []
    if dcb.carrier != dcb.c_type:
        # The destination is the `.b8`: the instruction writes the `.reg .b8`,
        # the wider carrier picks it up, and the C boundary truncates back.
        reg = f"__{dname}_reg"
        pre.append(f"{dcb.carrier} {reg};")
        post.append(f"__{dname} = {dcb.from_carrier.format(reg)};")
        output = f'"={dcb.constraint}"({reg})'
        block = [
            f".reg .b8 raw_{dname};",
            f"{opcode} raw_{dname}, {', '.join(texts)};",
            f"cvt.u16.u8 %0, raw_{dname};",
        ]
    else:
        # The `.b8` operand is the source `a`, always first in the list.
        aname = srcs[0][0]
        texts[0] = f"raw_{aname}"
        output = f'"={dcb.constraint}"(__{dname})'
        block = [
            f".reg .b8 raw_{aname};",
            f"cvt.u8.u16 raw_{aname}, %1;",
            f"{opcode} %0, {', '.join(texts)};",
        ]
    asm_text = "{ " + " ".join(block) + " }"
    asm_line = f'asm{volatile}("{asm_text}" : {output} : {", ".join(inputs)});'
    body = "\n".join(f"  {line}" for line in [*pre, asm_line, *post])
    return f"__forceinline__ __device__ void {helper}({', '.join(params)}) {{\n{body}\n}}\n"


def _check_cvt_tf32(m):
    """The two `.tf32` lines, ISA 9.7.9.22:18-19 --

        cvt.rna{.satfinite}.tf32.f32               d, a;
        cvt.frnd2{.satfinite}{.relu}.tf32.f32      d, a;

    One entry: same `d, a` shape, same types, and the only difference is which
    modifiers each spelling admits. `.rna` is written on the line that has no
    `{.relu}`, so the two never meet. (ptxas 13.2 agrees -- it answers
    "Modifier '.relu' cannot be combined with modifier '.rna'" -- but the
    grammar above is the reason this is rejected.)
    """
    if m["rnd"] == "rna" and m["relu"]:
        return "the .rna tf32 line has no .relu"
    return None


def _cvt_loses_precision(d: str, a: str) -> bool:
    """Whether a float->float conversion loses precision (ISA's own wording).

    Either fewer significand bits or fewer exponent bits loses information, so
    .bf16 and .f16 lose in *both* directions despite being the same width.
    """
    d_sig, d_exp = _CVT_FP_FORMAT[d]
    a_sig, a_exp = _CVT_FP_FORMAT[a]
    return d_sig < a_sig or d_exp < a_exp


def _cvt_int_covers(d: str, a: str) -> bool:
    """Whether integer type `d`'s value range is a superset of `a`'s."""
    d_signed, a_signed = d[0] == "s", a[0] == "s"
    if d_signed == a_signed:
        return _CVT_INT_BITS[d] >= _CVT_INT_BITS[a]
    # A signed destination holds every unsigned source only if strictly wider;
    # an unsigned destination never holds a signed source's negatives.
    return _CVT_INT_BITS[d] > _CVT_INT_BITS[a] if d_signed else False


def _check_cvt_same_type(t, rnd):
    """The `.dtype == .atype` sub-grid of the generic scalar line, ISA 9.7.9.22.

    A same-type cvt is a real instruction, not a move: an *integer* rounding
    mode may ride a float-to-float conversion here. The ISA licenses that for
    every *same-size* float-to-float pair, which over this entry's type list is
    six pairs, not four -- `.f16` and `.bf16` are both 16-bit. This toolchain
    assembles only the same-type four; see the third ptxas-only restriction in
    `_check_cvt_scalar`, which is where the cross-type pairs are ruled on. ISA
    9.7.9.22:329-331 (Integer Notes): "Integer rounding is required for
    float-to-integer conversions, and for same-size float-to-float conversions
    where the value is rounded to an integer. Integer rounding is illegal in
    all other instances." And 9.7.9.22:424-426: "A floating-point value may be
    rounded to an integral value using the integer rounding modifiers (see
    Integer Notes). The operands must be of the same size. The result is an
    integral value, stored in floating-point format." The section's Examples
    spell three of these forms outright (:645-646, :660): "cvt.rni.f32.f32 x,y;
    // round to nearest int, result is fp", "cvt.f32.f32 x,y;", and
    "cvt.bf16.bf16.rpi        b1, b2       // convert bf16 to corresponding int
    represented in bf16 format".

    So, for `.dtype == .atype == t`:

    - An integer rounding mode is legal exactly when `t` is a floating-point
      type -- ":425 The result is an integral value, stored in floating-point
      format." On an integer `t` the conversion rounds nothing, and ":330
      Integer rounding is illegal in all other instances."
    - A floating-point rounding mode is never legal: ":392-393
      Floating-point rounding is required for float-to-float conversions that
      result in loss of precision, and for integer-to-float conversions.
      Floating-point rounding is illegal in all other instances." -- and a
      conversion to its own type loses no precision.
    `.ftz` and `.sat` are not ruled on here: their rules do not depend on the
    two types being equal, so the caller's shared tail applies them to this
    sub-grid too. (For an integer `t` that tail rejects `.sat` because
    `_cvt_int_covers(t, t)` holds -- the destination range is the source range
    -- and it rejects `.sat` on .bf16 as a toolchain limit.)

    Together with that tail this admits 53 of the 432 same-type spellings in this entry's slot grid,
    which is exactly the set ptxas assembles (measured over the full grid, nvcc
    13.2 -arch=sm_90; no spelling in either direction differs).
    """
    if rnd in _CVT_FRND:
        return f"{t} from {t} is exact, so a floating-point rounding mode is illegal"
    if rnd and t not in _CVT_FP_FORMAT:
        return "integer rounding rounds nothing on an integer-to-same-integer cvt"
    return None


def _check_cvt_frnd2_scalar(d, a, rnd, ftz, sat):
    """The frnd2 *scalar* lines' sub-grid, ISA 9.7.9.22:10 and :14 --

        cvt.frnd2{.relu}{.satfinite}.f16.f32       d, a;
        cvt.frnd2{.relu}{.satfinite}.bf16.f32      d, a;

    -- read with ":49 .frnd2  = { .rn,  .rz };".

    These two lines have the generic scalar line's `d, a` shape and draw both
    of their types from its dtype/atype list, so they are the same entry; the
    only thing that distinguishes them is modifiers, and `.relu`/`.satfinite`
    are exactly the modifiers the generic line does not have. This function is
    that sub-grid: it runs when either token is written.

    - Types: the two lines convert `.f32` to `.f16` or to `.bf16`, and no
      other line in this entry's type product carries either token. The ISA
      lists the destination types `.satfinite` reaches at :204-209,
      ".satfinite modifier is only supported for conversions involving the
      following types: ... .f16, .bf16, .f16x2, .bf16x2, .tf32, .ue8m0x2 as
      destination types." -- the four that are not `.f16`/`.bf16` name shapes
      that live in other entries.
    - Rounding: mandatory, and `.frnd2` is `{.rn, .rz}`; `.rm`/`.rp` and the
      integer-rounding modes appear on no frnd2 line.
    - Neither line brackets `{.ftz}` or `{.sat}`.

    The section's Examples spell the shape out at :647, "cvt.rn.relu.f16.f32
    b, f;        // result is saturated with .relu saturation mode".
    """
    if (d, a) not in (("f16", "f32"), ("bf16", "f32")):
        return (
            f".relu/.satfinite ride the frnd2 lines, which convert f32 to f16 or bf16, "
            f"not {a} to {d}"
        )
    if rnd not in ("rn", "rz"):
        return "the frnd2 lines round .rn or .rz only"
    if ftz or sat:
        return "the frnd2 lines carry neither .ftz nor .sat"
    return None


def _check_cvt_scalar(m):
    """The generic scalar line's rules, quoting ISA 9.7.9.22.

    Rounding: "Integer rounding is required for float-to-integer conversions,
    and for same-size float-to-float conversions where the value is rounded to
    an integer. Integer rounding is illegal in all other instances";
    "Floating-point rounding is required
    for float-to-float conversions that result in loss of precision, and for
    integer-to-float conversions ... illegal in all other instances."

    The same-size float-to-float clause covers six pairs of this entry's types:
    the four same-type ones, which `_check_cvt_same_type` rules on, plus
    `.f16`/`.bf16` in either order, which are same-size but not same-type and
    so fall to the `.dtype != .atype` case below. The rest of this function is
    that case.

    ``.ftz``: "can only be specified when either .dtype or .atype is .f32".

    ``.sat``: "For integer destination types ... allowed only in cases where
    the destination type's value range is not a superset of the source type's
    value range". For float destinations it clamps to [0.0, 1.0] instead, so
    the superset rule does not apply there.

    Three restrictions are ptxas's rather than the ISA's, and are recorded here
    because the certification pass would otherwise report legal-looking forms
    as illegal: this toolchain assembles no conversion between .bf16 and an
    8-bit integer, takes no `.sat` on any .bf16 operand, and refuses an integer
    rounding mode on the two same-size cross-type float pairs -- `cvt.rni.bf16.f16`
    and `cvt.rni.f16.bf16` are "Illegal rounding modifier for instruction 'cvt'"
    at ptxas 13.2 / sm_90 even though the Integer Notes clause quoted above
    requires integer rounding for exactly those conversions. The `.dtype !=
    .atype` branch below therefore rejects them, which is a toolchain verdict,
    not the ISA's.

    ``.relu``/``.satfinite`` belong to the two frnd2 scalar lines, which share
    this entry's shape and type list; `_check_cvt_frnd2_scalar` is their
    sub-grid, and it runs whenever either token is written.
    """
    d, a, rnd, ftz, sat = m["dtype"], m["atype"], m["rnd"], m["ftz"], m["sat"]
    if m["relu"] or m["satfinite"]:
        return _check_cvt_frnd2_scalar(d, a, rnd, ftz, sat)
    d_int, a_int = d in _CVT_INT_BITS, a in _CVT_INT_BITS
    # The rounding rules are the only ones that turn on whether the two types
    # are equal. `.ftz`, `.sat` and the two toolchain limits below apply to
    # both sub-grids, so they are stated once, after this branch.
    if d == a:
        if bad := _check_cvt_same_type(d, rnd):
            return bad
    else:
        if d_int and not a_int:
            if rnd not in _CVT_IRND:
                return "float-to-integer requires an integer rounding mode"
        elif rnd in _CVT_IRND:
            return "integer rounding is illegal outside float-to-integer"
        if not d_int and a_int:
            if rnd not in _CVT_FRND:
                return "integer-to-float requires a floating-point rounding mode"
        elif not d_int and not a_int:
            if _cvt_loses_precision(d, a):
                if rnd not in _CVT_FRND:
                    return f"{d} from {a} loses precision, so a rounding mode is required"
            elif rnd:
                return f"{d} from {a} is exact, so a rounding mode is illegal"
        elif d_int and a_int and rnd:
            return "integer-to-integer takes no rounding mode"
    if ftz and "f32" not in (d, a):
        return ".ftz applies only where .f32 is one of the types"
    if sat and d_int and a_int and _cvt_int_covers(d, a):
        return f"{d} already covers {a}, so saturation is not possible"
    if sat and "bf16" in (d, a):
        return "this toolchain assembles no .sat on a .bf16 operand"
    if "bf16" in (d, a) and (d in ("u8", "s8") or a in ("u8", "s8")):
        return "this toolchain assembles no .bf16 <-> 8-bit-integer conversion"
    return None


# Which multiplicand-type *set* each mma / mma.sp syntax line draws BOTH of its
# type positions from. A line never pairs a type from one set with a type from
# another -- ISA 9.7.15.5.14 spells the integer lines as
#
#     mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;
#     .atype   = {.u8, .s8};
#     .btype   = {.u8, .s8};
#
# and again, as a separate line, with ".atype   = {.u4, .s4};" / ".btype   =
# {.u4, .s4};" -- so `.u8` and `.u4` never meet. Neither the dense nor the
# sparse entries can leave that to their slot domains: `mma_int` offers all of
# {u8, s8, u4, s4, b1} on both .atype and .btype, `mma` offers all of {f16,
# bf16, tf32, e4m3, e5m2}, and `mma_sp_int_pair` / `mma_sp_int_all` offer all of
# {u8, s8, u4, s4}, so the width-class pairing is enforced by the checks below
# and nowhere else. (`mma_sp_all` is the one entry whose slot domain does the
# job on its own: its .atype/.btype are exactly `.f8type = {.e4m3, .e5m2}`.)
_MMA_INT_LINE = {"u8": "i8", "s8": "i8", "u4": "i4", "s4": "i4", "b1": "b1"}
# The one floating-point line that quantifies its two multiplicand positions
# independently: "mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype"
# with ".f8type = {.e4m3, .e5m2};" (ISA 9.7.15.5.14:23,28). Every other fp line
# writes a literal token in both positions (.f16.f16, .bf16.bf16, .tf32.tf32)
# or, at .m16n8k8, names .atype/.btype separately and then requires them equal
# (:129) -- so outside .f8type the two must simply match.
_MMA_F8 = ("e4m3", "e5m2")


def _check_mma_fp_pair(a: str, b: str) -> str | None:
    """The multiplicand pairing rule shared by the dense fp checks."""
    if a != b and not (a in _MMA_F8 and b in _MMA_F8):
        return f"no syntax line pairs .{a} with .{b}"
    return None


def _check_mma_sp_fp_types(m):
    """The sparse floating-point lines spell one literal token in both
    multiplicand positions, per ISA 9.7.15.6.3:8-21 --

        mma.spvariant.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype  d, a, b, c, e, f;
        mma.spvariant.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32     d, a, b, c, e, f;
        mma.spvariant.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32      d, a, b, c, e, f;

    -- so `.f16 x .bf16` and friends are simply not in the grammar. That is a
    fact about these three lines' type positions, NOT a general mma.sp rule.
    The section's one type-equality sentence is at :109-112, "The qualifiers
    .dtype, .atype, .btype and .ctype indicate the data-type of the elements in
    the matrices D, A, B and C respectively. The qualifier .stype indicate the
    data-type of the elements in the matrices scale_A and scale_B. In case of
    shapes .m16n8k16, .m16n8k32 and .m16n8k64, .dtype must be the same as
    .ctype." -- .dtype/.ctype only; grepping "must be the same" over the whole
    9.7.15.6.3 slice returns that line and nothing else, so the section states
    no .atype/.btype equality rule.

    The blanket "the multiplicand types must match" this check used to apply to
    every sparse entry came from the **wmma.mma** section, a different
    instruction: ISA 9.7.15.4.5:77-78, "For integer wmma, .ctype and .dtype must
    be specified as .s32. Also, the values for .atype and .btype must be the
    same, i.e., either both are .s8 or both are .u8." The sparse lines that do
    name two independent type variables -- ".s32.atype.btype.s32" and
    "...f32.f8type.f8type.f32" -- are handled by `_check_mma_sp_int_types` and
    by `mma_sp_all`'s slot domain, and they mix freely.

    (.dtype == .ctype needs no test here: every sparse entry pins .dtype and
    .ctype to the same one-element slot domain, so the :112 rule holds by
    construction.)
    """
    if m["atype"] != m["btype"]:
        return f"no sparse floating-point line pairs .{m['atype']} with .{m['btype']}"
    return None


def _check_mma_sp_int_types(m):
    """The sparse integer lines pair by width class, per ISA 9.7.15.6.3:60-71 --

        mma.spvariant.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c, e, f;
        .atype     = {.u8, .s8};
        .btype     = {.u8, .s8};

    and the same line again with ".atype     = {.u4, .s4};" / ".btype     =
    {.u4, .s4};". Each line quantifies its two positions independently, so
    mixed signedness (.u8 x .s8, .u4 x .s4) is in the grammar; only crossing
    the 8-bit and 4-bit lines is not.

    9.7.15.6.3 states no .atype/.btype equality rule (see
    `_check_mma_sp_fp_types` for the section's one type-equality sentence);
    the equality this check used to apply is wmma.mma's, ISA 9.7.15.4.5:77-78.
    """
    a, b = m["atype"], m["btype"]
    if _MMA_INT_LINE[a] != _MMA_INT_LINE[b]:
        return f"no syntax line pairs .{a} with .{b}"
    return None


def _check_mma_sp_fp_thread(m):
    """The floating-point lines whose selector names one thread of four.

    ISA 9.7.15.6.1: .f16/.bf16 at .m16n8k16 and .tf32 at .m16n8k8. Sparse
    doubles K against the dense line it mirrors, so these shape lists differ
    from the dense `_check_mma_fp_f32`'s.
    """
    if (bad := _check_mma_sp_fp_types(m)) is not None:
        return bad
    want = "m16n8k8" if m["atype"] == "tf32" else "m16n8k16"
    if m["shape"] != want:
        return f"the one-thread selector line for .{m['atype']} is {want}"
    return None


def _check_mma_sp_fp_pair(m):
    """The floating-point lines whose selector names a thread-pair (0 or 1)."""
    if (bad := _check_mma_sp_fp_types(m)) is not None:
        return bad
    want = "m16n8k16" if m["atype"] == "tf32" else "m16n8k32"
    if m["shape"] != want:
        return f"the thread-pair selector line for .{m['atype']} is {want}"
    return None


def _check_mma_sp_int_pair(m):
    """Integer lines with a thread-pair selector: .u8/.s8 at k32, .u4/.s4 at k64."""
    if (bad := _check_mma_sp_int_types(m)) is not None:
        return bad
    want = "m16n8k32" if m["atype"] in ("u8", "s8") else "m16n8k64"
    if m["shape"] != want:
        return f"the thread-pair selector line for .{m['atype']} is {want}"
    return None


def _check_mma_sp_int_all(m):
    """Integer lines where all four threads contribute: .u8/.s8 at k64, .u4/.s4 at k128."""
    if (bad := _check_mma_sp_int_types(m)) is not None:
        return bad
    want = "m16n8k64" if m["atype"] in ("u8", "s8") else "m16n8k128"
    if m["shape"] != want:
        return f"the all-thread selector line for .{m['atype']} is {want}"
    return None


def _check_mma_fp_f16(m):
    """The .f16-accumulator forms: the ISA's f16 and f8 lines.

    The half-precision lines (ISA 9.7.15.5.14:8-10, with ".ctype   = {.f16,
    .f32};" / ".dtype   = {.f16, .f32};" at :14-15) spell the token literally in
    both multiplicand positions --

        mma.sync.aligned.m8n8k4.alayout.blayout.dtype.f16.f16.ctype  d, a, b, c;
        mma.sync.aligned.m16n8k8.row.col.dtype.f16.f16.ctype  d, a, b, c;
        mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype d, a, b, c;

    -- while the f8 line, ISA 9.7.15.5.14:23 with ".f8type     = {.e4m3,
    .e5m2};" (:28) and ".shape      = {.m16n8k16, .m16n8k32};" (:32),

        mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b, c;

    quantifies its two positions independently, so `.e4m3 x .e5m2` is in the
    grammar. All that is left to check across the two positions is that they
    come from the same line -- the entry's .atype/.btype slot domains both offer
    every type either line uses.

    This check used to reject every `.atype != .btype`. That blanket rule is a
    sentence from the **wmma.mma** section, a different instruction: ISA
    9.7.15.4.5:77-78, "For integer wmma, .ctype and .dtype must be specified as
    .s32. Also, the values for .atype and .btype must be the same, i.e., either
    both are .s8 or both are .u8." mma's own restriction block
    (9.7.15.5.14:122-135) scopes ".atype must be the same as .btype." to
    .m16n8k8 alone, and this entry reaches .m16n8k8 only through the .f16.f16
    line above -- the f8 line's shapes are .m16n8k16 / .m16n8k32 -- so that rule
    holds by construction here.

    The block's other rule, ".dtype must be the same as .ctype." at .m16n8k8 /
    .m16n8k16 / .m16n8k32, needs no test either: this entry pins both slots to
    .f16.
    """
    shape, a = m["shape"], m["atype"]
    if bad := _check_mma_fp_pair(a, m["btype"]):
        return bad
    if shape != "m8n8k4" and (m["alayout"], m["blayout"]) != ("row", "col"):
        return f"{shape} is spelled .row.col"
    if a == "f16" and shape not in ("m8n8k4", "m16n8k8", "m16n8k16"):
        return f"{shape} has no .f16 multiplicand line"
    if a in ("e4m3", "e5m2") and shape not in ("m16n8k16", "m16n8k32"):
        return f"{shape} has no .f8 multiplicand line"
    return None


def _check_mma_fp_f32(m):
    """One check per floating-point syntax line of ISA 9.7.15.5.14.

    The lines differ in which shapes pair with which operand types, and only
    the .m8n8k4 line leaves the layouts free -- every other line spells
    `.row.col`.

    Multiplicand types are NOT equal in general. This check used to reject
    every `.atype != .btype`; that blanket rule is a sentence from the
    **wmma.mma** section, a different instruction -- ISA 9.7.15.4.5:77-78, "For
    integer wmma, .ctype and .dtype must be specified as .s32. Also, the values
    for .atype and .btype must be the same, i.e., either both are .s8 or both
    are .u8." mma's own restriction block, ISA 9.7.15.5.14:122-135, reads:

        Specific shapes have type restrictions :
        .m8n8k4 : When .ctype is .f32, .dtype must also be .f32.
        .m16n8k8 :
        .dtype must be the same as .ctype.
        .atype must be the same as .btype.
        .m16n8k16 and .m16n8k32 :
        .dtype must be the same as .ctype.

    so `.atype == .btype` binds at .m16n8k8 alone. That is exactly the one
    alternate-fp line whose two type positions are separate variables over one
    set (9.7.15.5.14:21,26-27):

        mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32      d, a, b, c;
        .atype      = {.bf16, .tf32};
        .btype      = {.bf16, .tf32};

    The other alternate-fp lines spell one literal token in both positions
    (":20 ...m16n8k4.row.col.f32.tf32.tf32.f32", ":22
    ...m16n8k16.row.col.f32.bf16.bf16.f32"), so on the lines this entry covers
    .bf16 never pairs with .tf32. The .f8type line does quantify its positions
    independently
    (":23 mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b,
    c;", ":28 .f8type     = {.e4m3, .e5m2};"), so `.e4m3 x .e5m2` renders.

    Which line a type belongs to is the check's job here: the entry's
    .atype/.btype slot domains both offer every type any of these lines uses.
    """
    shape, d, c = m["shape"], m["dtype"], m["ctype"]
    a, b = m["atype"], m["btype"]
    if bad := _check_mma_fp_pair(a, b):
        return bad
    if shape != "m8n8k4" and (m["alayout"], m["blayout"]) != ("row", "col"):
        return f"{shape} is spelled .row.col"
    if a == "f16":
        # "mma.m8n8k4 / m16n8k8 / m16n8k16 .dtype.f16.f16.ctype"
        if shape not in ("m8n8k4", "m16n8k8", "m16n8k16"):
            return f"{shape} has no .f16 multiplicand line"
        if shape == "m8n8k4" and c == "f32" and d != "f32":
            return "m8n8k4 with a .f32 accumulator must produce .f32"
    elif a == "tf32":
        # "m16n8k4 .f32.tf32.tf32.f32" and the m16n8k8 atype/btype line.
        if shape not in ("m16n8k4", "m16n8k8") or d != "f32" or c != "f32":
            return "the .tf32 lines are m16n8k4 / m16n8k8, .f32 in and out"
    elif a == "bf16":
        # "m16n8k16 .f32.bf16.bf16.f32" and the m16n8k8 atype/btype line.
        if shape not in ("m16n8k8", "m16n8k16") or d != "f32" or c != "f32":
            return "the .bf16 lines are m16n8k8 / m16n8k16, .f32 in and out"
    else:  # e4m3 / e5m2
        # "mma.shape.row.col.dtype.f8type.f8type.ctype", shape in k16/k32.
        if shape not in ("m16n8k16", "m16n8k32"):
            return f"{shape} has no .f8 multiplicand line"
    if shape in ("m16n8k8", "m16n8k16", "m16n8k32") and d != c:
        # ".dtype must be the same as .ctype" on these shapes.
        return f"{shape} requires .dtype == .ctype"
    return None


def _check_mma_int(m):
    """The integer / sub-byte / single-bit lines of ISA 9.7.15.5.14.

    Mixed signedness is legal. This check used to reject every `.atype !=
    .btype` under the comment "the values for .atype and .btype must be the
    same" -- that sentence is the **wmma.mma** section's, a different
    instruction: ISA 9.7.15.4.5:77-78, "For integer wmma, .ctype and .dtype
    must be specified as .s32. Also, the values for .atype and .btype must be
    the same, i.e., either both are .s8 or both are .u8." mma's own restriction
    block (9.7.15.5.14:122-135) states no rule for the integer lines at all,
    and scopes ".atype must be the same as .btype." to .m16n8k8 -- a shape no
    integer or single-bit line lists. The integer lines quantify their two
    positions independently (9.7.15.5.14:67-77):

        mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;
        .shape   = {.m8n8k16, .m16n8k16, .m16n8k32}
        .atype   = {.u8, .s8};
        .btype   = {.u8, .s8};

    plus the same line again with ".shape   = {.m8n8k32, .m16n8k32,
    .m16n8k64}" / ".atype   = {.u4, .s4};" / ".btype   = {.u4, .s4};". What
    survives is the width-class pairing those two lines imply -- .u8 never
    meets .u4 -- which this check enforces, since the entry offers all of
    {u8, s8, u4, s4, b1} on both slots.

    ".dtype must be the same as .ctype." needs no test: the entry pins both to
    .s32.
    """
    shape, a, b = m["shape"], m["atype"], m["btype"]
    if _MMA_INT_LINE[a] != _MMA_INT_LINE[b]:
        return f"no syntax line pairs .{a} with .{b}"
    if a in ("u8", "s8"):
        if shape not in ("m8n8k16", "m16n8k16", "m16n8k32"):
            return f"{shape} has no 8-bit integer line"
    elif a in ("u4", "s4"):
        if shape not in ("m8n8k32", "m16n8k32", "m16n8k64"):
            return f"{shape} has no 4-bit integer line"
    else:  # b1
        if shape not in ("m8n8k128", "m16n8k128", "m16n8k256"):
            return f"{shape} has no single-bit line"
        if not m["bitop"]:
            # "mma.sync...s32.b1.b1.s32.bitOp.popc" -- bitOp is not optional.
            return "the single-bit line requires .xor or .and"
    if a != "b1" and (m["bitop"] or m["popc"]):
        # `.bitOp.popc` belongs to the single-bit line alone; ptxas otherwise
        # reports "Unexpected instruction types specified for 'mma'".
        return "only the single-bit line takes .bitOp.popc"
    if a == "b1":
        if m["satfinite"]:
            return "the single-bit line takes no .satfinite"
        if not m["popc"]:
            return "the single-bit line spells .popc"
    return None


# wgmma.mma_async register fragments, per ISA 9.7.16.5.1.1 (Register
# Fragments): across the 128-thread warpgroup the accumulator D holds
# M*N/128 = N/2 registers per thread (.f32 and .s32), and M*N/256 = N/4
# when .dtype is .f16 (two halves per register). The A fragment of the rs
# form works out to M*K/128/(32/bits) = 4 registers for every (K, type)
# pairing the ISA defines, so it is a plain `lanes=4`, not a function.
#
# The N domains, straight from each syntax line's `.shape =` set: the
# floating-point and fp8 lines take every multiple of 8 up to 256; the
# s8/u8 line drops 40, 56, ... (the odd multiples of 8 above 32) and stops
# at 224; the single-bit line adds 240 and 256 back.
_WGMMA_N_FULL = tuple(str(8 * i) for i in range(1, 33))
_WGMMA_N_S8 = ("8", "16", "24", "32", "48", "64", "80", "96", "112", "128",
               "144", "160", "176", "192", "208", "224")  # fmt: skip
_WGMMA_N_B1 = (*_WGMMA_N_S8, "240", "256")


def _wgmma_acc_lanes(m):
    n = int(m["shape"].split("n")[1].split("k")[0])
    return n // 4 if m["dtype"] == "f16" else n // 2


# cp.async.bulk.tensor coordinate vectors, per ISA 9.7.9.26.5.2: "Vector of
# n elements where n = .dim" -- except the gather4/scatter4 load modes, whose
# tensorCoords is a "fixed length vector of size 5" (one column index plus
# four row indices) whatever the dimension.
def _tma_coords_lanes(m):
    if m["load_mode"] in ("tile::gather4", "tile::scatter4"):
        return 5
    return int(m["dim"][0])


# `{, ctaMask}` exists exactly when `.multicast::cluster` is written.
_tma_mask_lanes = _present_lanes("multicast")


# `{, cache_policy}` exists exactly when `.L2::cache_hint` is written.
_tma_cache_lanes = _present_lanes("cache")


def _check_tma_gather4(m):
    """gather4/scatter4 reindex four rows of a 2D tensor (ISA: "the four rows
    in the 2-dimensional tensor"), so ptxas rejects every other .dim."""
    if m["load_mode"] in ("tile::gather4", "tile::scatter4") and m["dim"] != "2d":
        return f"{m['load_mode']} is a 2d-only load mode"
    return None


# tcgen05.mma disable-output-lane, per ISA 9.7.17.10.9.1: "The size of the
# vector is as follows: .cta_group::1 -> 4, .cta_group::2 -> 8".
def _tcgen05_mma_mask_lanes(m):
    return 8 if m["cta_group"] == "cta_group::2" else 4


def _check_tcgen05_mma_block_scale(m):
    """Valid .scale_vec sizes per kind (ISA "Scale factor" table): mxf8f6f4
    scales in 1X, mxf4 in 2X, mxf4nvf4 in 2X or 4X."""
    valid = {
        "kind::mxf8f6f4": ("scale_vec::1X",),
        "kind::mxf4": ("scale_vec::2X",),
        "kind::mxf4nvf4": ("scale_vec::2X", "scale_vec::4X"),
    }[m["kind"]]
    if m["scale_vec"] not in valid:
        return f"{m['kind']} scales in {'/'.join(valid)}"
    return None


# `{, byteMask}` exists exactly when `.cp_mask` is written.
_cp_mask_lanes = _present_lanes("cp_mask")


# `{, ignoreBytesLeft, ignoreBytesRight}` exists exactly when `.ignore_oob` is
# written -- one lane each, so the pair appears and disappears together.
_ignore_oob_lanes = _present_lanes("ignore_oob")


_ENTRIES = [
    # prefetch per PTX ISA 9.7.9.16, covering three of its four syntax lines:
    #   prefetch{.space}.level [a]
    #   prefetch.global.level::eviction_priority [a]
    #   prefetch{.tensormap_space}.tensormap [a]
    # `prefetchu.L1` is a different mnemonic and would be its own entry.
    InstructionEntry(
        name="prefetch",
        slots=(
            ModifierSlot("space", ("global", "local", "const", "param"), optional=True),
            ModifierSlot("level", ("L1", "L2"), optional=True),
            ModifierSlot("evict", ("L2::evict_last", "L2::evict_normal"), optional=True),
            ModifierSlot("tensormap", ("tensormap",), optional=True),
        ),
        check=_check_prefetch,
        operands=(OperandSlot("addr", role="addr"),),
    ),
    # Complete scalar `ld` per PTX ISA 9.7.9.8 + the 9.7.9.9 ld.global.nc forms.
    # NOT REGISTERED: each needs a mechanism this shape lacks --
    # - .level::cache_hint + the cache_policy operand (optional operand)
    # - .unified (variable-attribute addressing)
    # - .param/.const spaces (require kernel-parameter / const addresses,
    #   which cannot flow through the helper-function ABI)
    # The ISA permits @p on this instruction; ptxd does not, because it writes a
    # destination -- see InstructionEntry.has_dst for why that needs a "+"
    # constraint first.
    InstructionEntry(
        name="ld",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("weak", "acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys"), optional=True),
            # Must be named "space": that is the slot an `addr` operand reads to
            # choose between a 32-bit shared-window address and a generic
            # pointer. Any other name silently leaves every shared form
            # rendering a 64-bit generic pointer, which ptxas accepts (it
            # truncates) but which addresses the wrong thing.
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,  # omitted = generic addressing
            ),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_ld,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("addr", role="addr"),
        ),
    ),
    # Complete scalar `st` per PTX ISA 9.7.9.11, at parity with `ld`.
    # NOT REGISTERED: each needs a mechanism this shape lacks --
    # - .level::cache_hint + its trailing cache_policy operand (optional operand)
    # - .param::func (kernel-parameter addresses cannot flow through the
    #   helper-function ABI)
    InstructionEntry(
        name="st",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("weak", "release", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,  # omitted = generic addressing
            ),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_st,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
    ),
    # The `.vec` lines of ld / st (PTX ISA 9.7.9.8 / 9.7.9.11). Separate
    # entries rather than an optional slot on the scalar ones: a vector operand
    # is brace-enclosed even at one register, so vector-ness has to be a
    # property of the entry, and the .v8/.v4-64bit lines carry a
    # .level2::eviction_priority the scalar lines do not have.
    #
    # The memory-synchronization lines carry `{.vec}` too, so `.relaxed` and
    # `.acquire`/`.release` are on every entry below, each with the `.scope`
    # its line makes mandatory. PTX ISA 9.7.9.8, wrapped but otherwise verbatim:
    #
    #   ld.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}
    #      {.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache_policy};
    #   ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}
    #      {.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache_policy};
    #
    # and the 9.7.9.11 mirror with `.relaxed`/`.release` and no prefetch term.
    # Note these lines carry *both* eviction priorities, which is why the
    # 256-bit entries let `.L2::*` ride them -- unlike `.volatile`, whose line
    # (`ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];`) spells no
    # eviction position at all. Every rule beyond that is the scalar rule:
    # `_check_ld`/`_check_st` read both eviction priorities as one qualifier.
    #
    # NOT REGISTERED: the sink symbol `_` in the two 256-bit lines (it needs an
    # operand role for "this lane is discarded"), .level::cache_hint with
    # its trailing cache_policy operand, and `.mmio`, whose syntax line carries
    # no `{.vec}`.
    InstructionEntry(
        name="ld_vec",
        mnemonic="ld",
        slots=(
            ModifierSlot("sem", ("weak", "acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,
            ),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_ld_vec,
        operands=(
            OperandSlot("d", role="dst", lanes=_vec_lanes),
            OperandSlot("addr", role="addr"),
        ),
    ),
    InstructionEntry(
        # The two 256-bit lines. ptxas: "Feature '256 bit wide load/store'
        # requires .target sm_100 or higher", and .level2::eviction_priority is
        # spelled only here.
        name="ld_vec256",
        mnemonic="ld",
        slots=(
            ModifierSlot("sem", ("weak", "acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("l2ev", _L2_EVICT, optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("vec", ("v4", "v8")),
            ModifierSlot("type", _BITS32 + _BITS64),
        ),
        cert_arch="sm_100",
        check=_check_ld_vec256,
        operands=(
            OperandSlot("d", role="dst", lanes=_vec_lanes),
            OperandSlot("addr", role="addr"),
        ),
    ),
    InstructionEntry(
        name="st_vec",
        mnemonic="st",
        slots=(
            ModifierSlot("sem", ("weak", "release", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,
            ),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_st_vec,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value", lanes=_vec_lanes),
        ),
    ),
    InstructionEntry(
        name="st_vec256",
        mnemonic="st",
        slots=(
            ModifierSlot("sem", ("weak", "release", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("l2ev", _L2_EVICT, optional=True),
            ModifierSlot("vec", ("v4", "v8")),
            ModifierSlot("type", _BITS32 + _BITS64),
        ),
        cert_arch="sm_100",
        check=_check_st_vec256,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value", lanes=_vec_lanes),
        ),
    ),
    # red / atom scalar `.op` forms per PTX ISA 9.7.14.6 and 9.7.14.5.
    # NOT REGISTERED: each needs a mechanism this shape lacks --
    # - {.level::cache_hint} with its trailing cache_policy operand
    # - the .vec_16_bit/.vec_32_bit vector forms
    # - .f16/.bf16/.f16x2/.bf16x2 (add.noftz), which need half carrier types
    # - atom's .cas (3 operands) and .exch (own type set): other syntax shapes
    InstructionEntry(
        name="red",
        slots=(
            ModifierSlot("sem", _RED_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
    ),
    # The ISA permits @p on this instruction; ptxd does not, because it writes a
    # destination -- see InstructionEntry.has_dst for why that needs a "+"
    # constraint first.
    InstructionEntry(
        name="atom",
        slots=(
            ModifierSlot("sem", _ATOM_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
    ),
    # ex2 per PTX ISA 9.7.3.21 (`ex2.approx{.ftz}.f32`). The half-precision
    # forms of 9.7.4.10 (.f16/.f16x2/.bf16/.bf16x2) are deliberately excluded:
    # .f16/.bf16 have carriers now, but .f16x2/.bf16x2 need a b32 carrier and
    # .ftz is mandatory on the bf16 line while illegal on the f16 line, so they
    # cannot share this entry's optional ftz slot.
    InstructionEntry(
        name="ex2",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32",)),
        ),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("value", role="value"),
        ),
    ),
    # rcp per PTX ISA 9.7.3.13. `rcp.approx.ftz.f64` (a separate syntax line in
    # the ISA, with its own sm floor) is excluded until it is needed.
    InstructionEntry(
        name="rcp",
        slots=(
            ModifierSlot("mode", ("approx", "rn", "rz", "rm", "rp")),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_rcp,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("value", role="value"),
        ),
    ),
    # fns per PTX ISA 9.7.1.18: `fns.b32 d, mask, base, offset;` — one form.
    # The operands carry three different types (mask .b32, base .b32/.u32/.s32,
    # offset .s32), so each declares its own dtype rather than sharing the
    # entry's type slot.
    InstructionEntry(
        name="fns",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("mask", role="value", dtype="b32"),
            OperandSlot("base", role="value", dtype="b32"),
            OperandSlot("offset", role="value", dtype="s32"),
        ),
        asm_volatile=False,  # legacy fns carried no barrier
    ),
    # st.bulk per PTX ISA 9.7.9.14:
    #   st.bulk{.weak}{.shared::cta} [a], size, initval;  // initval must be zero
    # NOT REGISTERED: the 32-bit `size` form (ISA: "The 32-bit or 64-bit integer
    # operand size ..."), because no modifier token distinguishes the two -- it
    # would be an operand-shape axis, like mov's. Two ISA constraints are also
    # unenforceable here, being properties of a value rather than of the
    # modifier map that check() sees: "size must be a multiple of 8" and "The
    # maximum value of size operand can be 16777216".
    InstructionEntry(
        name="st_bulk",
        mnemonic="st.bulk",
        cert_arch="sm_100",  # PTX ISA 8.6; ISA: "Requires sm_100 or higher."
        slots=(
            ModifierSlot("weak", ("weak",), optional=True),
            ModifierSlot("space", ("shared::cta",), optional=True),
        ),
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("size", role="value", dtype="u64"),
            OperandSlot("initval", role="imm", literal="0"),
        ),
    ),
    # min / max per PTX ISA 9.7.1.13, 9.7.1.14 (integer), 9.7.3.11, 9.7.3.12
    # (single/double), 9.7.4.7, 9.7.4.8 (half).
    # Each mnemonic gets two entries because the ISA gives it two operand
    # shapes: the usual `d, a, b` and the three-source `d, a, b, c` line.
    # NOT REGISTERED: only the `.u8x4` and `{.relu}.s8x4` tokens of the integer
    # lines (reason at the `_MINMAX_INT_PLAIN` note above: sm_120f-only). Every
    # other syntax line and type token of both families is here.
    *[
        InstructionEntry(
            name=name,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("nan", ("NaN",), optional=True),
                ModifierSlot("xorsign", ("xorsign",), optional=True),
                ModifierSlot("abs", ("abs",), optional=True),
                ModifierSlot("relu", ("relu",), optional=True),
                ModifierSlot("type", (*_MINMAX_FP, *_MINMAX_INT_PLAIN, *_MINMAX_INT_RELU)),
            ),
            check=_check_minmax,
            # `.u16x2`, `{.relu}.s16x2` and `.relu.s32` need sm_90 (ISA Target
            # Notes); cert_arch is the max over the entry's variants.
            cert_arch="sm_90",
            operands=(
                OperandSlot("d", role="dst"),
                OperandSlot("a", role="value"),
                OperandSlot("b", role="value"),
            ),
        )
        for name in ("max", "min")
    ],
    *[
        InstructionEntry(
            name=f"{name}3",
            mnemonic=name,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("nan", ("NaN",), optional=True),
                ModifierSlot("abs", ("abs",), optional=True),
                ModifierSlot("type", ("f32",)),
            ),
            check=_check_minmax3,
            cert_arch="sm_100",  # "max.f32 with 3 input operands" requires sm_100
            operands=(
                OperandSlot("d", role="dst"),
                OperandSlot("a", role="value"),
                OperandSlot("b", role="value"),
                OperandSlot("c", role="value"),
            ),
        )
        for name in ("max", "min")
    ],
    # Floating-point add/sub/mul (PTX ISA 9.7.3.{3,4,5}) together with their
    # mixed-precision lines (9.7.5.{1,2}); `mul` has no mixed-precision line.
    #   add{.rnd}{.ftz}{.sat}.f32  d, a, b;   add{.rnd}{.ftz}.f32x2  d, a, b;
    #   add{.rnd}.f64              d, a, b;   add{.rnd}{.sat}.f32.atype  d, a, c;
    # cert_arch is the family's ceiling, not its floor: .f32/.f64 assemble
    # everywhere, but .f32x2 and the mixed lines need sm_100, and certification
    # has to run somewhere every legal variant is legal.
    *[
        InstructionEntry(
            name=name,
            cert_arch="sm_100",
            slots=(
                ModifierSlot("rnd", _FRND, optional=True),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("sat", ("sat",), optional=True),
                ModifierSlot("type", ("f32", "f64", "f32x2")),
                *((ModifierSlot("srctype", ("f16", "bf16"), optional=True),) if mixed else ()),
            ),
            check=_check_farith,
            operands=(
                OperandSlot("d", role="dst"),
                # On the mixed line `a` is the converted 16-bit source; on every
                # other line it is just the instruction type.
                OperandSlot("a", role="value", dtype="srctype" if mixed else None),
                OperandSlot("b", role="value"),
            ),
        )
        # `mul` is the one line with no mixed-precision form (ISA 9.7.5).
        for name, mixed in (("add", True), ("sub", True), ("mul", False))
    ],
    # Half-precision add/sub/mul (PTX ISA 9.7.4.{1,2,3}). Same operand shape as
    # the single/double lines above, so they differ only in their slot domains:
    #   sub{.rnd}{.ftz}{.sat}.f16 / .f16x2   sub{.rnd}.bf16 / .bf16x2   .rnd = {.rn}
    *[
        InstructionEntry(
            name=f"{name}_half",
            mnemonic=name,
            slots=(
                ModifierSlot("rnd", ("rn",), optional=True),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("sat", ("sat",), optional=True),
                ModifierSlot("type", ("f16", "f16x2", "bf16", "bf16x2")),
            ),
            check=_check_half_arith,
            operands=(
                OperandSlot("d", role="dst"),
                OperandSlot("a", role="value"),
                OperandSlot("b", role="value"),
            ),
        )
        for name in ("add", "sub", "mul")
    ],
    # neg (PTX ISA 9.7.3.10): `neg{.ftz}.f32 d, a;` and `neg.f64 d, a;`.
    InstructionEntry(
        name="neg",
        slots=(
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_neg,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="value"),
        ),
    ),
    # NOT REGISTERED: across this whole arithmetic group, the integer lines
    # (9.7.1.{1,2,3}), extended-precision add.cc/sub.cc (9.7.2.{1,3}), and the
    # half-precision fma line (9.7.4.4), which needs `.relu` and `.oob` slots.
    # Those two qualifiers appear on no add/sub/mul half line, which is why
    # 9.7.4.{1,2,3} are registered above and 9.7.4.4 is not.
    #
    # fma differs in shape, so it is its own entry: three sources, and .rnd is
    # mandatory on every line (PTX ISA 9.7.3.6 / 9.7.5.3).
    #   fma.rnd{.ftz}{.sat}.f32  d, a, b, c;   fma.rnd{.ftz}.f32x2  d, a, b, c;
    #   fma.rnd.f64              d, a, b, c;   fma.rnd{.sat}.f32.abtype  d, a, b, c;
    InstructionEntry(
        name="fma",
        cert_arch="sm_100",
        slots=(
            ModifierSlot("rnd", _FRND),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", ("f32", "f64", "f32x2")),
            ModifierSlot("srctype", ("f16", "bf16"), optional=True),
        ),
        check=_check_farith,
        operands=(
            OperandSlot("d", role="dst"),
            # .abtype converts both a and b; c is always the instruction type.
            OperandSlot("a", role="value", dtype="srctype"),
            OperandSlot("b", role="value", dtype="srctype"),
            OperandSlot("c", role="value"),
        ),
    ),
    # ------------------------------------------------------------------
    # mov, vector pack/unpack form (PTX ISA 9.7.9.4)
    #
    #   mov.type  d, a;   .type = {.b16, .b32, .b64, .b128}
    #
    # `.type` is the *aggregate* width, not the element width -- the ISA only
    # requires that "the overall size of the vector and the size of the scalar
    # must match the size of the instruction type". Neither the lane count nor
    # the lane type appears in the instruction text (the doc's own example is
    # `mov.b64 {lo,hi}, %x;  // %x is a double; lo,hi are .u32`), so both are
    # part of the operand shape, and each (direction, lanes, lane type) is its
    # own entry. They all share mnemonic "mov", so the emitted opcode is right
    # and the call spelling stays `T.ptxd.mov.b64(...)`.
    #
    # NOT REGISTERED -- legal PTX that CUDA C inline asm cannot express:
    #   mov.b16 d, {a,b}      2 x 8-bit lanes
    #   mov.b32 d, {a,b,c,d}  4 x 8-bit lanes
    # Inline asm has no 8-bit register constraint (only "h"/"r"/"l"/"q"/"f"/"d"),
    # so 8-bit values ride a 16-bit carrier and ptxas rejects the widths with
    # "Arguments mismatch for instruction 'mov'" (4x16 != 32). Both forms are
    # legal in hand-written PTX, and they are unreachable via make_uchar4 too
    # (nvcc emits shl/or, not the mov). Declaring `.reg .b8` inside the asm
    # block assembles but needs four cvt instructions to get the values in --
    # a multi-instruction template, which this dialect forbids.
    #
    # Also unregistered: the sink symbol `_` (ISA: "the sink symbol '_' may be
    # used for one or more elements"), which does assemble from inline asm but
    # needs an operand role for "this lane is discarded"; and scalar mov
    # (9.7.9.3), a different instruction that shares the mnemonic.
    *[
        InstructionEntry(
            name=f"mov_{direction}_{lane_dtype}x{lanes}",
            mnemonic="mov",
            slots=(ModifierSlot("type", (agg,)),),
            operands=(
                OperandSlot(
                    "d",
                    role="dst",
                    dtype=lane_dtype if unpack else agg,
                    lanes=lanes if unpack else 1,
                ),
                OperandSlot(
                    "a",
                    role="value",
                    dtype=agg if unpack else lane_dtype,
                    lanes=1 if unpack else lanes,
                ),
            ),
            # .b128 needs PTX ISA 8.3 / sm_70; the sm_90 certification default
            # already clears that, so no cert_arch is needed.
            asm_volatile=False,  # a register shuffle: let nvcc common it up
        )
        # Lane types are the bit types only: the dtype axis already lets a
        # `.b32` lane be int32 or float32, so a separate `f32` entry would be a
        # shape-for-shape duplicate and make the shared-mnemonic dispatch
        # ambiguous.
        for agg, lanes, lane_dtype in (
            ("b32", 2, "b16"),
            ("b64", 2, "b32"),
            ("b64", 4, "b16"),
            ("b128", 2, "b64"),
            ("b128", 4, "b32"),
        )
        for direction, unpack in (("pack", False), ("unpack", True))
    ],
    # cvta per PTX ISA 9.7.9.21. This entry exists to serve the engine's
    # shared-address coercion, so it registers exactly the one combination that
    # needs: 1 of the 32 legal (direction x space x size) forms.
    # NOT REGISTERED: the whole space->generic direction, seven of the eight
    # state spaces, and `.u32` -- the last genuinely unusable, since ptxas
    # rejects the 32-bit ABI on sm_90 and higher.
    InstructionEntry(
        name="cvta",
        slots=(
            ModifierSlot("dir", ("to",)),
            ModifierSlot("space", ("shared",)),
            ModifierSlot("type", ("u64",)),
        ),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("ptr", role="ptr"),
        ),
        asm_volatile=False,  # legacy cvta carried no barrier
    ),
    # cvt per PTX ISA 9.7.9.22.
    #
    # The generic scalar line first: `cvt{.irnd|.frnd}{.ftz}{.sat}.dtype.atype`
    # over the ISA's twelve-type dtype/atype product. The two spellings the ISA
    # writes as separate lines share one operand shape, so they are one entry
    # whose check picks which rounding set applies -- otherwise both would
    # render the same no-rounding variants.
    #
    # The two frnd2 *scalar* lines join them for the same reason:
    #
    #     cvt.frnd2{.relu}{.satfinite}.f16.f32       d, a;
    #     cvt.frnd2{.relu}{.satfinite}.bf16.f32      d, a;
    #
    # are `d, a` over two (dtype, atype) pairs this entry's slots already
    # spell, so what they add is the `.relu` and `.satfinite` tokens. Both are
    # optional slots and neither rides any other pair -- see
    # `_check_cvt_frnd2_scalar`, whose sub-grid runs whenever one is written.
    # Splitting them out instead would have made `cvt.rn.f16.f32` a variant of
    # two entries, since it is a legal spelling of both lines.
    #
    # The rules are the ISA's own, quoted in `_check_cvt_scalar`. Their product
    # was checked against ptxas over the 5184 no-relu/no-satfinite
    # combinations: every one of the 780 variants that sweep found this entry
    # renders assembles. The .relu/.satfinite slots added later widen the grid
    # to 20736 and the rendered set to 792; the twelve added variants are
    # covered by the certification pass rather than by that sweep.
    #
    # ptxas is more lenient than the ISA in fourteen bf16 spellings (it takes
    # `cvt.bf16.f16` with no rounding though the conversion loses precision,
    # and `cvt.rn.f32.bf16` though widening is exact). The table follows the
    # ISA and does not offer them.
    InstructionEntry(
        name="cvt",
        slots=(
            ModifierSlot(
                "rnd", ("rni", "rzi", "rmi", "rpi", "rn", "rz", "rm", "rp"), optional=True
            ),
            # The frnd2 lines write these two ahead of the types and carry no
            # `{.ftz}`/`{.sat}`, so the render order agrees with both lines.
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("dtype", _CVT_BASIC),
            ModifierSlot("atype", _CVT_BASIC),
        ),
        check=_check_cvt_scalar,
        # No floor of its own: the highest one this entry reaches is the .bf16
        # conversions' (":520-522 cvt.bf16.{u8/s8/u16/s16/u32/s32/u64/s64/f16/
        # f64/bf16}, cvt.{u8/s8/u16/s16/u32/s32/u64/s64/f16/f64}.bf16, and
        # cvt.tf32.f32.{relu}.{rn/rz} require sm_90 or higher."), which is the
        # arch the certification already defaults to. The two tokens added for
        # the frnd2 lines stay under it -- ":517-518 .relu modifier and
        # {.f16x2, .bf16, .bf16x2, .tf32} destination formats require sm_80 or
        # higher."
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
    ),
    # The two frnd2 lines that pack *two* .f32 sources into one register are a
    # different shape (`d, a, b`), so they are their own entries. ISA
    # 9.7.9.22:65-68: "For .f16x2 and .bf16x2 instruction type, two inputs a and
    # b of .f32 type are converted into .f16 or .bf16 type and the converted
    # values are packed in the destination register d, such that the value
    # converted from input a is stored in the upper half of d and the value
    # converted from input b is stored in the lower half of d".
    #
    # Carriers, ISA:69-71: "For .f16x2 instruction type, destination operand d
    # has .f16x2 or .b32 type. ... For .bf16x2 instruction type, operand d has
    # .b32 type."
    InstructionEntry(  # cvt.frnd2{.relu}{.satfinite}.f16x2.f32 d, a, b;
        name="cvt_f16x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn", "rz")),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("f16x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        # ":517-518 .relu modifier and {.f16x2, .bf16, .bf16x2, .tf32}
        # destination formats require sm_80 or higher." -- the whole entry is
        # under the sm_90 default, so it states no floor of its own.
        operands=(
            OperandSlot("d", role="dst", dtype="f16x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
    ),
    InstructionEntry(  # cvt.frnd2{.relu}{.satfinite}.bf16x2.f32 d, a, b;
        name="cvt_bf16x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn", "rz")),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
    ),
    # Both .tf32 lines, in one entry: see `_check_cvt_tf32` for why .rna and
    # .frnd2 share it. ISA:71 "For .tf32 instruction type, operand d has .b32
    # type."
    InstructionEntry(  # cvt.rna{.satfinite}.tf32.f32 / cvt.frnd2{.satfinite}{.relu}.tf32.f32 d, a;
        name="cvt_tf32_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rna", "rn", "rz")),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("tf32",)),
            ModifierSlot("atype", ("f32",)),
        ),
        check=_check_cvt_tf32,
        # The maximum floor over this entry's variants, ISA:526:
        # "cvt.{rn/rz}.satfinite.tf32.f32 requires sm_100 or higher." The rest
        # of the entry sits far lower (:521-522 puts cvt.tf32.f32.{relu}.{rn/rz}
        # at sm_90), but certifying the entry below sm_100 would report those
        # four satfinite spellings as illegal.
        cert_arch="sm_100",
        operands=(
            OperandSlot("d", role="dst", dtype="tf32"),
            OperandSlot("a", role="value", dtype="f32"),
        ),
    ),
    # The .rs (stochastic rounding) lines. Their trailing operand is one more
    # register, so each is its own shape: ISA:183 "rbits is a .b32 type register
    # operand used for providing random bits for .rs rounding mode."
    #
    # Every .rs entry certifies at sm_100a: ISA:602 ".rs rounding mode is
    # supported on following architectures:", and the list at :604-605 is
    # sm_100a and sm_103a.
    InstructionEntry(  # cvt.rs{.relu}{.satfinite}.f16x2.f32 d, a, b, rbits;
        name="cvt_rs_f16x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rs",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("f16x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="f16x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
            OperandSlot("rbits", role="value", dtype="b32"),
        ),
    ),
    InstructionEntry(  # cvt.rs{.relu}{.satfinite}.bf16x2.f32 d, a, b, rbits;
        name="cvt_rs_bf16x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rs",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
            OperandSlot("rbits", role="value", dtype="b32"),
        ),
    ),
    InstructionEntry(  # cvt.frnd3{.satfinite}.ue8m0x2.f32 d, a, b;
        name="cvt_ue8m0x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rz", "rp")),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("ue8m0x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        # bf16x2 / ue8m0x2 conversions are Blackwell lines.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="ue8m0x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
    ),
    InstructionEntry(  # cvt.frnd3{.satfinite}.ue8m0x2.bf16x2 d, a;
        name="cvt_ue8m0x2_bf16x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rz", "rp")),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("ue8m0x2",)),
            ModifierSlot("atype", ("bf16x2",)),
        ),
        # bf16x2 / ue8m0x2 conversions are Blackwell lines.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="ue8m0x2"),
            OperandSlot("a", role="value", dtype="bf16x2"),
        ),
    ),
    InstructionEntry(  # cvt.rn.bf16x2.ue8m0x2 d, a;
        name="cvt_bf16x2_ue8m0x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("ue8m0x2",)),
        ),
        # bf16x2 / ue8m0x2 conversions are Blackwell lines.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="ue8m0x2"),
        ),
    ),
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f8x2type.f32 d, a, b;
        name="cvt_f8x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e4m3x2", "e5m2x2")),
            ModifierSlot("atype", ("f32",)),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
    ),
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f8x2type.fp16x2 d, a;
        name="cvt_f8x2_fp16x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e4m3x2", "e5m2x2")),
            ModifierSlot("atype", ("f16x2", "bf16x2")),
        ),
        # The .bf16x2 source is the PTX ISA 9.2 line: "cvt.rn.satfinite{.relu}
        # {.e5m2x2/.e4m3x2}{.bf16x2} is supported on following family-specific
        # architectures:" (9.7.9.22), listing sm_100f, sm_110f and sm_120f.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
    ),
    InstructionEntry(  # cvt.rn{.relu}.f16x2.f8x2type d, a;
        name="cvt_f16x2_f8x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("f16x2",)),
            ModifierSlot("atype", ("e4m3x2", "e5m2x2")),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="f16x2"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
    ),
    # The scale-factor operand: ISA:86-87 "For .bf16x2 destination type optional
    # scale-factor operand of type .b16 can be specified along with
    # .scaled::n2::ue8m0 qualifier. Operand scale-factor stores two packed
    # scaling factors of type .ue8m0." It is present exactly when the qualifier
    # is written (`_cvt_scale_lanes`), so the qualifier and the operand are one
    # optional slot plus one 0-or-1 length function, not a second entry -- the
    # operand list is `d, a` either way, with one register appended.
    #
    # The carrier is the 16-bit register the ISA's `.b16` names; the table binds
    # it through the `.ue8m0x2` token, whose entry in PTX_TYPE_DTYPES is that
    # one carrier, so the unscaled variants keep a single dtype combination.
    # cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f8x2type
    #     d, a{, scale-factor};
    InstructionEntry(
        name="cvt_bf16x2_f8x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("e4m3x2", "e5m2x2")),
        ),
        # ISA:634-639 puts this line on "following family-specific
        # architectures": "sm_100f or higher in the same family", "sm_110f or
        # higher in the same family", "sm_120f or higher in the same family".
        # The entry certifies at sm_100a, which ptxas 13.2 accepts for every
        # variant here -- a toolchain fact; the sentence above is the rule.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="atype"),
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
    ),
    InstructionEntry(  # cvt.rs{.relu}.satfinite.f8x4type.f32 d, {a, b, e, f}, rbits;
        name="cvt_rs_f8x4_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rs",)),
            ModifierSlot("relu", ("relu",), optional=True),
            # Unbracketed on the line, and ISA:204-208 ".satfinite modifier is
            # only supported for conversions involving the following types:
            # .e4m3x2, .e5m2x2, .e2m1x2, .e2m3x2, .e3m2x2, .e4m3x4, .e5m2x4,
            # .e2m1x4, .e2m3x4, .e3m2x4, .s2f6x2 destination types. .satfinite
            # modifier is mandatory for such conversions."
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("dtype", ("e4m3x4", "e5m2x4")),
            ModifierSlot("atype", ("f32",)),
        ),
        # ISA:607 "cvt.rs{.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4}.f32 is
        # supported on following architectures:", listing sm_100a and sm_103a.
        cert_arch="sm_100a",
        operands=(
            # ISA:138-140 "When converting to .e5m2x4/.e4m3x4/.e3m2x4/.e2m3x4
            # data format, the destination operand d has .b32 type."
            OperandSlot("d", role="dst", dtype="dtype"),
            # The four sources are one brace-enclosed group in the operand
            # list, which is what `lanes` renders; the name is the ISA's own
            # spelling of the group, `{a, b, e, f}`.
            OperandSlot("abef", role="value", dtype="f32", lanes=4),
            OperandSlot("rbits", role="value", dtype="b32"),
        ),
    ),
    # The four `.f4x2type = { .e2m1x2 };` lines. These are the table's only
    # `raw_render` entries, and the reason is one sentence per direction:
    # ISA:92 "When converting to .e2m1x2 data formats, the destination operand d
    # has .b8 type." and :101 "When converting from .e2m1x2 to .f16x2/.bf16x2,
    # source operand a has .b8 type."
    #
    # Inline asm has no 8-bit constraint letter, so the derived path can only
    # offer that operand a wider register. The ISA says that should be fine --
    # :476-480 "A source register wider than the specified type may be used,
    # except when the source operand has .bf16 or .bf16x2 format." and :481-486
    # "A destination register wider than the specified type may be used, except
    # when the destination operand has .bf16, .bf16x2 or .tf32 format." Neither
    # exception names .e2m1x2. It still does not assemble. TOOLCHAIN FACTS,
    # measured on ptxas 13.2 at -arch=sm_100a, both carrier widths in both
    # directions:
    #
    #   - destination, 16-bit ("h") and 32-bit ("r"), on
    #     cvt.rn.satfinite.e2m1x2.f32, cvt.rn.satfinite.e2m1x2.f16x2 and
    #     cvt.rn.satfinite.e2m1x2.bf16x2 -- every one of the six is
    #     "Arguments mismatch for instruction 'cvt'".
    #   - source, the same two widths, on cvt.rn.f16x2.e2m1x2 and
    #     cvt.rn.bf16x2.e2m1x2 -- the same message on all four.
    #
    # So the operand has to be a `.reg .b8` declared inside the asm block, and
    # something has to move the value between it and the "h" carrier. ptxas
    # refuses `mov` for that bridge in both directions -- `mov.b16` and
    # `mov.u16` between a `.reg .b8` and a 16-bit register are "Arguments
    # mismatch for instruction 'mov'" (same toolchain, same probe). The
    # `cvt.u8.u16` / `cvt.u16.u8` pair the deleted legacy implementation used
    # assembles, which settles the question that idiom raised: it was the
    # correct bridge, not a defect to be cleaned up. `_cvt_f4x2_raw` writes it.
    #
    # Declaration plus instruction plus bridge is three PTX statements, so all
    # four entries are listed in the single-instruction invariant's exemption
    # set. The detector that test falsifies still rejects the shape, which is
    # the point of putting the exemption in the table walk and not in it.
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f4x2type.f32 d, a, b;
        name="cvt_f4x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e2m1x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        # ISA:527 "cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32
        # is supported on following architectures:", listing sm_100a,
        # "sm_101a (Renamed to sm_110a from PTX ISA version 9.0)" and sm_120a,
        # plus family-specific targets from PTX ISA version 8.8 at :533-540.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="e2m1x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
        raw_render=_cvt_f4x2_raw,
    ),
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a;
        name="cvt_f4x2_fp16x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e2m1x2",)),
            ModifierSlot("atype", ("f16x2", "bf16x2")),
        ),
        # ISA:619-624 puts this line on "following family-specific
        # architectures": "sm_100f or higher in the same family", "sm_110f or
        # higher in the same family", "sm_120f or higher in the same family".
        # Certified at sm_100a, which ptxas 13.2 accepts here (toolchain fact),
        # the same treatment cvt_f6x2_fp16x2 gets from the same sentence.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="e2m1x2"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
        raw_render=_cvt_f4x2_raw,
    ),
    InstructionEntry(  # cvt.rn{.relu}.f16x2.f4x2type d, a;
        name="cvt_f16x2_f4x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("f16x2",)),
            ModifierSlot("atype", ("e2m1x2",)),
        ),
        # ISA:542 "cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2} is supported on
        # following architectures:", the same sm_100a / sm_101a / sm_120a list
        # `cvt_f4x2_f32` cites, plus :548-555's family-specific targets.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="f16x2"),
            OperandSlot("a", role="value", dtype="e2m1x2"),
        ),
        raw_render=_cvt_f4x2_raw,
    ),
    # cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f4x2type
    #     d, a{, scale-factor};
    InstructionEntry(
        name="cvt_bf16x2_f4x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("e2m1x2",)),
        ),
        # ISA:634-639, the same family-specific list as cvt_bf16x2_f8x2.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="e2m1x2"),
            # ISA:106-107 "For .bf16x2 destination type optional scale-factor
            # operand of type .b16 can be specified along with
            # .scaled::n2::ue8m0 qualifier." -- .b16, not .b8, so only `a`
            # needs the staging register.
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
        raw_render=_cvt_f4x2_raw,
    ),
    InstructionEntry(  # cvt.rs{.relu}.satfinite.f4x4type.f32 d, {a, b, e, f}, rbits;
        name="cvt_rs_f4x4_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rs",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("dtype", ("e2m1x4",)),
            ModifierSlot("atype", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            # The x4 fp4 destination is a full 16-bit register, so this line
            # needs no `.reg .b8` staging the way the .e2m1x2 lines do:
            # ISA:112-113 "When converting to .e2m1x4 data format, the
            # destination operand d has .b16 type."
            OperandSlot("d", role="dst", dtype="e2m1x4"),
            OperandSlot("abef", role="value", dtype="f32", lanes=4),
            OperandSlot("rbits", role="value", dtype="b32"),
        ),
    ),
    # The fp6 lines. `.f6x2type = { .e2m3x2, .e3m2x2 };` and both members ride a
    # 16-bit register in either direction -- ISA:116-117 "When converting to
    # .e2m3x2/.e3m2x2 data formats, the destination operand d has .b16 type."
    # and :127 "When converting from .e2m3x2/.e3m2x2 to .f16x2/.bf16x2, source
    # operand a has .b16 type."
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f6x2type.f32 d, a, b;
        name="cvt_f6x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e2m3x2", "e3m2x2")),
            ModifierSlot("atype", ("f32",)),
        ),
        # ISA:527 "cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32
        # is supported on following architectures:", listing sm_100a,
        # "sm_101a (Renamed to sm_110a from PTX ISA version 9.0)" and sm_120a,
        # plus family-specific targets from PTX ISA version 8.8 at :533-540.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
        ),
    ),
    InstructionEntry(  # cvt.rn.satfinite{.relu}.f6x2type.fp16x2type d, a;
        name="cvt_f6x2_fp16x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("e2m3x2", "e3m2x2")),
            ModifierSlot("atype", ("f16x2", "bf16x2")),
        ),
        # ISA:619-624 puts this line on "following family-specific
        # architectures": "sm_100f or higher in the same family", "sm_110f or
        # higher in the same family", "sm_120f or higher in the same family".
        # Certified at sm_100a, which ptxas 13.2 accepts here (toolchain fact).
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
    ),
    InstructionEntry(  # cvt.rn{.relu}.f16x2.f6x2type d, a;
        name="cvt_f16x2_f6x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("dtype", ("f16x2",)),
            ModifierSlot("atype", ("e2m3x2", "e3m2x2")),
        ),
        # ISA:542 "cvt.rn{.relu}.f16x2.{e2m1x2/e2m3x2/e3m2x2} is supported on
        # following architectures:", the same sm_100a / sm_101a / sm_120a list
        # `cvt_f6x2_f32` cites, plus :548-555's family-specific targets.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="f16x2"),
            OperandSlot("a", role="value", dtype="atype"),
        ),
    ),
    # cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f6x2type
    #     d, a{, scale-factor};
    InstructionEntry(
        name="cvt_bf16x2_f6x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("e2m3x2", "e3m2x2")),
        ),
        # ISA:634-639, the same family-specific list as cvt_bf16x2_f8x2.
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="atype"),
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
    ),
    InstructionEntry(  # cvt.rs{.relu}.satfinite.f6x4type.f32 d, {a, b, e, f}, rbits;
        name="cvt_rs_f6x4_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rs",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("dtype", ("e2m3x4", "e3m2x4")),
            ModifierSlot("atype", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="dtype"),
            OperandSlot("abef", role="value", dtype="f32", lanes=4),
            OperandSlot("rbits", role="value", dtype="b32"),
        ),
    ),
    # The .s2f6x2 lines. ISA:154 "When converting to .s2f6x2 data formats, the
    # destination operand d has .b16 type." and :169 "When converting from
    # .s2f6x2 to .bf16x2, source operand a has .b16 type."
    #
    # All three certify at sm_100a: ISA:626 "cvt with .s2f6x2 instruction type
    # is supported on following architectures:", and the list at :628-632 is
    # sm_100a, sm_103a, sm_110a, sm_120a, sm_121a.
    # cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.f32
    #     d, a, b{, scale-factor};
    InstructionEntry(
        name="cvt_s2f6x2_f32",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("s2f6x2",)),
            ModifierSlot("atype", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="s2f6x2"),
            OperandSlot("a", role="value", dtype="f32"),
            OperandSlot("b", role="value", dtype="f32"),
            # ISA:162-163 "Optional operand scale-factor has type .b16 and
            # stores two packed scaling factors of type .ue8m0."
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
    ),
    # cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2
    #     d, a{, scale-factor};
    InstructionEntry(
        name="cvt_s2f6x2_bf16x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",)),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("s2f6x2",)),
            ModifierSlot("atype", ("bf16x2",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="s2f6x2"),
            OperandSlot("a", role="value", dtype="bf16x2"),
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
    ),
    # cvt.rn{.satfinite}{.relu}{.scaled::n2::ue8m0}.bf16x2.s2f6x2
    #     d, a{, scale-factor};
    InstructionEntry(
        name="cvt_bf16x2_s2f6x2",
        mnemonic="cvt",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("scaled", ("scaled::n2::ue8m0",), optional=True),
            ModifierSlot("dtype", ("bf16x2",)),
            ModifierSlot("atype", ("s2f6x2",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="dst", dtype="bf16x2"),
            OperandSlot("a", role="value", dtype="s2f6x2"),
            OperandSlot(
                "scale_factor",
                role="value",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
    ),
    # cp.async.bulk per PTX ISA 9.7.9.26.4.1, which has eight syntax lines (four
    # copy directions x plain/`.sem.scope...type`). This entry renders exactly
    # one of them: global -> shared::cta, no optional qualifier written. The
    # other three directions, plus .L2::cache_hint, .multicast::cluster and
    # .cp_mask, are registered by the cp_async_bulk_* entries further down, whose
    # g2s_cta form also subsumes this entry's rendering. For what those entries
    # do not render, see their NOT REGISTERED note -- it is the authority, and
    # this comment deliberately makes no completeness claim of its own.
    # `{.sem}` (`.weak`) is
    # additionally blocked by the toolchain: it is PTX ISA 9.3 and ptxas 13.2
    # assembles 9.2, the same situation _check_ld already records for
    # ld.mmio.acquire.
    #
    # Mixed-space operands: dst/mbar are shared::cta (u32 carriers), src is
    # global (pointer carrier), size is a plain u32 register — each operand
    # declares its own space/dtype instead of reading the entry-level slots.
    InstructionEntry(
        name="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst_space", ("shared::cta",)),
            ModifierSlot("src_space", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
        ),
        operands=(
            OperandSlot("dst", role="addr", space="shared::cta"),
            OperandSlot("src", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared::cta"),
        ),
    ),
    # mapa per PTX ISA 9.7.9.24: map a shared address into another CTA of the
    # cluster. All four syntax lines differ only in how `a` is spelled at the
    # PTX level (register / variable / variable+imm); through a C helper the
    # operand is always a register, so they collapse to one entry.
    # `.type` fixes the width of BOTH d and a, so the two type tokens are two
    # entries: `.u64` maps a generic address (a pointer), `.u32` maps a 32-bit
    # shared-window address (a plain register, not bracketed).
    InstructionEntry(
        name="mapa",
        slots=(
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("type", ("u64",)),
        ),
        asm_volatile=False,  # a pure address computation
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="ptr"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="mapa_u32",
        mnemonic="mapa",
        slots=(
            # Not optional here, unlike the .u64 entry: the ISA says that with
            # `.space` omitted "both a and d are registers containing generic
            # addresses", and a generic address does not fit 32 bits on this
            # target. ptxas tolerates the bare `mapa.u32`, but nothing can
            # legitimately call it.
            ModifierSlot("space", ("shared::cluster",)),
            ModifierSlot("type", ("u32",)),
        ),
        asm_volatile=False,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    # clusterlaunchcontrol.try_cancel per PTX ISA 9.7.14.18.
    InstructionEntry(
        name="clusterlaunchcontrol_try_cancel",
        mnemonic="clusterlaunchcontrol",
        slots=(
            ModifierSlot("action", ("try_cancel",)),
            ModifierSlot("async_", ("async",)),
            ModifierSlot("space", ("shared::cta",), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster::all",), optional=True),
            ModifierSlot("type", ("b128",)),
        ),
        # The instruction itself needs sm_100, but `.multicast::cluster::all`
        # is only on the arch-specific targets (ISA: sm_100a / sm_101a / sm_120a).
        cert_arch="sm_100a",
        # Neither address operand takes a fixed space: `.space` is optional on
        # the syntax line, and "The .space qualifier is specified, both operands
        # addr and mbar must be in the .shared::cta state space. Otherwise,
        # generic addressing will be assumed for both." (ISA 9.7.14.18). A
        # generic address is 64-bit on sm_100, so pinning both to shared would
        # bind 32-bit registers under the space-omitted spelling. Letting
        # `operand_space` read the entry's `space` slot gives each variant the
        # carrier its own spelling promises -- the mbarrier-family rule.
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("mbar", role="addr"),
        ),
    ),
    # clusterlaunchcontrol.query_cancel per PTX ISA 9.7.14.19: decode the
    # opaque b128 response a try_cancel wrote. Three syntax shapes, so three
    # entries -- the ISA writes them as separate lines (9.7.14.19, wrapped but
    # otherwise verbatim):
    #
    #   clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
    #       pred, try_cancel_response;
    #   clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    #       {xdim, ydim, zdim, _},  try_cancel_response;
    #   clusterlaunchcontrol.query_cancel.get_first_ctaid{::dimension}.b32.b128
    #       reg, try_cancel_response;
    #
    # The `.v4` line is where the dimension-suffix-free spelling lives: "By
    # default, the instruction returns a .v4 vector whose first three elements
    # are the x, y and z coordinate of first CTA in canceled cluster." Writing
    # the third line with `{::dimension}` omitted therefore does not give a
    # single-register form -- ptxas resolves the bare name to the `.v4` shape
    # and reports "Vector of size 4 is expected for argument 0 of instruction
    # 'clusterlaunchcontrol.query_cancel'" (toolchain fact, CUDA 13.2, sm_100a).
    # So the bare form is a different operand shape, i.e. its own entry below,
    # not a fourth token on the per-dimension slot.
    #
    # NOT REGISTERED: the sink-symbol spelling of the `.v4` line's fourth
    # destination ("The contents of the 4th element are unspecified"); `_` needs
    # an operand role for "this lane is discarded". The line itself is
    # registered with four ordinary registers, which is how the ISA's own
    # example writes it -- "@p clusterlaunchcontrol.query_cancel.
    # get_first_ctaid.v4.b32.b128 {xdim, ydim, zdim, ignr}  handle;".
    InstructionEntry(
        name="clusterlaunchcontrol_query_cancel_is_canceled",
        mnemonic="clusterlaunchcontrol",
        slots=(
            ModifierSlot("action", ("query_cancel",)),
            ModifierSlot("query", ("is_canceled",)),
            ModifierSlot("ptype", ("pred",)),
            ModifierSlot("type", ("b128",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("p", role="pred_dst"),
            OperandSlot("response", role="value", dtype="b128"),
        ),
    ),
    InstructionEntry(
        # `d` is role="acc", not "dst": the caller seeds it with a sentinel and
        # predicates the instruction on the cancellation having succeeded, so a
        # false predicate has to leave the seed intact. "=" would tell nvcc the
        # prior value is dead; "+" keeps it live, and (unlike a dst) it does not
        # block the framework's `@p` wrapper.
        name="clusterlaunchcontrol_query_cancel_get_first_ctaid",
        mnemonic="clusterlaunchcontrol",
        slots=(
            ModifierSlot("action", ("query_cancel",)),
            # `::dimension` binds to the query name with no dot between them
            # (ISA: `get_first_ctaid{::dimension}`), so it is part of this
            # token rather than a slot of its own -- the same shape as
            # tcgen05's `wait::ld`.
            ModifierSlot(
                "query",
                ("get_first_ctaid::x", "get_first_ctaid::y", "get_first_ctaid::z"),
            ),
            ModifierSlot("dtype", ("b32",)),
            ModifierSlot("type", ("b128",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="acc", dtype="b32"),
            OperandSlot("response", role="value", dtype="b128"),
        ),
    ),
    InstructionEntry(
        # The `.v4` line, which is also the `{::dimension}`-omitted spelling
        # (see the family comment). `d` is one operand of four registers, not
        # four operands: PTX writes it as the brace group `{xdim, ydim, zdim,
        # ignr}`. role="acc" for the same reason as the per-dimension entry --
        # the ISA's own example predicates this instruction, and "+" is what
        # keeps the caller's prior value live under a false predicate.
        name="clusterlaunchcontrol_query_cancel_get_first_ctaid_v4",
        mnemonic="clusterlaunchcontrol",
        slots=(
            ModifierSlot("action", ("query_cancel",)),
            ModifierSlot("query", ("get_first_ctaid",)),
            ModifierSlot("vec", ("v4",)),
            ModifierSlot("dtype", ("b32",)),
            ModifierSlot("type", ("b128",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", role="acc", dtype="b32", lanes=4),
            OperandSlot("response", role="value", dtype="b128"),
        ),
    ),
    # ------------------------------------------------------------------
    # fence / membar per PTX ISA 9.7.14.4, griddepcontrol per 9.7.14.14.
    #
    # These name no address yet constrain everyone else's memory order, so they
    # set `orders_memory=True` -- see InstructionEntry for why `asm volatile`
    # alone is not enough. Each syntax line whose token sequence differs is its
    # own entry, all sharing the `fence` mnemonic, so the surface stays
    # `T.ptxd.fence...` and the chain narrows on the tokens themselves.
    # NOT REGISTERED: the `.sync_restrict` lines and the fabric-proxy line
    # (fixed token sequences with no users yet), and the deprecated `membar`
    # spellings, which the ISA itself marks as the old style for `fence`.
    InstructionEntry(  # fence{.sem}.scope;
        name="fence",
        slots=(
            ModifierSlot("sem", ("sc", "acq_rel", "acquire", "release"), optional=True),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.mbarrier_init.release.cluster;
        name="fence_mbarrier_init",
        mnemonic="fence",
        slots=(
            ModifierSlot("op_restrict", ("mbarrier_init",)),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("cluster",)),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.proxykind;
        name="fence_proxy",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("alias", "async")),
            ModifierSlot("space", ("global", "shared::cta", "shared::cluster"), optional=True),
        ),
        check=_check_fence_proxy,
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.tensormap::generic.release.scope;
        name="fence_proxy_tensormap_release",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("tensormap::generic",)),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.tensormap::generic.acquire.scope [addr], 128;
        name="fence_proxy_tensormap_acquire",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("tensormap::generic",)),
            ModifierSlot("sem", ("acquire",)),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("addr", role="addr"),
            # "The only supported value for the size operand is 128, which must
            # be a constant integer literal" -- ISA 9.7.14.4.
            OperandSlot("size", role="imm", literal="128"),
        ),
    ),
    InstructionEntry(  # griddepcontrol.action;
        name="griddepcontrol",
        slots=(ModifierSlot("action", ("launch_dependents", "wait")),),
        orders_memory=True,
        operands=(),
    ),
    # ------------------------------------------------------------------
    # bar / barrier per PTX ISA 9.7.14.1, bar.warp.sync per 9.7.14.2,
    # barrier.cluster per 9.7.14.3.
    #
    #   barrier{.cta}.sync{.aligned}   a{, b};    bar{.cta}.sync   a{, b};
    #   barrier{.cta}.arrive{.aligned} a,  b;     bar{.cta}.arrive a,  b;
    #   bar.warp.sync      membermask;
    #
    # The optional thread count is its own syntax line, so `sync` is two
    # entries sharing a mnemonic, told apart by arity (as `mov`'s shapes are;
    # `pred` is keyword-only, so arity is unambiguous).
    # NOT REGISTERED: the `.red.popc` / `.red.op` lines, which take a `{!}c`
    # predicate source and produce a `.pred` result.
    *[
        InstructionEntry(  # bar{.cta}.sync a;  /  barrier{.cta}.sync{.aligned} a;
            name=f"{mnem}_sync",
            mnemonic=mnem,
            slots=(
                ModifierSlot("cta", ("cta",), optional=True),
                ModifierSlot("action", ("sync",)),
            )
            + (
                (ModifierSlot("aligned", ("aligned",), optional=True),) if mnem == "barrier" else ()
            ),
            orders_memory=True,
            operands=(OperandSlot("a", role="value", dtype="u32"),),
        )
        for mnem in ("bar", "barrier")
    ],
    InstructionEntry(
        name="bar_sync_count",
        mnemonic="bar",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("sync",)),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="bar_arrive",
        mnemonic="bar",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("arrive",)),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # bar.warp.sync      membermask;
        # The `bar` mnemonic's warp-level line, ISA 9.7.14.2 -- a different
        # section from the CTA barriers above and a different operand: not a
        # barrier id but a lane mask. "Operand membermask specifies a 32-bit
        # integer which is a mask indicating threads participating in barrier
        # where the bit position corresponds to thread's laneid."
        #
        # `warp` is the token that tells the two sections apart, the way
        # `cluster` does for barrier.cluster below.
        #
        # orders_memory, on the section's own words: "bar.warp.sync also
        # guarantee memory ordering among threads participating in barrier.
        # Thus, threads within warp that wish to communicate via memory can
        # store to memory, execute bar.warp.sync, and then safely read values
        # stored by other threads in warp." Without the clobber nvcc may sink
        # the stores below it or hoist the loads above it, and that sentence is
        # exactly the guarantee callers reach for.
        name="bar_warp_sync",
        mnemonic="bar",
        slots=(
            ModifierSlot("warp", ("warp",)),
            ModifierSlot("action", ("sync",)),
        ),
        orders_memory=True,
        operands=(OperandSlot("membermask", role="value", dtype="u32"),),
    ),
    InstructionEntry(
        name="barrier_sync_count",
        mnemonic="barrier",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("sync",)),
            ModifierSlot("aligned", ("aligned",), optional=True),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="barrier_arrive",
        mnemonic="barrier",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("aligned", ("aligned",), optional=True),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    *[
        InstructionEntry(  # barrier.cluster.arrive{.sem}{.aligned} / .wait{.acquire}{.aligned}
            name=f"barrier_cluster_{act}",
            # Shares the `barrier` mnemonic with 9.7.14.1 so the surface reads
            # `T.ptxd.barrier.cluster.arrive(...)`; `cluster` is the token that
            # tells the two ISA sections apart.
            mnemonic="barrier",
            slots=(
                ModifierSlot("cluster", ("cluster",)),
                ModifierSlot("action", (act,)),
                ModifierSlot(
                    "sem",
                    ("release", "relaxed") if act == "arrive" else ("acquire",),
                    optional=True,
                ),
                ModifierSlot("aligned", ("aligned",), optional=True),
            ),
            orders_memory=True,
            operands=(),
        )
        for act in ("arrive", "wait")
    ],
    # ------------------------------------------------------------------
    # tcgen05 memory-allocation and synchronisation, per PTX ISA 9.7.17:
    # alloc / dealloc / relinquish_alloc_permit 9.7.17.7.1, wait 9.7.17.8.5,
    # fence 9.7.17.11.1, commit 9.7.17.12.1.
    #
    # Every one of these carries `orders_memory=True`: the legacy helpers all
    # had `::: "memory"`, and the waits/fences name no address at all.
    #
    # NOT REGISTERED:
    # - `tcgen05.shift.cta_group.down [taddr]`: no call site. (The tmem address
    #   space its bracketed operand needs now exists -- `tcgen05.ld`/`.st`/`.cp`
    #   all use it -- so this one is registrable the day something calls it.)
    InstructionEntry(  # tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols;
        name="tcgen05_alloc",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("alloc",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("space", ("shared::cta",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            OperandSlot("dst", role="addr", space="shared::cta"),
            OperandSlot("ncols", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols;
        name="tcgen05_dealloc",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("dealloc",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            # `taddr` is a plain register operand here, not bracketed.
            OperandSlot("taddr", role="value", dtype="u32"),
            OperandSlot("ncols", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
        name="tcgen05_relinquish_alloc_permit",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("relinquish_alloc_permit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # tcgen05.wait::{ld,st}.sync.aligned;
        name="tcgen05_wait",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("wait::ld", "wait::st")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # tcgen05.fence::{before,after}_thread_sync;
        name="tcgen05_fence",
        mnemonic="tcgen05",
        slots=(ModifierSlot("action", ("fence::before_thread_sync", "fence::after_thread_sync")),),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    # tcgen05.commit.cta_group.mbarrier::arrive::one{.shared::cluster}.b64 [mbar];
    InstructionEntry(
        name="tcgen05_commit",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("commit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("completion", ("mbarrier::arrive::one",)),
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(OperandSlot("mbar", role="addr", space="shared::cluster"),),
    ),
    # tcgen05.commit...{.shared::cluster}.multicast::cluster.b64 [mbar], ctaMask;
    # The multicast form: `pred` is keyword-only, so the trailing mask
    # dispatches by arity against the unicast entry. The mask is `.b16`
    # (the legacy helper bound it "h").
    InstructionEntry(
        name="tcgen05_commit_multicast",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("commit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("completion", ("mbarrier::arrive::one",)),
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("multicast", ("multicast::cluster",)),
            ModifierSlot("type", ("b64",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            OperandSlot("mbar", role="addr", space="shared::cluster"),
            OperandSlot("mask", role="value", dtype="u16"),
        ),
    ),
    # cp.async completion tracking: cp.async.mbarrier.arrive per PTX ISA
    # 9.7.14.16.18, cp.async.commit_group per 9.7.9.26.3.2, cp.async.wait_group
    # and cp.async.wait_all per 9.7.9.26.3.3, cp.async.bulk.commit_group /
    # .wait_group per 9.7.9.26.6.1 / 9.7.9.26.6.2.
    #
    # The wait_group counts are caller-chosen immediates: the ISA gives N no
    # register form, so each value is its own helper, and the closed `choices`
    # set is what makes every one of them certifiable. 0..7 covers every call
    # site (pipeline depths); widen the tuple if a deeper pipeline appears.
    #
    # NOT REGISTERED: the `cp.async` ca/cg copy lines, which carry an optional
    # ignore-src operand.
    InstructionEntry(  # cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];
        name="cp_async_mbarrier_arrive",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("target", ("mbarrier",)),
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("noinc", ("noinc",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        # No fixed space on addr, for the reason the mbarrier family states at
        # length below: "If no state space is specified then Generic Addressing
        # is used." (ISA 9.7.14.16.18), and a generic address is 64-bit on
        # sm_90+, so pinning the operand to shared would bind a 32-bit register
        # under the space-omitted spelling. `operand_space` reads the entry's
        # `space` slot instead, so the carrier follows the spelling.
        operands=(OperandSlot("addr", role="addr"),),
    ),
    InstructionEntry(  # cp.async.commit_group;
        name="cp_async_commit_group",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("action", ("commit_group",)),
        ),
        operands=(),
    ),
    InstructionEntry(  # cp.async.wait_all ;
        # The ISA's second syntax line of 9.7.9.26.3.3, and the one with no
        # operand: "cp.async.wait_all is equivalent to :
        #
        #     cp.async.commit_group;
        #     cp.async.wait_group 0;"
        #
        # Registered as its own instruction rather than left to that pair --
        # PTX spells it, ptxas emits it, and the source stays the text it means.
        #
        # orders_memory, like every wait entry here: it names no address, so
        # the "memory" clobber cannot be derived from an operand, and without
        # it nvcc may hoist the loads of the copied data above the wait.
        # "Writes performed by cp.async operations are made visible to the
        # executing thread only after: The completion of cp.async.wait_all"
        # (9.7.9.26.3.3) is exactly what that clobber has to protect.
        name="cp_async_wait_all",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("action", ("wait_all",)),
        ),
        orders_memory=True,
        operands=(),
    ),
    *[
        InstructionEntry(  # cp.async{.bulk}.wait_group{.read} N;
            name=f"cp_async{'_bulk' if bulk else ''}_wait_group",
            mnemonic="cp",
            slots=(
                ModifierSlot("api", ("async",)),
                *((ModifierSlot("kind", ("bulk",)),) if bulk else ()),
                ModifierSlot("action", ("wait_group",)),
                *((ModifierSlot("read", ("read",), optional=True),) if bulk else ()),
            ),
            orders_memory=True,
            operands=(OperandSlot("group", role="imm", choices=tuple(str(n) for n in range(8))),),
        )
        for bulk in (False, True)
    ],
    # setmaxnreg per PTX ISA 9.7.20.5: the register count is an immediate the
    # ISA bounds to [24, 256] in steps of 8 -- exactly a closed choices set.
    InstructionEntry(
        name="setmaxnreg",
        slots=(
            ModifierSlot("action", ("inc", "dec")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("type", ("u32",)),
        ),
        cert_arch="sm_90a",
        orders_memory=True,
        operands=(
            OperandSlot("nreg", role="imm", choices=tuple(str(n) for n in range(24, 257, 8))),
        ),
    ),
    # wgmma.wait_group per PTX ISA 9.7.16.7.3, same caller-immediate shape.
    #
    # The ISA's domain for N is open: the whole section says only "Operand N is
    # an integer constant." (9.7.16.7.3:14), with no upper bound anywhere --
    # unlike setmaxnreg above, whose [24, 256] the ISA closes itself. The 0..7
    # below is therefore this table's closure, not an ISA rule: `choices` needs
    # a finite set to enumerate and certify, and every variant of a registered
    # entry must assemble.
    #
    # NOT REGISTERED: N >= 8. ptxas accepts them (measured at sm_90a up to
    # 2**32, every value assembling; only a negative N is refused, "Argument 0
    # of instruction 'wgmma.wait_group': unexpected value '-1', expected to be
    # non-negative integer"), but they are not distinct instructions: SASS shows
    # N in 0..7 lowering to `WARPGROUP.DEPBAR.LE gsb0, 0xN` and every N >= 8
    # lowering to `WARPGROUP.DEPBAR.LE gsb0, 0x7`, i.e. clamped to the same
    # instruction 7 already renders. Widen this if a call site ever wants the
    # wider spelling for source-level clarity.
    InstructionEntry(
        name="wgmma_wait_group",
        mnemonic="wgmma",
        slots=(
            ModifierSlot("action", ("wait_group",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_90a",
        orders_memory=True,
        operands=(OperandSlot("group", role="imm", choices=tuple(str(n) for n in range(8))),),
    ),
    InstructionEntry(  # cp.async.bulk.commit_group;
        name="cp_async_bulk_commit_group",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("action", ("commit_group",)),
        ),
        operands=(),
    ),
    # cp.async per PTX ISA 9.7.9.26.3.1: the non-bulk asynchronous copy.
    # cp-size is an integer constant the ISA closes to {4, 8, 16} (and to 16
    # alone under .cg) -- a choices immediate. The src-size arity zero-fills
    # the destination tail; it is a separate entry told apart by arity, the
    # mbarrier.arrive precedent.
    #
    # NOT REGISTERED: the `{, ignore-src}` lines. No call site ever used
    # them (the legacy intrinsics for those forms were unreachable from the
    # dispatcher), and their arity collides with the src-size lines -- the
    # operand-shape dispatch could not tell a src-size u32 from an
    # ignore-src predicate.
    *[
        InstructionEntry(
            name=f"cp_async_{cop}{'_src_size' if src_size else ''}",
            mnemonic="cp",
            slots=(
                ModifierSlot("api", ("async",)),
                ModifierSlot("cop", (cop,)),
                ModifierSlot("dst", ("shared", "shared::cta")),
                ModifierSlot("src", ("global",)),
                ModifierSlot("cache", ("L2::cache_hint",), optional=True),
                ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ),
            cert_arch="sm_90",
            operands=(
                OperandSlot("dst_mem", role="addr", space="shared"),
                OperandSlot("src_mem", role="addr", space="global"),
                OperandSlot(
                    "cp_size", role="imm", choices=("4", "8", "16") if cop == "ca" else ("16",)
                ),
                *((OperandSlot("src_size", role="value", dtype="u32"),) if src_size else ()),
                OperandSlot(
                    "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
                ),
            ),
        )
        for cop in ("ca", "cg")
        for src_size in (False, True)
    ],
    # cp.async.bulk (non-tensor), per PTX ISA 9.7.9.26.4.1: four directions.
    #
    # NOT REGISTERED:
    # - the `.sem.scope`/`.type` lines: "Support for .weak and .relaxed
    #   semantics, .scope and .type qualifiers are introduced in PTX ISA version
    #   9.3", which ptxas 13.2 in this toolchain (9.2) cannot assemble, and no
    #   call site uses them.
    #
    # `.ignore_oob` is registered, on the one entry below whose direction the
    # ISA gives it: "The qualifier .ignore_oob is only available for the global
    # to .shared::cta copy direction." (9.7.9.26.4.1). It is a PTX ISA 9.2
    # feature -- "Support for .ignore_oob qualifier introduced in PTX ISA
    # version 9.2." -- so unlike the .sem.scope/.type lines above it is inside
    # what this toolchain assembles; ptxas takes it at sm_90 and sm_100.
    InstructionEntry(  # global -> shared::cta
        # The `{.ignore_oob}` position and its two operands, from the syntax
        # line (9.7.9.26.4.1), wrapped but otherwise verbatim:
        #
        #   cp.async.bulk{.sem}.dst.src.completion_mechanism{.level::cache_hint}
        #      {.ignore_oob}
        #      [dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight},
        #      [mbar] {, cache_policy};
        #
        # The two operands sit between `size` and `[mbar]`, and exist exactly
        # when the qualifier is written -- the same lanes-0-or-1 shape
        # `cache_policy` uses for its own `{, cache_policy}` brace. "The 32-bit
        # operands ignoreBytesLeft and ignoreBytesRight are used to specify the
        # bytes from beginning and ending of the copy-chunk specified by size
        # that may go out of bounds. The only valid values for ignoreBytesLeft
        # and ignoreBytesRight are [0..15]" -- that range is a value bound, not
        # a spelling, so it stays a runtime u32 rather than a choices immediate.
        name="cp_async_bulk_g2s_cta",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cta",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("ignore_oob", ("ignore_oob",), optional=True),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cta"),
            OperandSlot("src_mem", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot(
                "ignore_bytes_left",
                role="value",
                dtype="u32",
                lanes=_ignore_oob_lanes,
                vector=False,
            ),
            OperandSlot(
                "ignore_bytes_right",
                role="value",
                dtype="u32",
                lanes=_ignore_oob_lanes,
                vector=False,
            ),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> shared::cluster
        name="cp_async_bulk_g2s_cluster",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster",), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("src_mem", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot("cta_mask", role="value", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # shared::cta -> shared::cluster (peer-CTA push)
        name="cp_async_bulk_s2c",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared"),
        ),
    ),
    InstructionEntry(  # shared::cta -> global
        name="cp_async_bulk_s2g",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            # .cp_mask masks bytes within each 16-byte chunk: "The i-th bit in
            # the 16-bit wide byteMask operand specifies whether the i-th byte
            # of each 16-byte wide chunk of source data is copied to the
            # destination." (ISA 9.7.9.26.4.1:181-182). It is independent of
            # .L2::cache_hint -- the syntax line brace-marks the two separately
            # ("cp.async.bulk{.sem}.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}",
            # :72) and the section's only couplings are ":173 When the optional
            # argument cache_policy is specified, the qualifier
            # .level::cache_hint is required." and ":180 When the optional
            # qualifier .cp_mask is specified, the argument byteMask is
            # required." -- both of which this entry's operand `lanes` already
            # encode. This entry used to reject bare .cp_mask on the claim that
            # ptxas required the pairing; ptxas (CUDA 13.2) assembles
            # `cp.async.bulk.global.shared::cta.bulk_group.cp_mask [%rd], [%r1],
            # %r2, %h;` at sm_100 and sm_100a, so the claim was false.
            # .cp_mask is a Blackwell feature -- below sm_100 ptxas reports
            # "Feature '.cp_mask' requires .target sm_100 or higher", which the
            # entry's cert_arch already covers.
            ModifierSlot("cp_mask", ("cp_mask",), optional=True),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("dst_mem", role="addr", space="global"),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
            # "the 16-bit wide byteMask operand" -- the legacy helper bound it
            # "r", but that form was unreachable and ptxas rejects the 32-bit
            # register here.
            OperandSlot("byte_mask", role="value", dtype="u16", lanes=_cp_mask_lanes, vector=False),
        ),
    ),
    # cp.async.bulk.tensor (TMA), per ISA 9.7.9.26.5.2-4. The tensor address
    # is the composite `[tensorMap, tensorCoords]` -- one PTX operand holding
    # a 64-bit tensor-map pointer plus an .s32 coordinate vector -- which is
    # what OperandSlot.bracket transcribes. The coordinate count follows .dim
    # except in the gather4/scatter4 modes (fixed 5); ctaMask and cache_policy
    # are trailing operands that exist exactly when their modifier is written,
    # a zero-lanes function each.
    #
    # NOT REGISTERED:
    # - the .im2col family of load modes (im2col/im2col::w/im2col::w::128 and
    #   s2g's im2col_no_offs) with their `{, im2colInfo}` operand: no call
    #   site uses im2col, and the legacy helpers never supported it.
    # - `.tile::scatter4` under cp.reduce: absent from that syntax line.
    InstructionEntry(  # global -> shared::cluster (the TMA load every kernel uses)
        name="cp_async_bulk_tensor_g2s_cluster",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("global",)),
            # ptxas accepts both spellings of the default load mode: written
            # `.tile` or omitted entirely (the legacy g2s helpers omitted it,
            # the s2g/prefetch ones wrote it).
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster",), optional=True),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot("cta_mask", role="value", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> shared::cta (no multicast operand)
        name="cp_async_bulk_tensor_g2s_cta",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("shared::cta",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cta"),
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # shared::cta -> global
        name="cp_async_bulk_tensor_s2g",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("load_mode", ("tile", "tile::scatter4"), optional=True),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="dst"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"
            ),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> L2 prefetch
        name="cp_async_bulk_tensor_prefetch",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("action", ("prefetch",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("level", ("L2",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # cp.reduce.async.bulk.tensor: shared::cta -> global, in place
        name="cp_reduce_async_bulk_tensor",
        mnemonic="cp",
        slots=(
            ModifierSlot("op", ("reduce",)),
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("redop", ("add", "min", "max", "inc", "dec", "and", "or", "xor")),
            ModifierSlot("load_mode", ("tile",), optional=True),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="dst"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"
            ),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # mbarrier, per the PTX ISA 9.7.14.16 chapter: init 9.7.14.16.12, inval
    # 9.7.14.16.13, expect_tx 9.7.14.16.14, complete_tx 9.7.14.16.15, arrive
    # 9.7.14.16.16, arrive_drop 9.7.14.16.17, test_wait / try_wait 9.7.14.16.19,
    # pending_count 9.7.14.16.20.
    #
    # The arrive lines come in a `state`-returning form and a sink form
    # (`_, [addr]`); the sink is what a void helper can express, so `_` is an
    # ISA-fixed immediate the way `st.bulk`'s initval is. That also leaves the
    # instruction without a destination, so `pred=` works. arrive_drop's five
    # syntax lines (9.7.14.16.17) are the same five shapes, so they are
    # registered as the same four entries with the action token swapped.
    #
    # NOT REGISTERED:
    # - the `state, [addr]` forms: the destination is the barrier's pre-arrival
    #   state, which nothing reads today. (`arrive.noComplete` has no sink form
    #   at all, so it is registered below with a real state result.)
    # - the non-parity try_wait / test_wait lines (their `state` operand is the
    #   arrive-returned token nothing reads today), the `.phase_type` qualifiers
    #   (no call site), and try_wait's timeHint-less arity (every caller passes
    #   the tick budget).
    # - the mbarrier layout facility, all of it: `mbarrier.init{.layout}`
    #   (9.7.14.16.12), `mbarrier.pending_count{.layout}` (9.7.14.16.20) and
    #   the whole `mbarrier.check_layout.layout{.ss}.b64 p, [addr];`
    #   instruction (9.7.14.16.21). Every one is a PTX ISA 9.3 feature --
    #   "Support for .layout qualifier introduced in PTX ISA version 9.3."
    #   (9.7.14.16.12, 9.7.14.16.20) and "Introduced in PTX ISA version 9.3."
    #   (9.7.14.16.21) -- and ptxas 13.2 in this toolchain tops out at PTX ISA
    #   9.2, so it cannot assemble them at any -arch: measured, sm_90 through
    #   sm_120a, "Unknown modifier '.layout::v1'" and "Not a name of any known
    #   instruction: 'mbarrier.check_layout'". Same reason the cp.async.bulk
    #   `.sem.scope`/`.type` lines above are unregistered. Register them when
    #   the toolchain moves to PTX ISA 9.3; the bare `mbarrier.pending_count`
    #   line, which predates the qualifier, is registered below.
    #
    # The addr slot takes no fixed space: with `.space` omitted the ISA means
    # a generic address, which is 64-bit on sm_90+, so pinning the operand to
    # shared would bind a 32-bit register there and ptxas rejects the 32-bit
    # ABI. Letting `operand_space` read the modifier picks the right carrier
    # for each variant, exactly as ld/st do.
    InstructionEntry(  # mbarrier.init{.shared{::cta}}.b64 [addr], count;
        name="mbarrier_init",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("init",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("count", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # mbarrier.inval{.shared{::cta}}.b64 [addr];
        name="mbarrier_inval",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("inval",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(OperandSlot("addr", role="addr"),),
    ),
    # mbarrier.arrive (9.7.14.16.16) and mbarrier.arrive_drop (9.7.14.16.17)
    # are the same five syntax lines under two action tokens -- compare
    #
    #   mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64           state, [addr]{, count};
    #   mbarrier.arrive{.sem.scope}{.shared::cluster}.b64         _, [addr] {,count}
    #   mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
    #   mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}.b64   _, [addr], txCount;
    #   mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64  state, [addr], count;
    #
    # with 9.7.14.16.17's five, which differ only in the mnemonic's action and
    # in arrive_drop also decrementing the expected count ("Decrements the
    # expected arrival count of the mbarrier object by the value specified by
    # the 32-bit integer operand count"). So each shape below is one
    # comprehension over the two tokens rather than two hand-copied entries.
    #
    # Both sections state the pairing rule `_check_mbarrier_sem_scope` enforces
    # in the same words: "Qualifiers .sem and .scope must be specified
    # together." (9.7.14.16.16, 9.7.14.16.17).
    *[
        InstructionEntry(  # mbarrier.<act>{.sem.scope}{.space}.b64 _, [addr];
            # The `{, count}` optionality is one ISA line; here it is two entries
            # told apart by arity, and the ISA defines the omitted count as 1.
            name=f"mbarrier_{act}_nocount",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sem", ("release", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("state", role="imm", literal="_"),
                OperandSlot("addr", role="addr"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    *[
        InstructionEntry(  # mbarrier.<act>{.sem.scope}{.space}.b64 _, [addr], count;
            name=f"mbarrier_{act}",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sem", ("release", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("state", role="imm", literal="_"),
                OperandSlot("addr", role="addr"),
                OperandSlot("count", role="value", dtype="u32"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    *[
        InstructionEntry(  # mbarrier.<act>.expect_tx{.sem.scope}{.space}.b64 _, [addr], txCount;
            name=f"mbarrier_{act}_expect_tx",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("expect_tx", ("expect_tx",)),
                ModifierSlot("sem", ("release", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("state", role="imm", literal="_"),
                OperandSlot("addr", role="addr"),
                OperandSlot("tx_count", role="value", dtype="u32"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    # mbarrier.<act>.noComplete{.release.cta}{.shared{::cta}}.b64 state, [addr], count;
    *[
        InstructionEntry(
            # This line has no sink form, so `state` is a real register result.
            # It rides role="acc" rather than dst: "+" keeps the old value live
            # under a false predicate, which is what lets `pred=` remain legal on
            # an instruction that writes a register. Its qualifier pair is fixed
            # (.release.cta only) and the space domain has no ::cluster.
            name=f"mbarrier_{act}_no_complete",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("nocomplete", ("noComplete",)),
                ModifierSlot("sem", ("release",), optional=True),
                ModifierSlot("scope", ("cta",), optional=True),
                ModifierSlot("space", ("shared", "shared::cta"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                # Pinned u64, not b64: the bit-type dtype axis would offer an f64
                # carrier, and ptxas rejects an .f64 register as the state operand
                # ("Arguments mismatch for instruction 'mbarrier.arrive'").
                OperandSlot("state", role="acc", dtype="u64"),
                OperandSlot("addr", role="addr"),
                OperandSlot("count", role="value", dtype="u32"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    InstructionEntry(  # mbarrier.pending_count.b64 count, state;
        # The reader of the `state` result the two `.noComplete` entries above
        # produce: "The state operand is a 64-bit register that must be the
        # result of a prior mbarrier.arrive.noComplete or
        # mbarrier.arrive_drop.noComplete instruction." (ISA 9.7.14.16.20).
        #
        # No address and no state space -- the instruction reads a register,
        # not the mbarrier object -- so there is no `space` slot here and
        # `count` is a plain destination: "The destination register count is a
        # 32-bit unsigned integer representing the pending count of the
        # mbarrier object prior to the arrive-on operation from which the state
        # register was obtained."
        #
        # `state` is pinned u64 for the reason its producer is: the bit-type
        # dtype axis would otherwise offer an .f64 carrier the instruction has
        # no use for. The `{.layout}` position is unregistered -- see the
        # region note above.
        name="mbarrier_pending_count",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("pending_count",)),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(
            OperandSlot("count", role="dst", dtype="u32"),
            OperandSlot("state", role="value", dtype="u64"),
        ),
    ),
    # mbarrier.test_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity;
    # mbarrier.try_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity, timeHint;
    #
    # waitComplete is a `.pred` result -- role="pred_dst", the in-block selp
    # materialization. try_wait is registered in its timeHint arity only (the
    # hint is a nanosecond budget the callers always pass).
    *[
        InstructionEntry(
            name=f"mbarrier_{act}_parity",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("parity", ("parity",)),
                ModifierSlot("sem", ("acquire", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("wait_complete", role="pred_dst"),
                OperandSlot("addr", role="addr"),
                OperandSlot("phase", role="value", dtype="u32"),
                *(
                    (OperandSlot("time_hint", role="value", dtype="u32"),)
                    if act == "try_wait"
                    else ()
                ),
            ),
        )
        for act in ("test_wait", "try_wait")
    ],
    *[
        InstructionEntry(  # mbarrier.{expect_tx,complete_tx}{.sem.scope}{.space}.b64 [addr], tx;
            name=f"mbarrier_{act}",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sem", ("relaxed",), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("addr", role="addr"),
                OperandSlot("tx_count", role="value", dtype="u32"),
            ),
        )
        for act in ("expect_tx", "complete_tx")
    ],
    # wgmma group synchronisation, per PTX ISA 9.7.16.7: fence 9.7.16.7.1,
    # commit_group 9.7.16.7.2, wait_group 9.7.16.7.3. (wait_group's textual
    # group count is the `choices` immediate registered above; the mma_async
    # lines are the acc-role entries further down.)
    *[
        InstructionEntry(
            name=f"wgmma_{act}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
            ),
            cert_arch="sm_90a",
            orders_memory=True,
            operands=(),
        )
        for act in ("fence", "commit_group")
    ],
    # ldmatrix per PTX ISA 9.7.15.5.15 -- warp-level matrix load. Three syntax
    # lines; the destination is a brace-enclosed vector of 1/2/4 32-bit
    # registers "as per the value of .num", which is what a callable `lanes`
    # transcribes. The first line's two shapes split into two entries because
    # their target floors differ (m8n8 is sm_75+; m16n16/.b8 are sm_100a) and
    # `cert_arch` is per entry.
    #
    #   ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type            r, [p];
    #   ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
    #   ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];
    #
    # NOT REGISTERED: nothing -- every syntax line of the family is here.
    InstructionEntry(  # line 1, .m8n8.b16: "Each matrix element holds 16-bit data"
        name="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b16",)),
        ),
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    InstructionEntry(  # line 1, .m16n16.b8: "only .x1 and .x2 are valid"
        name="ldmatrix_m16n16_b8",
        mnemonic="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m16n16",)),
            ModifierSlot("num", ("x1", "x2")),
            # The syntax line spells {.trans} optional, but ptxas requires it
            # whenever the shape is .m16n16 ("Modifier '.trans' require for
            # instruction ldmatrix with shape '.m16n16'") -- the 16x16 8-bit
            # load only exists in the transposed layout.
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b8",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    InstructionEntry(  # lines 2+3: the 6/4-bit decompression loads
        name="ldmatrix_b8fmt",
        mnemonic="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n16", "m16n16")),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("dst_fmt", ("b8x16",)),
            ModifierSlot("src_fmt", ("b6x16_p32", "b4x16_p64")),
        ),
        cert_arch="sm_100a",
        check=_check_ldmatrix_b8fmt,
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    # stmatrix per PTX ISA 9.7.15.5.16 -- the store mirror of ldmatrix. One
    # syntax line; the two shapes split into two entries because their type and
    # target floors differ.
    #
    #   stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;
    #
    # Note the operand order is the reverse of ldmatrix's: the address comes
    # first, the register group second.
    #
    # NOT REGISTERED: nothing -- every syntax line of the family is here.
    InstructionEntry(  # .m8n8.b16
        name="stmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b16",)),
        ),
        cert_arch="sm_90",  # ISA: "Requires sm_90 or higher."
        operands=(
            OperandSlot("p", role="addr"),
            OperandSlot("r", role="value", dtype="b32", lanes=_matrix_num_lanes),
        ),
    ),
    InstructionEntry(  # .m16n8.b8: ".m16n8 shape is valid only for .b8 type"
        name="stmatrix_m16n8_b8",
        mnemonic="stmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m16n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            # ISA: "for 16x8 matrices, .trans is mandatory".
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b8",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("p", role="addr"),
            OperandSlot("r", role="value", dtype="b32", lanes=_matrix_num_lanes),
        ),
    ),
    # tcgen05.ld / .st per PTX ISA 9.7.17.8.3 / 9.7.17.8.4. The register vector
    # length is `.num` scaled by the shape width, which is what a callable
    # `lanes` transcribes; `taddr` is a tmem address, a packed 32-bit
    # (row << 16 | col) value rather than a pointer.
    #
    #   tcgen05.ld.sync.aligned.shape.num{.pack::16b}.b32   r, [taddr];
    #   tcgen05.st.sync.aligned.shape.num{.unpack::16b}.b32 [taddr], r;
    #
    # Note the operand orders mirror each other.
    #
    # The `.16x32bx2` shape is the entry pair below: its syntax line carries an
    # extra `immHalfSplitoff` operand, which is a different operand shape.
    #
    # NOT REGISTERED:
    # - `tcgen05.ld.red`, which is sm_101a-only (not sm_100a, so it cannot be
    #   certified here) and has no call sites.
    InstructionEntry(
        name="tcgen05_ld",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("ld",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x64b", "16x128b", "16x256b", "32x32b")),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("pack", ("pack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        check=_check_tcgen05_ldst,
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_tcgen05_ldst_lanes),
            OperandSlot("taddr", role="addr", space="tmem"),
        ),
    ),
    InstructionEntry(
        name="tcgen05_st",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("st",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x64b", "16x128b", "16x256b", "32x32b")),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("unpack", ("unpack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        check=_check_tcgen05_ldst,
        operands=(
            OperandSlot("taddr", role="addr", space="tmem"),
            OperandSlot("r", role="value", dtype="b32", lanes=_tcgen05_ldst_lanes),
        ),
    ),
    # The `.16x32bx2` lines, ISA 9.7.17.8.3:11 / 9.7.17.8.4:9 --
    #
    #   tcgen05.ld.sync.aligned.16x32bx2.num{.pack}.b32   r, [taddr], immHalfSplitoff;
    #   tcgen05.st.sync.aligned.16x32bx2.num{.unpack}.b32 [taddr], immHalfSplitoff, r;
    #
    # One instruction, two half-accesses: "The base address of the first access
    # is specified by taddr and the base address of the second access is
    # specified by taddr+immHalfSplitoff, where immHalfSplitoff is an immediate
    # argument." (:49-51). The immediate is an OPEN imm operand: the ISA gives
    # it no value domain, so the table declares none and the caller passes any
    # compile-time constant -- certification proves the shape at sampled
    # values, not the caller's value.
    InstructionEntry(
        name="tcgen05_ld_split",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("ld",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x32bx2",)),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("pack", ("pack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_tcgen05_ldst_lanes),
            OperandSlot("taddr", role="addr", space="tmem"),
            OperandSlot("imm_half_splitoff", role="imm"),
        ),
    ),
    InstructionEntry(
        name="tcgen05_st_split",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("st",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x32bx2",)),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("unpack", ("unpack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("taddr", role="addr", space="tmem"),
            OperandSlot("imm_half_splitoff", role="imm"),
            OperandSlot("r", role="value", dtype="b32", lanes=_tcgen05_ldst_lanes),
        ),
    ),
    # tcgen05.cp per PTX ISA 9.7.17.9.2: an async shared -> tmem copy of one
    # shape, optionally multicast across the warps of a warpgroup and
    # optionally decompressing fp4/fp6 into fp8 on the way. `s-desc` is the
    # same 64-bit shared-memory matrix descriptor tcgen05.mma takes.
    #
    # The ISA states the decompression qualifiers as a pair (".dst_fmt and
    # .src_fmt"), so they are two slots that appear together, not the single
    # glued token ("b8x16.b6x16_p32") the legacy helper spelled.
    InstructionEntry(  # tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
        name="tcgen05_cp",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("cp",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("shape", ("128x256b", "4x256b", "128x128b", "64x128b", "32x128b")),
            ModifierSlot("multicast", ("warpx2::02_13", "warpx2::01_23", "warpx4"), optional=True),
            ModifierSlot("dst_fmt", ("b8x16",), optional=True),
            ModifierSlot("src_fmt", ("b6x16_p32", "b4x16_p64"), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tcgen05_cp,
        operands=(
            OperandSlot("taddr", role="addr", space="tmem"),
            OperandSlot("s_desc", role="value", dtype="b64"),
        ),
    ),
    # tcgen05.mma per PTX ISA 9.7.17.10.9.1 (sm_100a): D = A*B + D where D
    # lives in Tensor Memory (an address, not registers -- so the instruction
    # has no dst and @p stays available). Two entries per family split on the
    # A operand's home: a 64-bit shared-memory descriptor ("ss") or a tmem
    # address ("ts"), the same split the syntax lines draw. enable-input-d is
    # a runtime .pred argument -- role="pred_src", the in-block setp
    # conversion -- and disable-output-lane is a register vector whose length
    # follows .cta_group (4 or 8).
    #
    # NOT REGISTERED:
    # - `{, scale-input-d}`: an instruction-text immediate in [0, 15], no
    #   call site uses it.
    # - `tcgen05.mma.sp` / `.ws.sp` (sparse metadata operand), the
    #   .collector::/.ashift qualifiers, and the i8 convolution lines that
    #   only differ by those qualifiers: no call sites.
    # - .ws without the zero-column-mask-desc operand: every caller passes
    #   the mask (as literal zero).
    # - block_scale's .block16/.block32 vector sizes and its
    #   scale_vec-omitted spelling: the library always writes .scale_vec::NX.
    *[
        InstructionEntry(
            name=f"tcgen05_mma_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
                ModifierSlot("kind", ("kind::f16", "kind::tf32", "kind::f8f6f4", "kind::i8")),
            ),
            cert_arch="sm_100a",
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot(
                    "disable_output_lane",
                    role="value",
                    dtype="u32",
                    lanes=_tcgen05_mma_mask_lanes,
                ),
                OperandSlot("enable_input_d", role="pred_src"),
            ),
        )
        for form in ("ss", "ts")
    ],
    *[
        InstructionEntry(  # weight-stationary: no mask vector, a zero-column desc
            name=f"tcgen05_mma_ws_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("ws", ("ws",)),
                ModifierSlot("cta_group", ("cta_group::1",)),
                ModifierSlot("kind", ("kind::f16", "kind::tf32", "kind::f8f6f4", "kind::i8")),
            ),
            cert_arch="sm_100a",
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot("enable_input_d", role="pred_src"),
                OperandSlot("zero_col_mask", role="value", dtype="u64"),
            ),
        )
        for form in ("ss", "ts")
    ],
    *[
        InstructionEntry(  # block-scaled: A/B scale factors live in tmem
            name=f"tcgen05_mma_block_scale_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
                ModifierSlot("kind", ("kind::mxf8f6f4", "kind::mxf4", "kind::mxf4nvf4")),
                ModifierSlot("block_scale", ("block_scale",)),
                ModifierSlot("scale_vec", ("scale_vec::1X", "scale_vec::2X", "scale_vec::4X")),
            ),
            cert_arch="sm_100a",
            check=_check_tcgen05_mma_block_scale,
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot("sfa_tmem", role="addr", space="tmem"),
                OperandSlot("sfb_tmem", role="addr", space="tmem"),
                OperandSlot("enable_input_d", role="pred_src"),
            ),
        )
        for form in ("ss", "ts")
    ],
    # mma per PTX ISA 9.7.15.5.14. Four operand groups (d, a, b, c), each a
    # register vector whose length follows the Matrix Fragments tables -- four
    # callable `lanes`, one per group. `d` and `c` are separate operands: the
    # ISA lists them separately and the legacy helper bound them to separate
    # "=" and "r" constraints, so no read-modify-write constraint is involved
    # even when a caller passes the same registers for both.
    #
    # NOT REGISTERED:
    # - The `.kind::`/`.block_scale` lines and the .e3m2/.e2m3/.e2m1 types,
    #   which require sm_120a -- outside the architectures this table certifies.
    # - `mma.sp`, a separate instruction with a metadata operand.
    InstructionEntry(  # half precision, and the alternate formats that share it
        name="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16", "m16n8k32")),
            ModifierSlot("alayout", ("row", "col")),
            ModifierSlot("blayout", ("row", "col")),
            ModifierSlot("dtype", ("f32",)),
            ModifierSlot("atype", ("f16", "bf16", "tf32", "e4m3", "e5m2")),
            ModifierSlot("btype", ("f16", "bf16", "tf32", "e4m3", "e5m2")),
            ModifierSlot("ctype", ("f32",)),
        ),
        check=_check_mma_fp_f32,
        # Register carriers, not element types: A and B fragments are packed
        # into .b32 registers whatever the element format, so they bind "r"
        # through a uint32; an .f32 accumulator holds one float per register
        # and binds "f". Pinning each to a single-dtype PTX type keeps the
        # dtype axis from offering combinations ptxas refuses.
        operands=(
            OperandSlot("d", role="dst", dtype="f32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="f32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # the same lines with an .f16 accumulator
        name="mma_f16acc",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8k4", "m16n8k8", "m16n8k16", "m16n8k32")),
            ModifierSlot("alayout", ("row", "col")),
            ModifierSlot("blayout", ("row", "col")),
            ModifierSlot("dtype", ("f16",)),
            ModifierSlot("atype", ("f16", "e4m3", "e5m2")),
            ModifierSlot("btype", ("f16", "e4m3", "e5m2")),
            ModifierSlot("ctype", ("f16",)),
        ),
        check=_check_mma_fp_f16,
        operands=(
            OperandSlot("d", role="dst", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="u32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # mma.sync.aligned.m8n8k4.alayout.blayout.f32.f16.f16.f16
        # The one dense mma line whose .dtype and .ctype differ. The
        # half-precision syntax line of 9.7.15.5.14 quantifies both ends
        # independently --
        #
        #   mma.sync.aligned.m8n8k4.alayout.blayout.dtype.f16.f16.ctype  d, a, b, c;
        #   .ctype   = {.f16, .f32};
        #   .dtype   = {.f16, .f32};
        #
        # -- and the restriction block removes exactly one of the four
        # pairings: ".m8n8k4 : When .ctype is .f32, .dtype must also be .f32."
        # That forbids an .f16 result out of an .f32 accumulator and leaves
        # this one, which the section then writes out under the comment "// f16
        # elements in C and f32 elements in D":
        #
        #   mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16
        #   {%Rd0, %Rd1, %Rd2, %Rd3, %Rd4, %Rd5, %Rd6, %Rd7},
        #   {%Ra0, %Ra1}, {%Rb0, %Rb1}, {%Rc0, %Rc1, %Rc2, %Rc3};
        #
        # `mma` (.f32 in and out) and `mma_f16acc` (.f16 in and out) cover the
        # other two pairings; each pins one carrier for both accumulator
        # operands, and `OperandSlot.dtype` is per operand and fixed, so the
        # mixed line -- .f32 registers for `d`, packed .b32 for `c` -- can only
        # be a third entry. No `check` is needed: every combination this entry
        # enumerates is on that line, since the other shapes are unreachable
        # from here and would in any case be barred by ".m16n8k8 : .dtype must
        # be the same as .ctype." and the identical rule at .m16n8k16 /
        # .m16n8k32.
        #
        # ptxas 13.2 assembles all four layout pairs at sm_90. Its sm_90a SASS
        # is an FFMA expansion rather than an HMMA -- the same expansion the
        # already-registered .m8n8k4 spellings get there, and what the ISA's
        # own note leads one to expect: "mma.sync.m8n8k4 is optimized for
        # target architecture sm_70 and may have substantially reduced
        # performance on other target architectures." The destination values
        # are computed, so this is emulation, not the silent no-op .m8n8k4
        # produces with .bf16 / .tf32 multiplicands.
        name="mma_f16c_f32d",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8k4",)),
            ModifierSlot("alayout", ("row", "col")),
            ModifierSlot("blayout", ("row", "col")),
            ModifierSlot("dtype", ("f32",)),
            ModifierSlot("atype", ("f16",)),
            ModifierSlot("btype", ("f16",)),
            ModifierSlot("ctype", ("f16",)),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="f32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="u32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # integer, sub-byte and single-bit lines
        name="mma_int",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot(
                "shape",
                (
                    "m8n8k16",
                    "m16n8k16",
                    "m16n8k32",
                    "m8n8k32",
                    "m16n8k64",
                    "m8n8k128",
                    "m16n8k128",
                    "m16n8k256",
                ),
            ),
            ModifierSlot("alayout", ("row",)),
            ModifierSlot("blayout", ("col",)),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("s32",)),
            ModifierSlot("atype", ("u8", "s8", "u4", "s4", "b1")),
            ModifierSlot("btype", ("u8", "s8", "u4", "s4", "b1")),
            ModifierSlot("ctype", ("s32",)),
            ModifierSlot("bitop", ("xor", "and"), optional=True),
            ModifierSlot("popc", ("popc",), optional=True),
        ),
        check=_check_mma_int,
        operands=(
            OperandSlot("d", role="dst", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="u32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # double precision
        name="mma_f64",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            # The ISA writes ".m8n84" in this line's shape list -- a typo for
            # .m8n8k4, which the same section resolves three times over: ".f64
            # floating point type mma operation with .m8n8k4 shape introduced
            # in PTX ISA version 7.0.", ".f64 floating point type mma operation
            # with .m8n8k4 shape requires sm_80 or higher.", and a fragment
            # section of its own, 9.7.15.5.2 "Matrix Fragments for mma.m8n8k4
            # with .f64 floating point type".
            #
            # This slot used to omit the shape under a note claiming ptxas
            # rejects it ("Argument vector size mismatch"). Conclusion and
            # evidence were both wrong: that probe fed the operand vectors
            # `_mma_threads` produced while it still divided every .m8n8k4 by
            # 8, and ptxas was objecting to the vectors, not to the shape. With
            # 9.7.15.5.2's counts (a = b = 1, c = d = 2) ptxas 13.2 assembles
            # the line at sm_80, sm_90 and sm_90a, and its sm_90a SASS is
            # `DMMA.8x8x4 R4, R4, R6, R8` -- one real hardware instruction,
            # unlike the .m8n8k4 .bf16 / .tf32 spellings ptxas takes and then
            # emits nothing for.
            #
            # cert_arch stays the entry default: .m8n8k4 needs sm_80 but ".f64
            # floating point type mma operation with .m16n8k4, .m16n8k8, and
            # .m16n8k16 shapes require sm_90 or higher", and the floor is the
            # maximum over the entry's variants.
            ModifierSlot("shape", ("m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16")),
            ModifierSlot("alayout", ("row",)),
            ModifierSlot("blayout", ("col",)),
            ModifierSlot("dtype", ("f64",)),
            ModifierSlot("atype", ("f64",)),
            ModifierSlot("btype", ("f64",)),
            ModifierSlot("ctype", ("f64",)),
            # ".f64 floating point operations: Precision of the element-wise
            # multiplication and addition operation is identical to that of
            # .f64 precision fused multiply-add. Supported rounding modifiers
            # are : .rn : mantissa LSB rounds to nearest even. This is the
            # default. .rz : mantissa LSB rounds towards zero. .rm : mantissa
            # LSB rounds towards negative infinity. .rp : mantissa LSB rounds
            # towards positive infinity." (9.7.15.5.14)
            #
            # The syntax line leaves the position out; the section's own f64
            # examples spell it, last, after .ctype -- they write the operands
            # as brace-enclosed register vectors on the following lines, so the
            # opcode stands alone:
            #
            #     mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64.rn
            #       {%Rd0, %Rd1, %Rd2, %Rd3},
            #       {%Ra0, %Ra1},
            #       {%Rb0},
            #       {%Rc0, %Rc1, %Rc2, %Rc3};
            #
            # -- so that is where the slot sits. Optional because .rn is the
            # default, which keeps every spelling this entry rendered before
            # exactly as it was. All four modifiers assemble on all four
            # shapes (ptxas 13.2, sm_90).
            ModifierSlot("rnd", ("rn", "rz", "rm", "rp"), optional=True),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="f64", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="f64", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="f64", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="f64", lanes=_mma_lanes("c")),
        ),
    ),
    # mma.sp / mma.sp::ordered_metadata per PTX ISA 9.7.15.6.3: the same
    # multiply-accumulate with a structured-sparse A. A holds half of K (the
    # other half is implied), so only its group shrinks; `e` carries the
    # sparsity metadata and `f` selects which threads contributed it -- the ISA
    # calls it "a 32-bit integer constant with values in the range 0..3", an
    # instruction-text immediate rather than a register.
    #
    # That domain is not 0..3 everywhere: ISA 9.7.15.6.1 states it per shape and
    # type as "one thread within a group of four" (0..3), "a thread-pair" (0 or
    # 1), or "all threads ... must be 0". A `check` sees only the modifier map,
    # never the immediate axis, so each selector domain is its own entry -- the
    # split the ISA's own prose draws. ptxas enforces it exactly ("Argument 5 of
    # instruction 'mma': unexpected value '1', expected to be 0").
    #
    # NOT REGISTERED:
    # - The `.block_scale` lines (.kind::mxf4 / .mxf4nvf4 / .mxf8f6f4): they add
    #   scale-a/scale-b data plus `{byte-id, thread-id}` selector tuples, an
    #   operand shape no caller needs yet.
    # - The `.kind::f8f6f4` line, whose multiplicands mix 4- and 6-bit types in
    #   one .b32 group: same story, no call site.
    *[
        InstructionEntry(  # mma.spvariant.sync.aligned.shape.row.col.f32.atype.btype.f32
            name=f"mma_sp{suffix}",
            mnemonic="mma",
            slots=(
                ModifierSlot("spvariant", ("sp", "sp::ordered_metadata")),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", shapes),
                ModifierSlot("alayout", ("row",)),
                ModifierSlot("blayout", ("col",)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", types),
                ModifierSlot("btype", types),
                ModifierSlot("ctype", ("f32",)),
            ),
            check=check,
            operands=(
                OperandSlot("d", role="dst", dtype="f32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", role="value", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", role="value", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", role="value", dtype="f32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", role="value", dtype="u32"),
                OperandSlot("f", role="imm", choices=selector),
            ),
        )
        for suffix, shapes, types, selector, check in (
            # "One thread within a group of four consecutive threads contributes
            # the metadata for the entire group."
            (
                "",
                ("m16n8k8", "m16n8k16"),
                ("f16", "bf16", "tf32"),
                ("0", "1", "2", "3"),
                _check_mma_sp_fp_thread,
            ),
            # "A thread-pair within a group of four ... must be either 0 or 1."
            (
                "_pair",
                ("m16n8k16", "m16n8k32"),
                ("f16", "bf16", "tf32"),
                ("0", "1"),
                _check_mma_sp_fp_pair,
            ),
            # "All threads within a group of four ... must be 0."
            #
            # No check: this entry is one syntax line, ISA 9.7.15.6.3:22,
            # "mma.spvariant.sync.aligned.m16n8k64.row.col.f32.f8type.f8type.f32
            # d, a, b, c, e, f;" with ".f8type     = {.e4m3, .e5m2};", and the
            # slots spell it exactly -- one shape, and .atype/.btype domains
            # that ARE .f8type. The two positions are quantified independently,
            # so .e4m3 x .e5m2 is in the grammar (the section's own example at
            # :249 is "mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32").
            # The same-type rule this entry used to carry was wmma.mma's
            # (9.7.15.4.5:77-78), not mma.sp's -- see `_check_mma_sp_fp_types`.
            ("_all", ("m16n8k64",), ("e4m3", "e5m2"), ("0",), None),
        )
    ],
    *[
        InstructionEntry(  # the same .f16 lines with an .f16 accumulator
            name=f"mma_sp_f16acc{suffix}",
            mnemonic="mma",
            slots=(
                ModifierSlot("spvariant", ("sp", "sp::ordered_metadata")),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", (shape,)),
                ModifierSlot("alayout", ("row",)),
                ModifierSlot("blayout", ("col",)),
                ModifierSlot("dtype", ("f16",)),
                ModifierSlot("atype", ("f16",)),
                ModifierSlot("btype", ("f16",)),
                ModifierSlot("ctype", ("f16",)),
            ),
            operands=(
                OperandSlot("d", role="dst", dtype="u32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", role="value", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", role="value", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", role="value", dtype="u32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", role="value", dtype="u32"),
                OperandSlot("f", role="imm", choices=selector),
            ),
        )
        for suffix, shape, selector in (
            ("", "m16n8k16", ("0", "1", "2", "3")),
            ("_pair", "m16n8k32", ("0", "1")),
        )
    ],
    *[
        InstructionEntry(  # integer and sub-byte lines
            name=f"mma_sp_int{suffix}",
            mnemonic="mma",
            slots=(
                ModifierSlot("spvariant", ("sp", "sp::ordered_metadata")),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", shapes),
                ModifierSlot("alayout", ("row",)),
                ModifierSlot("blayout", ("col",)),
                ModifierSlot("satfinite", ("satfinite",), optional=True),
                ModifierSlot("dtype", ("s32",)),
                ModifierSlot("atype", ("u8", "s8", "u4", "s4")),
                ModifierSlot("btype", ("u8", "s8", "u4", "s4")),
                ModifierSlot("ctype", ("s32",)),
            ),
            check=check,
            operands=(
                OperandSlot("d", role="dst", dtype="u32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", role="value", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", role="value", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", role="value", dtype="u32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", role="value", dtype="u32"),
                OperandSlot("f", role="imm", choices=selector),
            ),
        )
        for suffix, shapes, selector, check in (
            ("_pair", ("m16n8k32", "m16n8k64"), ("0", "1"), _check_mma_sp_int_pair),
            ("_all", ("m16n8k64", "m16n8k128"), ("0",), _check_mma_sp_int_all),
        )
    ],
    # wgmma.mma_async per PTX ISA 9.7.16.5.2 (sm_90a). Six type groups, each
    # with an ss line (both A and B from shared memory, named by 64-bit matrix
    # descriptors) and an rs line (A from registers -- always four .b32, see
    # the fragment note above -- which also drops imm-trans-a). Like mma, one
    # entry per accumulator register type so every operand dtype is pinned.
    #
    # The accumulator is read and written in place: D = A*B + D, so `d` is
    # role="acc", the "+" constraint. The trailing arguments all live in the
    # instruction text as caller immediates:
    #   - scale-d: "the operation of the form D = A*B is issued when the input
    #     predicate argument scale-d is false". The syntax calls it a
    #     predicate, but ptxas accepts the literals 0 and 1 in that position
    #     (certified), and every call site passes a compile-time constant --
    #     so it is a choices immediate, not a runtime operand. (The legacy
    #     helper burned a setp + predicate register on it per call.)
    #   - imm-scale-a/b: "the valid values ... are -1 and 1". The legacy
    #     helper emitted 0 for "no negate", outside the ISA's domain; ptxd
    #     transcribes the documented set.
    #   - imm-trans-a/b: {0, 1}, f16/bf16 lines only (k-major inputs cannot
    #     be transposed); the rs line has no imm-trans-a.
    #
    # NOT REGISTERED: `wgmma.mma_async.sp` (sparse A, a separate instruction
    # with a metadata operand) and the `wgmma.fence`/`commit_group`/
    # `wait_group` companions (registered above).
    *[
        InstructionEntry(  # .f16 inputs, .f32 accumulator
            name=f"wgmma_f16_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("f16",)),
                ModifierSlot("btype", ("f16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # the same lines with an .f16 accumulator (f16x2 pairs)
            name=f"wgmma_f16_f16acc_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f16",)),
                ModifierSlot("atype", ("f16",)),
                ModifierSlot("btype", ("f16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # .bf16 inputs (.f32 accumulator only)
            name=f"wgmma_bf16_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("bf16",)),
                ModifierSlot("btype", ("bf16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # .tf32 inputs: k8, no transpose immediates
            name=f"wgmma_tf32_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k8" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("tf32",)),
                ModifierSlot("btype", ("tf32",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # fp8 inputs, all four e4m3/e5m2 pairings, .f32 acc
            name=f"wgmma_fp8_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("e4m3", "e5m2")),
                ModifierSlot("btype", ("e4m3", "e5m2")),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # fp8 inputs with an .f16 accumulator
            name=f"wgmma_fp8_f16acc_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f16",)),
                ModifierSlot("atype", ("e4m3", "e5m2")),
                ModifierSlot("btype", ("e4m3", "e5m2")),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # s8/u8 inputs: {.satfinite}, no scale-a/b
            name=f"wgmma_int_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_S8)),
                ModifierSlot("satfinite", ("satfinite",), optional=True),
                ModifierSlot("dtype", ("s32",)),
                ModifierSlot("atype", ("s8", "u8")),
                ModifierSlot("btype", ("s8", "u8")),
            ),
            cert_arch="sm_90a",
            operands=(
                # An .s32 accumulator rides a u32 carrier register, the same
                # bit-identical pinning as mma_int's d/c operands.
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # single-bit inputs: .op = {.and}, .popc accumulate
            name=f"wgmma_b1_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k256" for n in _WGMMA_N_B1)),
                ModifierSlot("dtype", ("s32",)),
                ModifierSlot("atype", ("b1",)),
                ModifierSlot("btype", ("b1",)),
                ModifierSlot("bitop", ("and",)),
                ModifierSlot("popc", ("popc",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
]

TABLE: dict[str, InstructionEntry] = {e.name: e for e in _ENTRIES}
