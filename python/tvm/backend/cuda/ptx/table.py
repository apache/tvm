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
"""Instruction table for the ``T.ptx`` table-driven PTX dialect.

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
and instructions then write into it — so a ptx call mirrors the PTX text
exactly. Destinations are ordinary operands (``rw="w"``, in PTX operand
order), every helper is ``void``, and every call is a statement::

    acc: T.float32                        # .reg .f32 acc;
    T.ptx.add.rn.f32(acc, x, acc)        # add.rn.f32 acc, x, acc;
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
    # `b32i`/`b64i` are the table's own tokens, not the ISA's: a bit-size type
    # with the floating carrier removed. ISA 5.2 says a `.bN` operand accepts
    # any fundamental type of that width, and for most instructions it does --
    # but a handful refuse a float register in one position while taking both
    # integer signednesses. Measured on ptxas 13.2, at sm_90 unless noted:
    # bmsk (all three operands), redux.sync's bitwise line (both), match.sync
    # and elect.sync's destination, mbarrier's phase-state token, and
    # tensormap.replace's dimension fields (sm_90a) -- each answers
    # "Arguments mismatch" to `.f32`/`.f64` and assembles with `.u32`/`.s32`
    # (resp. 64-bit). Naming that domain once is what keeps those operands from
    # being pinned to `.u32`, which would also throw away the signed spelling
    # ptxas accepts. The token never reaches the asm text -- it only chooses
    # the dtype axis -- so a name outside the ISA's vocabulary is safe here.
    "b32i": ("uint32", "int32"),
    "b64i": ("uint64", "int64"),
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
    # `.pred` is an ISA fundamental type (5.2), declared `.reg .pred p` exactly
    # as `.reg .b32 r` is -- so it is a dtype here, not a role. Inline asm has
    # no constraint letter for it, so like `.e2m1x2` above the uint32 is the
    # C-boundary carrier, not the register the instruction sees; the bridge
    # (`setp` in, `selp` out) lives in `render.BRIDGE`. The carrier is shared
    # with ordinary integer operands, which is why a `.pred` argument has to be
    # *evidenced* at the call site (`T.ptx.pred(x)`) rather than inferred from
    # its dtype: ptxas tells `%p` from `%r` by the declared register class, and
    # the tag is that declaration lifted to the call.
    "pred": ("uint32",),
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
DtypeFn = Callable[[dict], str]  # modifier map -> this operand's PTX type token
DtypesFn = Callable[[dict], tuple[str, ...]]  # modifier map -> accepted TVM dtypes


@dataclass(frozen=True)
class OperandSlot:
    """One operand of an instruction family, in PTX operand order.

    Two independent axes describe an operand, mirroring how PTX itself
    describes one: ``rw`` is the data direction, ``dtype`` is the register
    class, and ``kind`` names the few operands that are not plain registers.

    ``rw`` -- direction, which fixes the whole C-boundary story:
      - ``"r"``   an input. Passed by value, bound ``"r"``-style. ``@p`` ok.
      - ``"w"``   a destination the instruction writes; the previous value is
        dead, so it binds ``"="`` and the helper takes a C++ reference. The
        caller passes a writable lvalue (a scalar or a buffer element),
        mirroring PTX's own "declare the register, then name it as an
        operand". Blocks ``@p``: a false predicate leaves the register
        unwritten, but ``"="`` has already told nvcc the old value is dead.
      - ``"rw"``  a register the instruction reads AND writes -- an in-place
        accumulator, bound ``"+"``. Takes an lvalue like a ``"w"``; unlike one
        it does not block ``@p``, because ``"+"`` keeps the old value live
        under a false predicate.

    ``kind`` -- what the operand *is*, for the three that are not registers.
    All of them are inherently read-only (no ISA line writes an address
    register or a text immediate), so they leave ``rw`` at its default:
      - ``"reg"``  (default) a register operand typed by ``dtype`` (fixed per
        operand) or else the entry's ``type`` modifier slot.
      - ``"addr"`` memory operand, rendered ``[%k]``; state space comes from
        ``space`` (fixed per operand, for instructions whose operands live in
        different spaces like cp.async.bulk) or else the entry's ``space``
        modifier slot. Shared-space addresses are auto-coerced (generic
        pointer -> cvta) or accepted as raw ``uint32``.
      - ``"ptr"``  raw pointer value, rendered ``%k`` (e.g. ``cvta`` input).
      - ``"imm"``  an operand that lives in the instruction *text*, not in a
        register. With ``literal`` the ISA fixes its value (st.bulk's initval
        "must be zero"): no C parameter, no call argument. With ``choices``
        the caller picks the value -- a compile-time constant, validated
        against the closed set at trace time and baked into the text, with
        one helper generated per value (the way `cp.async.wait_group N` or
        `setmaxnreg`'s nreg exist only as integer literals in the ISA).

    ``dtype`` names the ISA type, and two of those name a register class the
    inline-asm constraint alphabet cannot bind (``.pred`` has no letter at
    all, ``.b8`` is below ``"h"``). Those dtypes carry a *bridge* in
    :data:`render.BRIDGE`: the value crosses the C boundary in a wider
    carrier, and a conversion instruction inside the asm block moves it
    between the carrier and a block-local register of the real class. ``rw``
    picks which conversions fire -- in for ``"r"``, out for ``"w"``, both for
    ``"rw"``. These are the asm block's sanctioned exceptions to the
    single-instruction invariant: boundary conversions, never semantics.

    Like ``lanes``, ``dtype`` may be a *function* of the modifier map, for the
    operands the ISA types by formula rather than by a token in the
    instruction text: ``mul.wide``'s destination ("d is twice as wide as a and
    b", 9.7.1.3) and ``dp4a``'s accumulator ("c has type .u32 if both .atype
    and .btype are .u32 else .s32", 9.7.1.24). Neither type is spellable as a
    slot reference, because neither is written in the instruction -- and a
    slot could not hold it either: every written token is rendered into the
    opcode, so a `.s64` slot would emit `mul.wide.s32.s64`.

    Such a function must be pure in the modifier map, total over every map
    ``variants()`` yields (all required slots filled, ``check`` passed), and
    return a :data:`PTX_TYPE_DTYPES` key -- the same contract ``lanes`` and
    ``sinkable`` carry, and for the same reason: the modifier set is closed,
    so every derived token is still enumerable, dispatchable and certifiable.
    Write it as a module-level function, never a per-entry lambda: a frozen
    dataclass hashes a callable field by identity, so a fresh lambda per entry
    would make otherwise-equal slots compare unequal.

    ``dtypes`` optionally narrows or widens the TVM dtype domain independently
    of the ISA type. This is for operands whose register may legally exceed the
    instruction type, such as the sign-extending destination of ``ld.s32``.
    Like ``dtype``, a callable must be pure, total, and module-level. Leaving it
    unset uses :data:`PTX_TYPE_DTYPES` for the resolved ISA type.

    ``lanes`` > 1 makes the operand a brace-enclosed register group. PTX writes
    the group in the operand list (``mov.b64 d, {lo, hi}``), so the group is
    part of the *shape*, never of the dotted modifier text.
    """

    name: str
    rw: str = "r"  # "r" | "w" | "rw" -- direction; see the docstring
    kind: str = "reg"  # "reg" | "addr" | "ptr" | "imm"
    space: str | None = None
    dtype: str | DtypeFn | None = None
    dtypes: tuple[str, ...] | DtypesFn | None = None
    # Whether this independent byte-address operand accepts
    # ``T.ptx.addr(base, byte_offset)``. Composite address members and tmem
    # addresses are different PTX operand classes and must leave this false.
    allow_imm_offset: bool = False
    # kind="imm" is a value in the instruction *text* (never a C parameter),
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
    # A pipe-joined operand pair: adjacent slots naming the same `pipe` render
    # as one PTX operand written `p|q`, each keeping its own C parameter,
    # constraint and bridge. Same idea as `bracket` above, different separator
    # and a different reason: `setp.lt.and.s32 p|q, a, b, r;` (ISA 9.7.6.2)
    # writes two predicates, the second one from the *complement* of the
    # compare result, so `q` is a second destination and not a restatement of
    # `p`. The ISA writes the pair in one operand position, so the table does
    # too -- a group, like `bracket`, rather than two operands the renderer
    # would separate with a comma.
    pipe: str | None = None
    # Whether the ISA lets a lane of this operand be written `_`, the sink
    # symbol: "this element takes no register". What that means follows the
    # direction, and it is NOT a destination-only spelling -- the ISA puts it
    # on a source too:
    #   rw="w"   the element is not written   (ld's `d`, mov's unpack `d`)
    #   rw="rw"  neither read nor written     (clusterlaunchcontrol's `.v4`,
    #            whose own example is `{xctaid, _, _, _}`)
    #   rw="r"   the element is not stored    (st's `b`, ISA 9.7.9.11)
    # So this is a per-slot fact read off the syntax line, never derived from
    # the direction.
    #
    # Like `lanes`, this may be a function of the modifier map: the ISA states
    # the condition per syntax line, and for the 256-bit ld/st it depends on
    # the vector width and the element type.
    #
    # An always-sunk operand is NOT this: it is a fixed value in the
    # instruction text, i.e. a `kind="imm"` with `literal="_"`, which is how
    # mbarrier.arrive's `state` is registered. This field is for the case where
    # the caller chooses.
    #
    # Unlike every other field this one grants a *per-call* choice rather than
    # describing the instruction, so the chosen mask is part of the variant:
    # a sunk lane has no C parameter and no constraint, which is a different
    # helper signature. The mask 0 variant keeps the name it had before the
    # slot became sinkable.
    sinkable: bool | Callable[[dict], bool] = False


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
    # It currently has no users. Its one client was `.e2m1x2`, whose operand
    # the ISA types .b8 -- a register class inline asm cannot bind, which needs
    # staging through a block-local `.reg .b8`. That turned out to be the same
    # shape as `.pred`'s setp/selp rather than a one-off, so it became a
    # `render.BRIDGE` row and the hand-written body went away. The hatch stays
    # for a family that is genuinely irregular; every raw entry must be listed
    # in the single-instruction invariant's exemption set, so taking it is
    # never silent.
    raw_render: Callable[["InstructionEntry", str, str, tuple, tuple], str] | None = None

    @property
    def op_name(self) -> str:
        return f"tirx.ptx.{self.name}"

    @property
    def ptx_name(self) -> str:
        return self.mnemonic or self.name

    @property
    def family(self) -> str:
        """The attribute users type: ``T.ptx.<family>``.

        The mnemonic with dots folded to underscores. Equal to ``name`` for
        every single-shape family (``st.bulk`` -> ``st_bulk``); the shared
        surface name where several entries differ only in operand shape, as
        all the ``mov_*`` entries do.
        """
        return self.ptx_name.replace(".", "_")

    @functools.cached_property
    def typed_operands(self) -> tuple[OperandSlot, ...]:
        """The operands carrying a dtype, in order: a dtype tuple aligns with these."""
        return tuple(s for s in self.operands if s.kind == "reg")

    @property
    def has_dst(self) -> bool:
        """Whether the instruction writes a destination operand.

        A false predicate leaves destinations unwritten. The default ``"="``
        output constraint means the inactive value is undefined to the caller;
        ``preserve_dst=True`` explicitly requests a read-write binding instead.
        An accumulator (``rw="rw"``) already binds "+", so it does not count
        here. A ``.pred`` result is a ``rw="w"`` register like any other and
        counts without needing a case of its own.
        """
        return any(s.kind == "reg" and s.rw == "w" for s in self.operands)


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
        if slot.kind == "imm" and slot.literal is not None:
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

    A callable ``dtype`` derives the token from the modifiers instead, for the
    operands the ISA types by formula (see :class:`OperandSlot`). This is the
    single place any of that is resolved, so every layer downstream -- the
    dtype axis, coercion, dispatch, rendering, certification -- sees a derived
    token exactly as it sees a written one.
    """
    if callable(slot.dtype):
        return slot.dtype(mod_map)
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
    if callable(slot.dtypes):
        return slot.dtypes(mod_map)
    if slot.dtypes is not None:
        return slot.dtypes
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


def sink_combos(entry: InstructionEntry, tokens) -> tuple[frozenset, ...]:
    """Every legal assignment of the sink symbol ``_`` to this entry's lanes.

    A sink is spelled per element, so the domain is the subsets of the
    sinkable lanes -- minus the all-sunk one. ISA 9.7.9.4 states that
    exclusion for `mov` ("provided that at least one element is a scalar
    register"); it is the conservative reading everywhere else, and an
    instruction whose every destination is discarded has nothing left to do.

    Empty set first, so the un-sunk variant keeps the helper name it had
    before the slot became sinkable.
    """
    mod_map = mods(entry, tokens)
    lanes = [
        (slot.name, lane)
        for slot in entry.operands
        if (slot.sinkable(mod_map) if callable(slot.sinkable) else slot.sinkable)
        for lane in range(lanes_of(slot, mod_map))
    ]
    if not lanes:
        return (frozenset(),)
    combos = [
        frozenset(subset)
        for size in range(len(lanes))  # never all of them
        for subset in itertools.combinations(lanes, size)
    ]
    return tuple(combos)


def renderings(entry: InstructionEntry):
    """Every ``(tokens, dtypes, predicated, imms, sinks)`` this entry renders to.

    Four axes -- modifiers, operand dtypes, caller immediates, predication --
    are multiplied. The fifth, the sink symbols, is *added*: the full product
    is walked with nothing sunk, and then the sink domain is walked once, at
    the first token combination that has any sinkable lane.

    That asymmetry is deliberate, and it is the difference between a walk this
    machine can finish and one it cannot. A sink is spelled per element, so its
    domain is 2**lanes -- 256 for the 256-bit `.v8` lines. Multiplying that by
    their 25392 modifier combinations would be 3.45 million rows for one entry,
    against 161108 for the whole table. Nothing is bought by it: whether ptxas
    accepts `_` at a given lane does not depend on `.cop`, `.scope` or the
    eviction priorities, so the product re-proves one fact 25392 times.

    Callers are unaffected either way. This function enumerates what gets
    *verified*; a call site renders from the sink mask it actually wrote, so an
    unvisited mask still compiles. The same is already true of an OPEN
    immediate, and for the same reason -- see `imm_combos`.
    """
    # One representative per *distinct* sink domain, not one per entry: `.v4`
    # and `.v8` sink four and eight lanes, so they are different domains and a
    # single representative would leave one of them unproven.
    sink_at = {}
    for tokens in variants(entry):
        combos = sink_combos(entry, tokens)
        if len(combos) > 1:
            sink_at.setdefault(frozenset().union(*combos), tokens)
    for tokens in variants(entry):
        for dtypes in dtype_combos(entry, tokens):
            for imms in imm_combos(entry):
                for predicated in pred_forms(entry):
                    yield tokens, dtypes, predicated, imms, frozenset()
    for tokens in sink_at.values():
        for sinks in sink_combos(entry, tokens):
            if not sinks:
                continue  # already yielded, above
            for dtypes in dtype_combos(entry, tokens):
                for imms in imm_combos(entry):
                    for predicated in pred_forms(entry):
                        yield tokens, dtypes, predicated, imms, sinks


def imm_slots(entry: InstructionEntry) -> tuple[OperandSlot, ...]:
    """The caller-passed immediates (choices or open), in operand order; an imm
    tuple aligns with these. Literal imms are table-owned and not in it."""
    return tuple(s for s in entry.operands if s.kind == "imm" and s.literal is None)


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


def _check_cache_hint(m):
    """`{.level::cache_hint}` and its `{, cache_policy}` operand, per ISA
    9.7.9.8/9.7.9.11/9.7.12.x and MEASURED on ptxas 13.2 at -arch=sm_90.

    The qualifier is an L2 policy, so it is spelled only on lines that can
    reach L2: `.global` or generic addressing. `.local`, `.shared` and the
    `.volatile` line (whose ISA syntax carries no `{.level::cache_hint}` at
    all) are each answered with "Modifier '.L2::cache_hint' cannot be used
    with ...". `.mmio` is excluded by its own "no cache qualifiers" rule, which
    had to grow to name this qualifier: ptxas answers "Modifier
    '.L2::cache_hint' cannot be combined with modifier '.mmio'".
    Accepted alongside .cop, .nc, both eviction priorities, .level::prefetch_size
    and the .acquire/.relaxed scoped lines -- probed, all OK.
    """
    if not m.get("cache"):
        return None
    if m["space"] in ("local", "shared", "shared::cta", "shared::cluster"):
        return f".level::cache_hint is an L2 policy and is not valid on .{m['space']}"
    if m.get("sem") == "volatile":
        return "the .volatile syntax line carries no .level::cache_hint"
    return None


def _check_ld(m):
    """Scalar ld grammar per PTX ISA 9.7.9.8 (ld) and 9.7.9.9 (ld.global.nc)."""
    hint = _check_cache_hint(m)
    if hint:
        return hint
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
        if cop or nc or l1ev or prefetch or m.get("cache"):
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


def _sink256(m):
    """Whether the 256-bit lines admit `_` under these modifiers.

    ISA 9.7.9.8/9.7.9.11, verbatim: "sink symbol '_' can be used in vector
    expression d when: .vec is .v8 and .type is .b32 or .s32 or .u32 or .f32
    OR .vec is .v4 and .type is .b64 or .s64 or .u64 or .f64" (and the same
    sentence for `b` on st). It is the same pairing that gates
    `.level2::eviction_priority`, which is not a coincidence -- both are
    properties of the 32-byte access, not of the qualifier.

    ptxas is looser than this: it also takes `_` on `.v4` with a 32-bit type
    (measured, CUDA 13.2 at sm_100). The ISA is the law here -- toolchain
    evidence narrows what the ISA permits, it never widens it -- so that
    spelling stays out.
    """
    return (m["vec"] == "v8" and m["type"] in ("b32", "s32", "u32", "f32")) or (
        m["vec"] == "v4" and m["type"] in ("b64", "s64", "u64", "f64")
    )


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
    hint = _check_cache_hint(m)
    if hint:
        return hint
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
        if cop or l1ev or m.get("cache"):
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
    """rcp's four syntax lines, across two ISA subsections.

        rcp.approx{.ftz}.f32  d, a;   rcp.rnd{.ftz}.f32  d, a;   (9.7.3.13)
        rcp.rnd.f64           d, a;
        rcp.approx.ftz.f64    d, a;                              (9.7.3.14)

    The ISA gives the last one a subsection of its own because it is a
    different *computation* -- a gross approximation off the top 20 mantissa
    bits, with its own corner-case table -- but its syntax is one more cell of
    this grid, and the shape (`d, a`) is unchanged. So it lives here, with the
    mandatory `.ftz` of its syntax line enforced below rather than by a second
    entry that would render identically.
    """
    if m["mode"] == "approx":
        if m["type"] == "f64" and not m["ftz"]:
            return (
                "there is no rcp.approx.f64: the f64 approximation is spelled "
                "rcp.approx.ftz.f64, with .ftz mandatory (PTX ISA 9.7.3.14)"
            )
        return None
    if m["type"] == "f64" and m["ftz"]:
        return "rcp.rnd.f64 takes no .ftz"
    return None


def _check_sqrt(m):
    """sqrt's three lines (PTX ISA 9.7.3.15).

        sqrt.approx{.ftz}.f32  d, a;   sqrt.rnd{.ftz}.f32  d, a;
        sqrt.rnd.f64           d, a;

    Unlike rcp, there is no f64 approximation at any spelling -- 9.7.3.15 is
    the whole of sqrt, and it offers `.approx` on the .f32 line only.
    """
    if m["mode"] == "approx" and m["type"] != "f32":
        return "sqrt.approx exists only on the .f32 line; .f64 requires .rn/.rz/.rm/.rp"
    if m["type"] == "f64" and m["ftz"]:
        return "sqrt.rnd.f64 takes no .ftz"
    return None


def _check_div_f(m):
    """The floating-point divide lines (PTX ISA 9.7.3.8).

        div.approx{.ftz}.f32  d, a, b;   div.full{.ftz}.f32  d, a, b;
        div.rnd{.ftz}.f32     d, a, b;   div.rnd.f64         d, a, b;

    Both approximations (`.approx`, the fast one, and `.full`, the full-range
    one) are single-precision only; the ISA spells no rounding modifier on
    either, which is why `mode` fuses them with `.rnd` into one slot.
    """
    if m["mode"] in ("approx", "full") and m["type"] != "f32":
        return f"div.{m['mode']} exists only on the .f32 line; .f64 requires .rn/.rz/.rm/.rp"
    if m["type"] == "f64" and m["ftz"]:
        return "div.rnd.f64 takes no .ftz"
    return None


# The .type line shared by most of PTX ISA 9.7.1: mul/mad (9.7.1.3/4), sad
# (9.7.1.8), div (9.7.1.9) and rem (9.7.1.10) all spell exactly this set.
_INT_TYPES = ("u16", "u32", "u64", "s16", "s32", "s64")

# `.wide`'s result type (ISA 9.7.1.3: "If .wide is specified, then d is twice
# as wide as a and b to receive the full result of the multiplication"; 9.7.1.4
# says the same of mad's d *and* c). Its keys are also the domain of the .wide
# lines -- "The .wide suffix is supported only for 16- and 32-bit integer
# types" -- so the entry's type slot and this table cannot drift apart.
_WIDE_RESULT = {"u16": "u32", "s16": "s32", "u32": "u64", "s32": "s64"}


def _wide_dtype(m):
    """`.wide`'s double-width operand type (ISA 9.7.1.3/9.7.1.4)."""
    return _WIDE_RESULT[m["type"]]


def _ld_dst_dtypes(m):
    """Scalar ``ld.s32`` may sign-extend into a wider destination register."""
    if m["type"] == "s32":
        return ("int32", "int64")
    return PTX_TYPE_DTYPES[m["type"]]


def _dp_acc_dtype(m):
    """dp4a/dp2a's accumulator type (ISA 9.7.1.24/9.7.1.25).

    "Operand c has type .u32 if both .atype and .btype are .u32 else operand c
    has type .s32" -- and d accumulates into the same type. It is a function of
    two modifiers, so no single slot reference can express it.
    """
    return "u32" if (m["atype"], m["btype"]) == ("u32", "u32") else "s32"


def _check_int_addsub(m):
    """Which types the integer add/sub lines take `.sat` on (ISA 9.7.1.1/9.7.1.2).

        add.type1        d, a, b;   .type1 = {.u16, .u64, .s16, .s64}
        add{.sat}.type2  d, a, b;   .type2 = {.u32, .u16x2, .u8x4, .s32, .s16x2, .s8x4}
        sub.type1        d, a, b;   .type1 = {.u16, .u32, .u64, .s16, .s64}
        sub{.sat}.type2  d, a, b;   .type2 = {.s32, .u8x4, .s8x4}

    Of the saturating forms only `.s32` is registered: `add.sat` on
    .u32/.u16x2/.s16x2 arrived in PTX ISA 9.2 as "sm_120f or higher in the same
    family", and the `.u8x4`/`.s8x4` types are excluded outright for the same
    reason (see the `_MINMAX_INT_PLAIN` note below). `add.sat.s32` is the PTX
    1.0 form and is supported everywhere.
    """
    if m["sat"] and m["type"] != "s32":
        return (
            f".sat is registered on .s32 only; .sat.{m['type']} is an sm_120f-only form "
            f"(PTX ISA 9.2)"
        )
    return None


def _check_int_mad(m):
    """`.sat` on the multiply-add lines: `.hi` mode, `.s32` type, nothing else.

    Both lines spell it as a syntax line of its own -- `mad.hi.sat.s32 d, a, b,
    c;` (ISA 9.7.1.4) and `mad24.hi.sat.s32 d, a, b, c;` (9.7.1.7) -- with the
    Notes repeating "Applies only to .s32 type in .hi mode".
    """
    if m["sat"] and (m["mode"], m["type"]) != ("hi", "s32"):
        return ".sat applies only to the .hi.s32 form"
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


# The .type line every PTX ISA 9.7.4 instruction spells: a scalar and a packed
# pair in each of the two half formats. `.f16`/`.bf16` ride a 16-bit register
# and the packed pairs a 32-bit one (PTX_TYPE_DTYPES).
_HALF_TYPES = ("f16", "f16x2", "bf16", "bf16x2")


def _check_half_fma(m):
    """The half-precision fma lines, ISA 9.7.4.4.

        fma.rnd{.ftz}{.sat}.f16    d, a, b, c;   fma.rnd{.ftz}.relu.f16    d, a, b, c;
        fma.rnd{.ftz}{.sat}.f16x2  d, a, b, c;   fma.rnd{.ftz}.relu.f16x2  d, a, b, c;
        fma.rnd{.relu}.bf16        d, a, b, c;   fma.rnd.oob{.relu}.type   d, a, b, c;
        fma.rnd{.relu}.bf16x2      d, a, b, c;

    Three rules read off those lines. The bf16 ones take neither `.ftz` nor
    `.sat`, as they do for add/sub/mul. `.sat` and `.relu` are two different
    clampings on two different syntax lines, so no line offers both. And the
    `.oob` line spells only `{.relu}` beside it -- it is written over the whole
    `.type` set, but with no `.ftz` and no `.sat`.
    """
    if m["type"].startswith("bf16") and (m["ftz"] or m["sat"]):
        return f".{m['type']} takes no .ftz or .sat"
    if m["oob"] and (m["ftz"] or m["sat"]):
        return "the .oob line spells no .ftz or .sat, only {.relu}"
    if m["sat"] and m["relu"]:
        return ".sat and .relu are separate syntax lines: write one or the other"
    return None


def _check_half_absneg(m):
    """The half-precision abs/neg lines, ISA 9.7.4.6 / 9.7.4.5.

        op{.ftz}.f16  d, a;   op{.ftz}.f16x2  d, a;
        op.bf16       d, a;   op.bf16x2       d, a;

    Same shape as the f32/f64 pair (`_check_absneg`), with the bf16 rule of the
    half group in place of the f64 one. It cannot share `_check_half_arith`:
    that reads a `.sat` slot these entries do not declare.
    """
    if m["type"].startswith("bf16") and m["ftz"]:
        return f".{m['type']} takes no .ftz"
    return None


def _check_half_ex2(m):
    """The half-precision ex2 lines, ISA 9.7.4.10.

        ex2.approx.atype      d, a;   .atype = {.f16,  .f16x2}
        ex2.approx.ftz.btype  d, a;   .btype = {.bf16, .bf16x2}

    `.ftz` is not optional here in either direction: the bf16 line spells it
    mandatorily and the f16 line does not offer it at all. That is what keeps
    these lines out of the `.f32` ex2 entry, whose `.ftz` is optional.
    """
    if m["type"].startswith("bf16"):
        if not m["ftz"]:
            return "the bf16 lines spell .ftz mandatorily: ex2.approx.ftz.bf16{x2}"
        return None
    if m["ftz"]:
        return f".{m['type']} has no .ftz line (only the bf16 lines spell it)"
    return None


# The comparison operators of set/setp (PTX ISA 9.7.6.1/9.7.6.2), grouped the
# way the Integer Notes and Floating Point Notes group them.
_CMP_ORDERED_OPS = ("eq", "ne", "lt", "le", "gt", "ge")
# "For unsigned values, the comparison operators lo, ls, hi, and hs ... may be
# used instead of lt, le, gt, ge" -- alternates, not additions, and unsigned only.
_CMP_UNSIGNED_OPS = ("lo", "ls", "hi", "hs")
# The unordered counterparts, plus the two NaN predicates. Floating point only.
_CMP_UNORDERED_OPS = ("equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan")
_CMP_OPS = (*_CMP_ORDERED_OPS, *_CMP_UNSIGNED_OPS, *_CMP_UNORDERED_OPS)
# `.BoolOp`, which combines the compare result with a third predicate operand.
_CMP_BOOL_OPS = ("and", "or", "xor")
# The source type line, shared by set, setp and selp.
_CMP_TYPES = ("b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64")


def _check_cmp(m):
    """Which comparison operators each source type accepts (ISA 9.7.6.1/9.7.6.2).

    The `.CmpOp` line in the Syntax block is the union over all types; the
    per-type rule is in the Notes, and it is three disjoint sets:

    - bit-size: "The untyped, bit-size comparisons are eq and ne."
    - signed:   the six ordered operators, and nothing else.
    - unsigned: those six plus lo/ls/hi/hs, the unsigned-only alternates.
    - float:    the six ordered ones plus the unordered equ/neu/ltu/leu/gtu/geu
      and the NaN predicates num/nan -- never the unsigned alternates.

    `.ftz` is separate and simpler: "Modifier .ftz applies only to .f32
    comparisons."
    """
    # `set` names its source type `stype` (its `dtype` is the destination);
    # `setp` has a single `type` slot. The comparison is on the source either way.
    ty = m.get("stype") or m["type"]
    op = m["cmp"]
    if m["ftz"] and ty != "f32":
        return ".ftz applies only to .f32 comparisons"
    if ty.startswith("b"):
        if op not in ("eq", "ne"):
            return f"the bit-size type .{ty} compares only with eq/ne"
        return None
    if ty.startswith("s"):
        if op not in _CMP_ORDERED_OPS:
            return f".{ty} is signed: {'/'.join(_CMP_ORDERED_OPS)} only"
        return None
    if ty.startswith("u"):
        if op in _CMP_UNORDERED_OPS:
            return f"{op} is a floating-point comparison, not valid on .{ty}"
        return None
    if op in _CMP_UNSIGNED_OPS:
        return f"{op} is an unsigned-only alternate, not valid on .{ty}"
    return None


# The `.CmpOp` line of the half-precision comparisons (PTX ISA 9.7.7): the
# ordered six and the unordered six plus num/nan. The unsigned alternates
# lo/ls/hi/hs of 9.7.6 are absent from this section's line entirely.
#
# MEASURED, NOT REGISTERED: ptxas does accept them on an integer source with a
# half destination (`set.lo.f16.u32` assembles at sm_90), but no 9.7.7 syntax
# line spells them, so they stay out -- the table follows the ISA where ptxas
# is the more permissive of the two.
_HALF_CMP_OPS = (*_CMP_ORDERED_OPS, *_CMP_UNORDERED_OPS)
# The `.stype` line of `set.CmpOp{.ftz}.f16.stype` and its bf16 twin. Note what
# is NOT in it: `.bf16` itself. (`set.bf16.bf16` also assembles, and is left out
# for the same reason as above -- the ISA spells no line for it.)
_SET_HALF_STYPES = (
    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f16", "f32", "f64",
)  # fmt: skip
# The packed half types, which 9.7.7 treats as one pair of lanes per register.
_HALF_X2 = ("f16x2", "bf16x2")
# The bit-size widths the logic and shift instructions operate on (PTX ISA
# 9.7.8): "fundamentally untyped ... provided the operands are of the same
# size". `_LOGIC_TYPES` adds the predicate line that and/or/xor/not also carry.
_BIT_TYPES = ("b16", "b32", "b64")
_LOGIC_TYPES = ("pred", *_BIT_TYPES)
# scalar mov's type line (PTX ISA 9.7.9.3): "Although only predicate and
# bit-size types are required, we include the arithmetic types for the
# programmer's convenience".
_MOV_TYPES = ("pred", "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64")
# The state spaces cvta converts between and isspacep queries (ISA 9.7.9.21,
# 9.7.9.20) -- the same eight, spelled the same way, in both instructions.
_CVTA_SPACES = (
    "const", "global", "local", "shared", "shared::cta", "shared::cluster", "param", "param::entry",
)  # fmt: skip
# The source types `set` will take `.ftz` with. MEASURED: the ISA writes
# `{.ftz}` across the whole `set.CmpOp{.ftz}.f16.stype` line, but ptxas 13.2
# answers "Illegal modifier '.ftz' for instruction 'set'" for every source
# outside this set -- probed over all 8 destination x 15 source pairs at
# sm_90. The rule that survives the probe is the one the modifier means:
# `.ftz` flushes subnormal *inputs*, so it attaches to the source's precision,
# and only these three have subnormals to flush at single-or-half width.
_FTZ_SET_STYPES = ("f16", "f32", "f16x2")


def _check_half_set(m):
    """The six syntax-line groups of the half-precision `set` (ISA 9.7.7.1).

        set.CmpOp{.ftz}.f16.stype     / set.CmpOp.bf16.stype      .stype = the 12 above
        set.CmpOp{.ftz}.dtype.f16     / set.CmpOp.dtype.bf16      .dtype = {u16,s16,u32,s32}
        set.CmpOp{.ftz}.dtype.f16x2   .dtype = {f16x2,u32,s32}
        set.CmpOp.dtype.bf16x2        .dtype = {bf16x2,u32,s32}

    Three independent rules come out of that, and each was probed against ptxas
    before being written here:

    - The (dtype, stype) pairing is a grid, not a product: a half type appears
      on exactly one side of every line. The integer destinations reach only
      half sources, and the packed destinations only their own packed source.
      The wide integer cells this leaves out -- (u32, b32) and friends -- are
      not lost: they are the 9.7.6 `set` lines, which own that dtype already.
    - `.ftz` follows the *source* precision, not the line it is written on:
      only a `.f16`/`.f32`/`.f16x2` source has subnormals to flush at this
      width, and no bf16 destination takes the modifier at all. See
      `_FTZ_SET_STYPES` -- this one is measured against ptxas rather than read
      off the syntax line, which writes `{.ftz}` more broadly than ptxas
      accepts it.
    - The comparison operator follows the *source*: a bit-size source compares
      only for equality, an integer source takes the ordered six, and a
      floating-point source takes all fourteen.
    """
    dt, st, op = m["dtype"], m["stype"], m["cmp"]
    paired = (
        (dt in ("f16", "bf16") and st in _SET_HALF_STYPES)
        or (dt in ("u16", "s16", "u32", "s32") and st in ("f16", "bf16"))
        or (dt in ("u32", "s32") and st in ("f16x2", "bf16x2"))
        or (dt == "f16x2" and st == "f16x2")
        or (dt == "bf16x2" and st == "bf16x2")
    )
    if not paired:
        return f"no half-precision syntax line pairs .{dt} with .{st} (ISA 9.7.7.1)"
    if m["ftz"]:
        if dt.startswith("bf16"):
            return f"a .{dt} destination takes no .ftz"
        if st not in _FTZ_SET_STYPES:
            return f".ftz flushes subnormal inputs, so a .{st} source does not take it"
    if st.startswith("b"):
        if op not in ("eq", "ne"):
            return f"the bit-size source .{st} compares only with eq/ne"
    elif st[0] in "us":
        if op not in _CMP_ORDERED_OPS:
            return f"the integer source .{st} takes {'/'.join(_CMP_ORDERED_OPS)} only"
    return None


def _check_half_setp(m):
    """`.ftz` on the half-precision `setp` lines (ISA 9.7.7.2).

    Spelled on `.f16`/`.f16x2`, on neither bf16 line -- the same split the half
    arithmetic lines make. The other half of this section's shape, single
    destination versus `p|q`, is not a check: the ISA gives each type exactly
    one of the two, so the type domains of the entries say it instead.
    """
    if m["ftz"] and m["type"].startswith("bf16"):
        return f".{m['type']} spells no .ftz"
    return None


# --- PTX ISA 9.7.9.12 (st.async) and 9.7.9.15 (multimem) --------------------
# The vector lines of st.async: "`.v2` is supported with .b32, .b64, .s32,
# .s64, .u32, .u64, .f32 and .f64 types. `.v4` qualifier is supported with
# .b32, .s32, .u32 and .f32 types."
_ST_ASYNC_TYPES = ("b32", "b64", "b128", "u32", "u64", "s32", "s64", "f32", "f64")
_ST_ASYNC_V2 = ("b32", "b64", "u32", "u64", "s32", "s64", "f32", "f64")
_ST_ASYNC_V4 = ("b32", "u32", "s32", "f32")
# The release line's type list (9.7.9.12, second syntax block), which reaches
# below the 32-bit floor of the mbarrier line.
# MEASURED: the ISA writes `.b8`, `.u8` and `.s8` into that list, but ptxas
# 13.2 rejects all three at sm_100 -- with and without `.mmio`, at either scope
# -- while every 16-bit and wider type assembles. So the line starts at 16 bits
# here, and the byte forms wait for a toolchain that accepts them.
_ST_ASYNC_REL_TYPES = (
    "b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64", "f32", "f64",
)  # fmt: skip

# createpolicy's `.level::primary_priority` line (ISA 9.7.9.19). Wider than
# the `_L2_EVICT` set the ld/st eviction hints use: this instruction is where
# a priority is *created*, so it spells all four.
_CACHE_PRIORITIES = (
    "L2::evict_last", "L2::evict_normal", "L2::evict_first", "L2::evict_unchanged",
)  # fmt: skip

_MM_INT_TYPES = ("b32", "b64", "u32", "u64", "s32", "s64")
_MM_FLOAT_TYPES = ("f16", "f16x2", "bf16", "bf16x2", "f32", "f64")
# op x type for the integer multimem lines, measured cell by cell against
# ptxas (the ISA states it as a table, and its `.add` row omits .s64).
_MM_INT_OPS = {
    "min": ("u32", "u64", "s32", "s64"),
    "max": ("u32", "u64", "s32", "s64"),
    "add": ("u32", "u64", "s32"),
    "and": ("b32", "b64"),
    "or": ("b32", "b64"),
    "xor": ("b32", "b64"),
}
# .vec x base type for the floating-point multimem lines. The ISA gives this
# as a table ("the size of the specified type along with .vec must equal either
# 32-bits or 64-bits or 128-bits"); every cell below was probed.
_MM_VEC_TYPES = {
    "": ("f16x2", "bf16x2", "f32", "f64"),
    "v2": ("f16", "f16x2", "bf16", "bf16x2", "f32"),
    "v4": ("f16", "f16x2", "bf16", "bf16x2", "f32"),
    "v8": ("f16", "bf16"),
}


def _check_multimem_sem(m):
    """How `.sem` and `.scope` go together on every multimem line (ISA 9.7.9.15).

    The ISA writes two syntax lines per mnemonic -- `{.sem}{.scope}` and a
    `.weak` one with no scope position -- which reads as "either may be
    omitted". ptxas is stricter, and its rule is the one that makes the two
    lines distinct: everywhere but `.weak`, the pair is the same both-or-
    neither rule the mbarrier sections state in words, so that check is what
    decides it here. ptxas says both halves -- "Modifier '.relaxed' requires
    scope with 'multimem.st' instruction" and "Modifier '.cta' requires order
    with 'multimem.st' instruction". What multimem adds is `.weak`: the line
    with no scope position at all.
    """
    if m.get("sem", "") == "weak":
        return ".weak is a syntax line of its own and takes no scope" if m["scope"] else None
    return _check_mbarrier_sem_scope(m)


def _check_multimem_int(m):
    """op x type on the integer multimem lines (ISA 9.7.9.15).

    The `.type` line in the Syntax block is the union over ops; the pairing is
    the "valid combinations of .op and base type" table, and `.add` is the row
    that surprises -- it takes .s32 but not .s64.

    The `.sem`/`.scope` pairing is shared with the floating lines; see
    `_check_multimem_sem`.
    """
    error = _check_multimem_sem(m)
    if error:
        return error
    op = m.get("op", "")
    if op and m["type"] not in _MM_INT_OPS[op]:
        return f".{op} takes {' / '.join('.' + t for t in _MM_INT_OPS[op])}"
    return None


def _check_multimem_f(m):
    """`.vec` x type, `.op` x type and `.acc::f32` on the floating multimem lines.

    Three rules, all measured against ptxas over the full grid:

    - The vector width and the element type together have to make 32, 64 or
      128 bits, which is what `_MM_VEC_TYPES` tabulates. Note `.f64` takes no
      `.vec` at all, and the scalar (no-`.vec`) line takes no `.f16`/`.bf16` --
      a lone half is not one of the allowed widths.
    - `.min`/`.max` are half-only: the ISA's op table gives `.f32`/`.f64` to
      `.add` alone.
    - `.acc::f32` raises the accumulation precision of a half type, so it is
      spelled only where the type is one.

    The `.sem`/`.scope` pairing is shared with the integer lines; see
    `_check_multimem_sem`.
    """
    error = _check_multimem_sem(m)
    if error:
        return error
    ty, vec, op = m["type"], m.get("vec", ""), m.get("op", "")
    if ty not in _MM_VEC_TYPES[vec]:
        allowed = " / ".join("." + t for t in _MM_VEC_TYPES[vec])
        return f"{'.' + vec if vec else 'the scalar line'} takes {allowed}"
    if op in ("min", "max") and ty in ("f32", "f64"):
        return f".{op} is half-precision only; .{ty} is on the .add row"
    if m.get("acc"):
        if ty in ("f32", "f64"):
            return f".acc::f32 raises a half accumulation to .f32; .{ty} is already wider"
        if op != "add":
            # ptxas: "Illegal reduction operation for instruction
            # 'multimem.ld_reduce.acc::f32'". Only a sum accumulates, so only a
            # sum has an accumulation precision to raise.
            return f".acc::f32 applies to .add; .{op} keeps the operand precision"
    return None


def _check_st_async(m):
    """st.async's mbarrier line (ISA 9.7.9.12), whose `.vec` narrows the types.

    `.v4` is the 32-bit four and `.v2` everything but `.b128`, which the ISA
    gives no vector form at all.
    """
    vec, ty = m["vec"], m["type"]
    allowed = _ST_ASYNC_V2 if vec == "v2" else _ST_ASYNC_V4
    if ty not in allowed:
        return f".{vec} takes {' / '.join('.' + t for t in allowed)}"
    return None


def _check_st_async_rel(m):
    """st.async's release line: "If .mmio is specified, .scope must be .sys"
    (ISA 9.7.9.12), which ptxas enforces as an illegal-modifier error."""
    if m["mmio"] and m["scope"] != "sys":
        return ".mmio requires .sys scope"
    return None


# --- PTX ISA 9.7.14 warp-level and async reductions -------------------------
# red.async's four mbarrier lines (9.7.14.7), one op group each.
_RED_ASYNC_OPS = {"inc": ("u32",), "dec": ("u32",), "min": ("u32", "s32"),
                  "max": ("u32", "s32"), "and": ("b32",), "or": ("b32",), "xor": ("b32",),
                  "add": ("u32", "s32", "u64")}  # fmt: skip
# atom's two vector lines (9.7.14.5). A half element rides `.v2`/`.v4`/`.v8`;
# a packed pair, being twice as wide, stops at `.v4`; and `.f32` has only the
# `.add` line. Every cell probed.
_ATOM_VEC_HALF = ("f16", "bf16")


def _check_red_async(m):
    """op x type on red.async's mbarrier lines (ISA 9.7.14.7).

    The ISA writes one syntax line per op group rather than a table, so the
    grid is read off those four lines: increment/decrement on the unsigned
    word, min/max on either signed word, the bitwise ops on the untyped one,
    and add reaching to 64 bits.
    """
    op, ty = m["op"], m["type"]
    if ty not in _RED_ASYNC_OPS[op]:
        return f".{op} takes {' / '.join('.' + t for t in _RED_ASYNC_OPS[op])}"
    return None


def _check_atom_vec(m):
    """`.vec` x type on atom's vector lines (ISA 9.7.14.5).

        atom{.sem}{.scope}{.global}.add{.cache}.vec_32_bit.f32               d, [a], b;
        atom{...}.op.noftz{.cache}.vec_16_bit.half_word_type                 d, [a], b;
        atom{...}.op.noftz{.cache}.vec_32_bit.packed_type                    d, [a], b;

    The element width bounds the vector, and one cell of that is all this
    check has left to say. The `.f32` line's own entry stops its `.vec` slot at
    `.v4`, and the op split falls out of the two entries' op slots; what no
    slot can express is that the half line's `.v8` belongs to a *lone* half,
    since a packed pair is already 32 bits wide.
    """
    if m["vec"] == "v8" and m["type"] in _HALF_X2:
        return f".{m['type']} is already a 32-bit pair, so it reaches .v4; .v8 is for a lone half"
    return None


def _check_slct(m):
    """slct's two lines (ISA 9.7.6.4), which differ only in the selector type.

        slct.dtype.s32        d, a, b, c;
        slct{.ftz}.dtype.f32  d, a, b, c;

    `.ftz` is spelled on the .f32 selector line alone -- there is nothing to
    flush when the sign being tested is an integer's.
    """
    if m["ftz"] and m["ctype"] != "f32":
        return ".ftz is spelled only on the .f32 selector line"
    return None


def _check_absneg(m):
    """The one-source sign lines, `op{.ftz}.f32 d, a;` and `op.f64 d, a;`.

    abs (ISA 9.7.3.9) and neg (9.7.3.10) are spelled identically -- the f64
    line of each carries no `.ftz` -- so they share this check.
    """
    if m["ftz"] and m["type"] != "f32":
        return ".ftz appears only on the .f32 line"
    return None


def _check_farith(m):
    """Which qualifiers each add/sub/mul/fma/mad line allows (PTX ISA 9.7.3.{3,4,5,6,7}, 9.7.5).

    Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 | op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64
    Mixed-precision lines: op{.rnd}{.sat}.f32.atype  (.atype = .f16 | .bf16)

    `mad` (9.7.3.7) is the same grid minus `.f32x2` and minus the mixed lines,
    both of which its entry excludes by slot domain, so it shares this check:
    with no `srctype` slot the mixed branch never fires, and `.ftz`/`.sat`
    stay gated to the .f32 line exactly as the syntax spells them.
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
    hint = _check_cache_hint(m)
    if hint:
        return hint
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


def _check_tcgen05_mma_block_scale_block(m):
    """Valid block sizes per kind: mxf8f6f4/mxf4 use block32, while
    mxf4nvf4 supports block16 and block32."""
    valid = {
        "kind::mxf8f6f4": ("block32",),
        "kind::mxf4": ("block32",),
        "kind::mxf4nvf4": ("block16", "block32"),
    }[m["kind"]]
    if m["block_size"] not in valid:
        return f"{m['kind']} supports {'/'.join(valid)}"
    return None


# `{, byteMask}` exists exactly when `.cp_mask` is written.
_cp_mask_lanes = _present_lanes("cp_mask")


# `{, ignoreBytesLeft, ignoreBytesRight}` exists exactly when `.ignore_oob` is
# written -- one lane each, so the pair appears and disappears together.
_ignore_oob_lanes = _present_lanes("ignore_oob")


_ENTRIES = [
    # ------------------------------------------------------------------
    # PTX ISA 9.7.1 — Integer Arithmetic Instructions
    # ------------------------------------------------------------------
    # The section is registered in full, with two exceptions, each stated at
    # the entry it belongs to: the packed `.u8x4`/`.s8x4` types (and the
    # saturating forms that arrived with them), which the ISA gives only to
    # "sm_120f or higher in the same family"; and `clmad` (9.7.1.5), which is
    # PTX ISA 9.3 and does not assemble on this toolchain.
    #
    # Several mnemonics have a floating-point line further down the table. The
    # integer lines are separate entries because they share no qualifier with
    # the fp ones -- one merged entry would offer `.rnd`/`.ftz` on `.s32` -- and
    # dispatch resolves them apart on their type tokens, exactly as it already
    # does for `add`/`add_half`.
    #
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
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b"),
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
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b"),
                OperandSlot("c"),
            ),
        )
        for name in ("max", "min")
    ],
    # fns per PTX ISA 9.7.1.18: `fns.b32 d, mask, base, offset;` — one form.
    # The operands carry three different types (mask .b32, base .b32/.u32/.s32,
    # offset .s32), so each declares its own dtype rather than sharing the
    # entry's type slot.
    InstructionEntry(
        name="fns",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("mask", dtype="b32"),
            OperandSlot("base", dtype="b32"),
            OperandSlot("offset", dtype="s32"),
        ),
        asm_volatile=False,  # legacy fns carried no barrier
    ),
    # The rest of 9.7.1, in ISA order. All of it keeps the default
    # `asm_volatile`, which is also what every hand-written arithmetic helper
    # in the ecosystem does: CUTLASS (`cute/arch/simd_sm100.hpp`'s
    # add/mul/fma.f32x2, `fast_math.h`'s tanh, `functional.h`'s rcp/min/max)
    # and flashinfer (`math.cuh`'s ex2/lg2/rcp/rsqrt/tanh) write `asm volatile`
    # on register-only instructions without exception. The needless-barrier
    # worry is about the "memory" clobber, which is derived and stays off here
    # (see render.py); `volatile` alone only stops nvcc reordering or commoning
    # up the asm, and the whole arithmetic surface is uniform in taking it.
    #
    # add / sub per PTX ISA 9.7.1.1, 9.7.1.2:
    #   add.type1        d, a, b;   .type1 = {.u16, .u64, .s16, .s64}
    #   add{.sat}.type2  d, a, b;   .type2 = {.u32, .u16x2, .u8x4, .s32, .s16x2, .s8x4}
    #   sub.type1        d, a, b;   .type1 = {.u16, .u32, .u64, .s16, .s64}
    #   sub{.sat}.type2  d, a, b;   .type2 = {.s32, .u8x4, .s8x4}
    # The two lines differ only in whether `.sat` is offered, so one entry with
    # an optional slot spells both and `_check_int_addsub` draws the boundary.
    # NOT REGISTERED: the `.u8x4`/`.s8x4` types on either mnemonic, and `.sat`
    # on .u32/.u16x2/.s16x2 -- all sm_120f-only (PTX ISA 9.2), the same
    # exclusion the min/max entry above makes for the same reason.
    InstructionEntry(
        name="add_int",
        mnemonic="add",
        slots=(
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", ("u16", "u32", "u64", "u16x2", "s16", "s32", "s64", "s16x2")),
        ),
        check=_check_int_addsub,
        # `.u16x2`/`.s16x2` require sm_90 (ISA Target Notes); cert_arch is the
        # max floor over the entry's variants.
        cert_arch="sm_90",
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    InstructionEntry(
        name="sub_int",
        mnemonic="sub",
        slots=(
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", _INT_TYPES),
        ),
        check=_check_int_addsub,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    # mul / mad per PTX ISA 9.7.1.3, 9.7.1.4:
    #   mul.mode.type  d, a, b;      mad.mode.type   d, a, b, c;
    #   .mode = {.hi, .lo, .wide}    mad.hi.sat.s32  d, a, b, c;
    # `.wide` is a separate entry per mnemonic, not a third token in `mode`:
    # it changes the result *structure* ("d is twice as wide as a and b", and
    # for mad so is c), which is the table's rule for where one entry ends.
    # Splitting it also lets the slot domains state "supported only for 16- and
    # 32-bit integer types" directly, with no check needed.
    InstructionEntry(
        name="mul_int",
        mnemonic="mul",
        slots=(
            ModifierSlot("mode", ("hi", "lo")),
            ModifierSlot("type", _INT_TYPES),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    InstructionEntry(
        name="mul_wide",
        mnemonic="mul",
        slots=(
            ModifierSlot("mode", ("wide",)),
            ModifierSlot("type", tuple(_WIDE_RESULT)),
        ),
        operands=(
            OperandSlot("d", rw="w", dtype=_wide_dtype),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    InstructionEntry(
        name="mad_int",
        mnemonic="mad",
        slots=(
            ModifierSlot("mode", ("hi", "lo")),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", _INT_TYPES),
        ),
        check=_check_int_mad,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    InstructionEntry(
        name="mad_wide",
        mnemonic="mad",
        slots=(
            ModifierSlot("mode", ("wide",)),
            ModifierSlot("type", tuple(_WIDE_RESULT)),
        ),
        operands=(
            OperandSlot("d", rw="w", dtype=_wide_dtype),
            OperandSlot("a"),
            OperandSlot("b"),
            # "If .wide is specified, then d and c are twice as wide as a and b
            # to receive the result of the multiplication" (ISA 9.7.1.4).
            OperandSlot("c", dtype=_wide_dtype),
        ),
    ),
    # NOT REGISTERED: clmad (PTX ISA 9.7.1.5, `clmad.mode.u64 d, a, b, c;`).
    # It was introduced in PTX ISA 9.3; this toolchain assembles 9.2, so ptxas
    # rejects every form and certification could not prove a single variant.
    # Register it when the toolchain catches up -- it is an ordinary
    # mode + fixed-type entry with four .u64 operands.
    #
    # mul24 / mad24 per PTX ISA 9.7.1.6, 9.7.1.7: a 24x24-bit multiply held in
    # 32-bit registers, so there is no `.wide` and the type line is 32-bit only.
    #   mul24.mode.type d, a, b;     mad24.mode.type  d, a, b, c;
    #   .mode = {.hi, .lo}           mad24.hi.sat.s32 d, a, b, c;
    InstructionEntry(
        name="mul24",
        slots=(
            ModifierSlot("mode", ("hi", "lo")),
            ModifierSlot("type", ("u32", "s32")),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    InstructionEntry(
        name="mad24",
        slots=(
            ModifierSlot("mode", ("hi", "lo")),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", ("u32", "s32")),
        ),
        check=_check_int_mad,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    # sad per PTX ISA 9.7.1.8: `sad.type d, a, b, c;` -- d = c + |a - b|.
    InstructionEntry(
        name="sad",
        slots=(ModifierSlot("type", _INT_TYPES),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    # div / rem per PTX ISA 9.7.1.9, 9.7.1.10: `div.type d, a, b;`. These are
    # the integer lines; the floating-point `div` (9.7.3.8) is the `div_f`
    # entry below, sharing this mnemonic.
    *[
        InstructionEntry(
            name=name,
            slots=(ModifierSlot("type", _INT_TYPES),),
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b"),
            ),
        )
        for name in ("div", "rem")
    ],
    # abs / neg per PTX ISA 9.7.1.11, 9.7.1.12 -- signed integers only. Their
    # floating-point lines (9.7.3.9, 9.7.3.10) are the `abs_f` and `neg`
    # entries below; which of each pair carries a suffix is just which arrived
    # second. NOT REGISTERED: `neg.s8x4`, sm_120f-only as above.
    InstructionEntry(
        name="abs",
        slots=(ModifierSlot("type", ("s16", "s32", "s64")),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
    ),
    InstructionEntry(
        name="neg_int",
        mnemonic="neg",
        slots=(ModifierSlot("type", ("s16", "s32", "s64")),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
    ),
    # popc / clz per PTX ISA 9.7.1.15, 9.7.1.16: `popc.type d, a;`. Both count
    # bits of a `.b32`/`.b64` source into a `.u32` destination whatever the
    # source width -- "destination d has type .u32" -- so d is typed outright
    # while a takes the instruction type.
    *[
        InstructionEntry(
            name=name,
            slots=(ModifierSlot("type", ("b32", "b64")),),
            operands=(
                OperandSlot("d", rw="w", dtype="u32"),
                OperandSlot("a"),
            ),
        )
        for name in ("popc", "clz")
    ],
    # bfind per PTX ISA 9.7.1.17: `bfind{.shiftamt}.type d, a;`, d again .u32.
    InstructionEntry(
        name="bfind",
        slots=(
            ModifierSlot("shiftamt", ("shiftamt",), optional=True),
            ModifierSlot("type", ("u32", "u64", "s32", "s64")),
        ),
        operands=(
            OperandSlot("d", rw="w", dtype="u32"),
            OperandSlot("a"),
        ),
    ),
    # brev per PTX ISA 9.7.1.19: `brev.type d, a;`.
    InstructionEntry(
        name="brev",
        slots=(ModifierSlot("type", ("b32", "b64")),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
    ),
    # bfe / bfi per PTX ISA 9.7.1.20, 9.7.1.21. Both take their position and
    # length operands as `.u32` regardless of the instruction type ("Operands b
    # and c are type .u32, but are restricted to the 8-bit value range 0..255").
    #   bfe.type  d, a, b, c;    .type = {.u32, .u64, .s32, .s64}
    #   bfi.type  f, a, b, c, d; .type = {.b32, .b64}
    InstructionEntry(
        name="bfe",
        slots=(ModifierSlot("type", ("u32", "u64", "s32", "s64")),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b", dtype="u32"),  # start bit position
            OperandSlot("c", dtype="u32"),  # field length
        ),
    ),
    InstructionEntry(
        name="bfi",
        # The ISA's own operand names, kept as they are: the destination is `f`
        # and `d` is the *length input*, the one family where `d` is not the
        # result. Renaming them would make the helper unreadable against 9.7.1.21.
        slots=(ModifierSlot("type", ("b32", "b64")),),
        operands=(
            OperandSlot("f", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c", dtype="u32"),  # start bit position
            OperandSlot("d", dtype="u32"),  # field length
        ),
    ),
    # szext / bmsk per PTX ISA 9.7.1.22, 9.7.1.23 -- both `.mode = {.clamp,
    # .wrap}`, and both take their width operand as an unsigned 32-bit value.
    #   szext.mode.type d, a, b;   .type = {.u32, .s32}
    #   bmsk.mode.b32   d, a, b;   -- a (position) and b (width) are .u32
    InstructionEntry(
        name="szext",
        slots=(
            ModifierSlot("mode", ("clamp", "wrap")),
            ModifierSlot("type", ("u32", "s32")),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b", dtype="u32"),  # N, the extended-from width
        ),
    ),
    InstructionEntry(
        name="bmsk",
        slots=(
            ModifierSlot("mode", ("clamp", "wrap")),
            ModifierSlot("type", ("b32",)),
        ),
        operands=(
            # `b32i`, not `b32`: bmsk is one of the instructions that refuses a
            # float register (ptxas: "Arguments mismatch for instruction
            # 'bmsk'", in any of the three positions) while taking either
            # integer signedness. Probed against popc, clz, brev, bfi and mov,
            # which all accept `.f32` -- so this is bmsk's own rule, not the
            # bit-size type's.
            OperandSlot("d", rw="w", dtype="b32i"),
            OperandSlot("a", dtype="b32i"),  # start bit position
            OperandSlot("b", dtype="b32i"),  # mask width
        ),
    ),
    # dp4a / dp2a per PTX ISA 9.7.1.24, 9.7.1.25:
    #   dp4a.atype.btype       d, a, b, c;   .atype = .btype = {.u32, .s32}
    #   dp2a.mode.atype.btype  d, a, b, c;   .mode = {.lo, .hi}
    # The only families in the table with no `type` slot at all: every operand
    # is typed individually. a and b name the two written tokens; c and d take
    # the accumulator type, which is a function of both (`_dp_acc_dtype`).
    InstructionEntry(
        name="dp4a",
        slots=(
            ModifierSlot("atype", ("u32", "s32")),
            ModifierSlot("btype", ("u32", "s32")),
        ),
        operands=(
            OperandSlot("d", rw="w", dtype=_dp_acc_dtype),
            OperandSlot("a", dtype="atype"),
            OperandSlot("b", dtype="btype"),
            OperandSlot("c", dtype=_dp_acc_dtype),
        ),
    ),
    InstructionEntry(
        name="dp2a",
        slots=(
            ModifierSlot("mode", ("lo", "hi")),
            ModifierSlot("atype", ("u32", "s32")),
            ModifierSlot("btype", ("u32", "s32")),
        ),
        operands=(
            OperandSlot("d", rw="w", dtype=_dp_acc_dtype),
            OperandSlot("a", dtype="atype"),
            OperandSlot("b", dtype="btype"),
            OperandSlot("c", dtype=_dp_acc_dtype),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.3 — Floating-Point Instructions
    # ------------------------------------------------------------------
    # The section is registered in full. Its one exclusion is a form no
    # supported architecture assembles: `mad`'s sm_1x line, which omits the
    # rounding modifier (stated at the mad entry below). min/max live in the
    # 9.7.1 group above, where their integer lines share the entry.
    #
    # This group also carries the whole of PTX ISA 9.7.5 (Mixed Precision),
    # whose three subsections are not separate instructions but a fourth syntax
    # line of three that are already here -- `op{.rnd}{.sat}.f32.atype`, which
    # converts a 16-bit source to .f32 before operating. They are the optional
    # `srctype` slot on `add`/`sub` (9.7.5.{1,2}) and `fma` (9.7.5.3) below;
    # `mul` is the one mnemonic of the four the ISA gives no mixed line, which
    # is why its entry declares no such slot.
    #
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
                OperandSlot("d", rw="w"),
                # On the mixed line `a` is the converted 16-bit source; on every
                # other line it is just the instruction type.
                OperandSlot("a", dtype="srctype" if mixed else None),
                OperandSlot("b"),
            ),
        )
        # `mul` is the one line with no mixed-precision form (ISA 9.7.5).
        for name, mixed in (("add", True), ("sub", True), ("mul", False))
    ],
    # NOT REGISTERED: across this whole arithmetic group, only the
    # extended-precision lines add.cc/addc/sub.cc/subc (9.7.2.{1,2,3,4}), whose
    # carry flag is a piece of state no other instruction here has. Everything
    # else is registered: the integer lines (9.7.1.{1,2,3}) as
    # `add_int`/`sub_int`/`mul_int`/`mul_wide` above, and the half lines
    # (9.7.4.{1,2,3,4}) as `add_half`/`sub_half`/`mul_half`/`fma_half` below.
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
            OperandSlot("d", rw="w"),
            # .abtype converts both a and b; c is always the instruction type.
            OperandSlot("a", dtype="srctype"),
            OperandSlot("b", dtype="srctype"),
            OperandSlot("c"),
        ),
    ),
    # mad per PTX ISA 9.7.3.7 -- the same grid as fma above minus `.f32x2` and
    # minus the mixed-precision lines, so it shares `_check_farith`:
    #   mad.rnd{.ftz}{.sat}.f32  d, a, b, c;   mad.rnd.f64  d, a, b, c;
    # The ISA notes mad.{f32,f64} is the same instruction as fma.{f32,f64}; it
    # is registered anyway because it is a mnemonic ptxas accepts and PTX
    # written elsewhere uses.
    # NOT REGISTERED: the sm_1x line `mad{.ftz}{.sat}.f32` with no `.rnd`. The
    # ISA's own Errata says ptxas enforces the rounding modifier from PTX ISA
    # 3.2 onward, so the no-rnd form assembles on no architecture this dialect
    # targets -- which is why `rnd` is a required slot here and optional on the
    # add/sub/mul entries above.
    InstructionEntry(
        name="mad_f",
        mnemonic="mad",
        slots=(
            ModifierSlot("rnd", _FRND),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_farith,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    # div per PTX ISA 9.7.3.8 -- the floating-point lines, beside the integer
    # `div` in the 9.7.1 group above. `mode` fuses the two approximations with
    # the IEEE rounding modes because the ISA requires exactly one of them
    # ("one of .approx, .full, or .rnd is required") and spells them in the
    # same position.
    InstructionEntry(
        name="div_f",
        mnemonic="div",
        slots=(
            ModifierSlot("mode", ("approx", "full", "rn", "rz", "rm", "rp")),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_div_f,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    # abs / neg per PTX ISA 9.7.3.9, 9.7.3.10 -- one shape, one check:
    #   op{.ftz}.f32 d, a;   op.f64 d, a;
    # `abs_f` carries the suffix because the integer `abs` (9.7.1.11) took the
    # bare name first; `neg` keeps it because the integer one came second.
    *[
        InstructionEntry(
            name=name,
            mnemonic=mnemonic,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", ("f32", "f64")),
            ),
            check=_check_absneg,
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
            ),
        )
        for name, mnemonic in (("abs_f", "abs"), ("neg", None))
    ],
    # rcp / sqrt per PTX ISA 9.7.3.13 + 9.7.3.14, and 9.7.3.15. Same shape and
    # same slots; they differ only in whether an f64 approximation exists, which
    # is what separates their checks (see `_check_rcp` on why 9.7.3.14 is a cell
    # of this grid rather than an entry of its own).
    *[
        InstructionEntry(
            name=name,
            slots=(
                ModifierSlot("mode", ("approx", "rn", "rz", "rm", "rp")),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", ("f32", "f64")),
            ),
            check=check,
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("value"),
            ),
        )
        for name, check in (("rcp", _check_rcp), ("sqrt", _check_sqrt))
    ],
    # rsqrt per PTX ISA 9.7.3.16 + 9.7.3.17. No check: every cell of this 2x2
    # grid is a syntax line the ISA spells -- `rsqrt.approx{.ftz}.f32` and
    # `rsqrt.approx.f64` at 9.7.3.16, and `rsqrt.approx.ftz.f64` at 9.7.3.17.
    # `.approx` is mandatory on all of them, hence the single-choice slot.
    InstructionEntry(
        name="rsqrt",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("value"),
        ),
    ),
    # sin / cos / lg2 / ex2 per PTX ISA 9.7.3.18-9.7.3.21: four mnemonics, one
    # syntax line each, all of it `op.approx{.ftz}.f32 d, a;`. ex2's
    # half-precision forms (9.7.4.10) are the `ex2_half` entry below: they
    # cannot share this optional `.ftz` slot, because theirs is mandatory on
    # the bf16 line and unspelled on the f16 one.
    *[
        InstructionEntry(
            name=name,
            slots=(
                ModifierSlot("mode", ("approx",)),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", ("f32",)),
            ),
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("value"),
            ),
        )
        for name in ("ex2", "sin", "cos", "lg2")
    ],
    # tanh per PTX ISA 9.7.3.22: `tanh.approx.f32 d, a;` -- alone among the
    # approximations in spelling no `.ftz`, so it gets no such slot. Requires
    # sm_75, below the certification floor.
    InstructionEntry(
        name="tanh",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("type", ("f32",)),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("value"),
        ),
    ),
    # testp per PTX ISA 9.7.3.1: `testp.op.type p, a;`, whose result "is .pred".
    # The destination is a predicate register like any other in this table --
    # `render.BRIDGE` materializes it through the selp the C boundary needs.
    InstructionEntry(
        name="testp",
        slots=(
            ModifierSlot(
                "op", ("finite", "infinite", "number", "notanumber", "normal", "subnormal")
            ),
            ModifierSlot("type", ("f32", "f64")),
        ),
        operands=(
            OperandSlot("p", rw="w", dtype="pred"),
            OperandSlot("a"),
        ),
    ),
    # copysign per PTX ISA 9.7.3.2: `copysign.type d, a, b;` -- the sign bit of
    # a onto the value of b.
    InstructionEntry(
        name="copysign",
        slots=(ModifierSlot("type", ("f32", "f64")),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.4 — Half Precision Floating-Point Instructions
    # ------------------------------------------------------------------
    # The section is registered in full -- every syntax line of all ten
    # subsections, with no exclusions. Every entry here carries `_HALF_TYPES`,
    # whose four tokens ride two carriers: `.f16`/`.bf16` a 16-bit register and
    # the packed pairs a 32-bit one (see PTX_TYPE_DTYPES), which is why none of
    # these entries needs to say anything about operand types.
    #
    # min/max (9.7.4.7, 9.7.4.8) are the exception to the placement, not to the
    # coverage: their half lines live in the merged entry up in the 9.7.1 group.
    #
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
                ModifierSlot("type", _HALF_TYPES),
            ),
            check=_check_half_arith,
            # The bf16 lines of all three require sm_90 (ISA Target Notes);
            # cert_arch is the max floor over the entry's variants.
            cert_arch="sm_90",
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b"),
            ),
        )
        for name in ("add", "sub", "mul")
    ],
    # Half-precision fma (PTX ISA 9.7.4.4) -- the add/sub/mul grid plus two
    # clamping qualifiers the same-precision lines never carry:
    #   fma.rnd{.ftz}{.sat}.f16{x2}  d,a,b,c;   fma.rnd{.ftz}.relu.f16{x2}  d,a,b,c;
    #   fma.rnd{.relu}.bf16{x2}      d,a,b,c;   fma.rnd.oob{.relu}.type     d,a,b,c;
    # `.relu` clamps negatives to zero and `.oob` forces the result to +0.0 when
    # an operand is the out-of-bounds NaN (see Tensors); `_check_half_fma` holds
    # which line offers which. `.rnd` is required, as on the f32/f64 fma.
    InstructionEntry(
        name="fma_half",
        mnemonic="fma",
        slots=(
            ModifierSlot("rnd", ("rn",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("oob", ("oob",), optional=True),
            ModifierSlot("relu", ("relu",), optional=True),
            ModifierSlot("type", _HALF_TYPES),
        ),
        check=_check_half_fma,
        cert_arch="sm_90",  # the `.oob` floor, the highest over this entry
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    # Half-precision neg / abs (PTX ISA 9.7.4.5, 9.7.4.6): `op{.ftz}.f16{x2}`
    # and `op.bf16{x2}`, the same one-source shape as their f32/f64 lines.
    *[
        InstructionEntry(
            name=f"{name}_half",
            mnemonic=name,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", _HALF_TYPES),
            ),
            check=_check_half_absneg,
            cert_arch="sm_90",  # the bf16 lines need sm_80; sm_90 clears them
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
            ),
        )
        for name in ("neg", "abs")
    ],
    # Half-precision tanh (PTX ISA 9.7.4.9): `tanh.approx.type d, a;` over all
    # four types, and -- like the .f32 line of 9.7.3.22 -- no `.ftz` anywhere.
    InstructionEntry(
        name="tanh_half",
        mnemonic="tanh",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("type", _HALF_TYPES),
        ),
        cert_arch="sm_90",  # tanh.approx.bf16{x2} requires sm_90
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("value"),
        ),
    ),
    # Half-precision ex2 (PTX ISA 9.7.4.10): two lines that differ by more than
    # their type token -- `ex2.approx.f16{x2}` has no `.ftz` and
    # `ex2.approx.ftz.bf16{x2}` requires it. `_check_half_ex2` enforces both
    # directions, which is exactly why these cannot join the `.f32` ex2 entry.
    InstructionEntry(
        name="ex2_half",
        mnemonic="ex2",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", _HALF_TYPES),
        ),
        check=_check_half_ex2,
        cert_arch="sm_90",  # ex2.approx.ftz.bf16{x2} requires sm_90
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("value"),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.6 — Comparison and Selection Instructions
    # ------------------------------------------------------------------
    # The section is registered in full. Two spellings are deliberately absent,
    # neither of which is a syntax line:
    #
    #   `{!}c`, the negated predicate source of the BoolOp forms. `c` reaches
    #   this surface as a uint32 evidenced by `T.ptx.pred(x)` and turned into a
    #   predicate by a setp bridge, so a caller writes the complement into the
    #   expression it passes -- `!c` spells no program the plain form cannot.
    #
    #   `_` in place of one of setp's two destinations. The ISA allows it
    #   ("The sink symbol '_' may be used in place of any one of the
    #   destination operands"), but the same program is written by passing a
    #   scratch lvalue for the half that is not wanted, and the engine's
    #   sink guard is per operand rather than per pipe group -- it would reject
    #   `p|_` as "every lane sunk" without surgery that buys nothing.
    #
    # set / setp per PTX ISA 9.7.6.1, 9.7.6.2. Both compare a and b and
    # optionally fold a third predicate in with a Boolean operator; they differ
    # in where the answer goes -- set writes a value of `.dtype` (0xffffffff or
    # 1.0f for true), setp writes predicates.
    #   set.CmpOp{.ftz}.dtype.stype         d, a, b;
    #   set.CmpOp.BoolOp{.ftz}.dtype.stype  d, a, b, {!}c;
    #   setp.CmpOp{.ftz}.type               p[|q], a, b;
    #   setp.CmpOp.BoolOp{.ftz}.type        p[|q], a, b, {!}c;
    # The `{.BoolOp}` and `[|q]` brackets are each a separate syntax line rather
    # than an optional token: the first adds an operand, the second adds a
    # destination. That is two independent shape choices, so setp is four
    # entries and set is two, all dispatched apart by arity and operand class.
    *[
        InstructionEntry(
            name="set" if not boolop else "set_bool",
            mnemonic="set",
            slots=(
                ModifierSlot("cmp", _CMP_OPS),
                *((ModifierSlot("boolop", _CMP_BOOL_OPS),) if boolop else ()),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("dtype", ("u32", "s32", "f32")),
                ModifierSlot("stype", _CMP_TYPES),
            ),
            check=_check_cmp,
            operands=(
                # "Operand d has type .dtype; operands a and b have type
                # .stype; operand c has type .pred."
                OperandSlot("d", rw="w", dtype="dtype"),
                OperandSlot("a", dtype="stype"),
                OperandSlot("b", dtype="stype"),
                *((OperandSlot("c", dtype="pred"),) if boolop else ()),
            ),
        )
        for boolop in (False, True)
    ],
    *[
        InstructionEntry(
            name="setp" + ("_bool" if boolop else "") + ("_pq" if pq else ""),
            mnemonic="setp",
            slots=(
                ModifierSlot("cmp", _CMP_OPS),
                *((ModifierSlot("boolop", _CMP_BOOL_OPS),) if boolop else ()),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", _CMP_TYPES),
            ),
            check=_check_cmp,
            operands=(
                # `p|q` is one operand position holding two predicates: q takes
                # the Boolean applied to the *complement* of the compare, so it
                # is a second result and not a restatement of p. `pipe` renders
                # the pair with the ISA's separator (see OperandSlot.pipe).
                OperandSlot("p", rw="w", dtype="pred", pipe="pq" if pq else None),
                *((OperandSlot("q", rw="w", dtype="pred", pipe="pq"),) if pq else ()),
                OperandSlot("a"),
                OperandSlot("b"),
                *((OperandSlot("c", dtype="pred"),) if boolop else ()),
            ),
        )
        for boolop in (False, True)
        for pq in (False, True)
    ],
    # selp per PTX ISA 9.7.6.3: `selp.type d, a, b, c;` -- a if the predicate
    # is true, b otherwise. The one selection instruction whose selector is a
    # predicate rather than a number.
    InstructionEntry(
        name="selp",
        slots=(ModifierSlot("type", _CMP_TYPES),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c", dtype="pred"),
        ),
    ),
    # slct per PTX ISA 9.7.6.4: `slct{.ftz}.dtype.ctype d, a, b, c;` -- a if
    # c >= 0, else b. Two instruction types, because the value being selected
    # and the number whose sign selects it are independent: "operands d, a and
    # b are treated as a bitsize type of the same width as the first
    # instruction type; operand c must match the second".
    InstructionEntry(
        name="slct",
        slots=(
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("dtype", _CMP_TYPES),
            ModifierSlot("ctype", ("s32", "f32")),
        ),
        check=_check_slct,
        operands=(
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="dtype"),
            OperandSlot("b", dtype="dtype"),
            OperandSlot("c", dtype="ctype"),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.7 — Half Precision Comparison Instructions
    # ------------------------------------------------------------------
    # The section is registered in full: `set` (9.7.7.1) and `setp` (9.7.7.2)
    # over the half types, beside the 9.7.6 entries of the same two mnemonics.
    # They are separate entries because the type grids are disjoint -- no 9.7.6
    # slot holds a half token -- and because this section's `.CmpOp` line is a
    # different set (no unsigned alternates; see `_HALF_CMP_OPS`).
    #
    # `{!}c` is left out here for the reason given at the 9.7.6 banner above.
    # Three forms ptxas accepts that no syntax line spells are also left out,
    # each noted where it belongs: lo/ls/hi/hs on an integer source
    # (`_HALF_CMP_OPS`), `set.bf16.bf16` (`_SET_HALF_STYPES`), and a lone `p`
    # on the packed setp lines (at the pq entries below).
    #
    # set per PTX ISA 9.7.7.1. Its two type tokens are a grid rather than a
    # product -- `_check_half_set` holds the six line groups.
    *[
        InstructionEntry(
            name="set_half_bool" if boolop else "set_half",
            mnemonic="set",
            slots=(
                ModifierSlot("cmp", _HALF_CMP_OPS),
                *((ModifierSlot("boolop", _CMP_BOOL_OPS),) if boolop else ()),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("dtype", ("f16", "bf16", "u16", "s16", "u32", "s32", *_HALF_X2)),
                ModifierSlot("stype", (*_SET_HALF_STYPES, "bf16", *_HALF_X2)),
            ),
            check=_check_half_set,
            cert_arch="sm_90",  # every bf16 form; the f16 ones need only sm_53
            operands=(
                OperandSlot("d", rw="w", dtype="dtype"),
                OperandSlot("a", dtype="stype"),
                OperandSlot("b", dtype="stype"),
                *((OperandSlot("c", dtype="pred"),) if boolop else ()),
            ),
        )
        for boolop in (False, True)
    ],
    # setp per PTX ISA 9.7.7.2. Unlike 9.7.6's setp, which offers both
    # destination shapes on every type, here the type *decides* the shape: the
    # scalar lines spell one predicate, the packed lines spell `p|q`, and there
    # q is the second half's comparison rather than the complement of the first
    # (`p = BoolOp(t[0], c); q = BoolOp(t[1], c)`). So the two shapes carry
    # disjoint type domains instead of a shared one.
    # MEASURED, NOT REGISTERED: ptxas also accepts a lone `p` on the packed
    # lines, discarding the upper half's result. The ISA spells only `p|q`
    # there, and what the discarded form would compute is unstated, so a caller
    # who wants one half writes the pair and ignores q.
    *[
        InstructionEntry(
            name="setp_half" + ("_bool" if boolop else "") + ("_pq" if pq else ""),
            mnemonic="setp",
            slots=(
                ModifierSlot("cmp", _HALF_CMP_OPS),
                *((ModifierSlot("boolop", _CMP_BOOL_OPS),) if boolop else ()),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("type", _HALF_X2 if pq else ("f16", "bf16")),
            ),
            check=_check_half_setp,
            cert_arch="sm_90",  # the bf16 lines; the f16 ones need only sm_53
            operands=(
                OperandSlot("p", rw="w", dtype="pred", pipe="pq" if pq else None),
                *((OperandSlot("q", rw="w", dtype="pred", pipe="pq"),) if pq else ()),
                OperandSlot("a"),
                OperandSlot("b"),
                *((OperandSlot("c", dtype="pred"),) if boolop else ()),
            ),
        )
        for boolop in (False, True)
        for pq in (False, True)
    ],
    # ------------------------------------------------------------------
    # PTX ISA 9.7.8 — Logic and Shift Instructions
    # ------------------------------------------------------------------
    # The section is registered in full. It is the one group that needs no
    # check function anywhere: "fundamentally untyped, performing bit-wise
    # operations on operands of any type, provided the operands are of the same
    # size", so every entry's slot domain is already exactly its syntax line.
    #
    # Three of the mnemonics -- `and`, `or`, `not` -- are Python keywords, so
    # the surface spells them `T.ptx.and_`, `T.ptx.or_`, `T.ptx.not_` and
    # `escape_token`/`unescape_token` carry them across the boundary. The PTX
    # text is unaffected; only the attribute a program types is.
    #
    # and / or / xor (PTX ISA 9.7.8.1-9.7.8.3) and not (9.7.8.4): `op.type d,
    # a, b;` over `.pred` and the three bit widths. The predicate line is a
    # real one here -- "Allowed types include predicate registers" -- so with
    # `.pred` the sources want `T.ptx.pred(x)` (or a bool expression) and the
    # destination a uint32 lvalue, and the helper carries the setp/selp bridges
    # around the single instruction.
    *[
        InstructionEntry(
            name=name,
            slots=(ModifierSlot("type", _LOGIC_TYPES),),
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b"),
            ),
        )
        for name in ("and", "or", "xor")
    ],
    InstructionEntry(
        name="not",
        slots=(ModifierSlot("type", _LOGIC_TYPES),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
    ),
    # cnot per PTX ISA 9.7.8.5: `d = (a==0) ? 1 : 0`, C's `!` rather than the
    # bitwise `not` above. Its type line stops at the bit widths -- ptxas
    # answers "Unexpected instruction types specified for 'cnot'" to `.pred`,
    # which is why this entry cannot share the domain of the four above.
    InstructionEntry(
        name="cnot",
        slots=(ModifierSlot("type", _BIT_TYPES),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
    ),
    # lop3 per PTX ISA 9.7.8.6: any of the 256 three-input logical functions,
    # named by a look-up table byte rather than by an opcode.
    #   lop3.b32          d,   a, b, c, immLut;
    #   lop3.BoolOp.b32   d|p, a, b, c, immLut, q;
    # `immLut` is an OPEN immediate: it lives in the instruction text, and the
    # ISA gives it a range (0..255) rather than a list of meaningful values, so
    # enumerating 256 helpers per dtype combination would be certifying an
    # arithmetic identity 256 times. Certification samples it, exactly as it
    # does for tcgen05's immHalfSplitoff.
    InstructionEntry(
        name="lop3",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
            OperandSlot("immLut", kind="imm"),
        ),
    ),
    # The BoolOp form folds a predicate in after the fact: `p = (d != 0)
    # BoolOp q`, so it writes a bit-size result AND a predicate. `d|p` is the
    # pipe pair again (see OperandSlot.pipe), here holding two *different*
    # register classes -- the mechanism only groups the text, so each half
    # keeps its own constraint and its own bridge.
    # NOT REGISTERED: `.xor` (ptxas: "Illegal operation '.xor' for instruction
    # 'lop3'" -- the ISA's `.BoolOp` line stops at `.or`/`.and`), and `_` in
    # place of `d`, which the ISA allows here but which the engine's per-slot
    # sink guard would read as "every lane sunk"; pass a scratch lvalue.
    InstructionEntry(
        name="lop3_bool",
        mnemonic="lop3",
        slots=(
            ModifierSlot("boolop", ("or", "and")),
            ModifierSlot("type", ("b32",)),
        ),
        operands=(
            OperandSlot("d", rw="w", pipe="dp"),
            OperandSlot("p", rw="w", dtype="pred", pipe="dp"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
            OperandSlot("immLut", kind="imm"),
            OperandSlot("q", dtype="pred"),
        ),
    ),
    # shf per PTX ISA 9.7.8.7: the funnel shift. a and b are the low and high
    # halves of one 64-bit source ("Operand b holds bits 63:32 and operand a
    # holds bits 31:0"), and the direction picks which 32 bits of the shifted
    # result land in d. `.mode` bounds the shift amount: 0..32 clamping, or
    # 0..31 wrapping.
    InstructionEntry(
        name="shf",
        slots=(
            ModifierSlot("dir", ("l", "r")),
            ModifierSlot("mode", ("clamp", "wrap")),
            ModifierSlot("type", ("b32",)),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c", dtype="u32"),  # the shift amount
        ),
    ),
    # shl / shr per PTX ISA 9.7.8.8, 9.7.8.9. They differ in their type line,
    # not their shape: shl is untyped because a left shift zero-fills whatever
    # the operand means, while shr has to know whether to fill with the sign
    # bit, so it carries the signed and unsigned lines too (and keeps the
    # bit-size ones "for symmetry with shl").
    # In both, "the b operand must be a 32-bit value, regardless of the
    # instruction type" -- hence the fixed .u32 on the shift amount.
    *[
        InstructionEntry(
            name=name,
            slots=(ModifierSlot("type", types),),
            operands=(
                OperandSlot("d", rw="w"),
                OperandSlot("a"),
                OperandSlot("b", dtype="u32"),  # the shift amount
            ),
        )
        for name, types in (
            ("shl", _BIT_TYPES),
            ("shr", (*_BIT_TYPES, "u16", "u32", "u64", "s16", "s32", "s64")),
        )
    ],
    # ------------------------------------------------------------------
    # PTX ISA 9.7.9 — Data Movement and Conversion Instructions
    # ------------------------------------------------------------------
    # The largest section of the ISA, and registered across all 27 of its
    # subsections. What is left out is listed at the entry it belongs to;
    # the exclusions that are about the toolchain or the architecture rather
    # than about this table are:
    #   - `shfl` without `.sync` (9.7.9.5), removed by the ISA for sm_70+;
    #   - `multimem.st.async` (9.7.9.13), PTX ISA 9.3, which ptxas 13.2
    #     (ISA 9.2) cannot assemble;
    #   - the FP8 multimem types and `.acc::f16` (9.7.9.15), and
    #     `tensormap.replace.swizzle_atomicity` (9.7.9.27), all scoped to
    #     architectures above the one their instruction is certified at.
    # Everything else absent is an operand the helper ABI cannot carry -- a
    # symbol rather than a value -- and says so where it is excluded.
    #
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
    # and the call spelling stays `T.ptx.mov.b64(...)`.
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
    # The sink symbol `_` IS registered on the unpack destinations, per ISA
    # 9.7.9.4: "When destination operand d is a vector register, the sink
    # symbol '_' may be used for one or more elements provided that at least
    # one element is a scalar register." That proviso is `sink_combos`' rule.
    #
    # Also unregistered: scalar mov (9.7.9.3), a different instruction that
    # shares the mnemonic.
    *[
        InstructionEntry(
            name=f"mov_{direction}_{lane_dtype}x{lanes}",
            mnemonic="mov",
            slots=(ModifierSlot("type", (agg,)),),
            operands=(
                OperandSlot(
                    "d",
                    rw="w",
                    dtype=lane_dtype if unpack else agg,
                    lanes=lanes if unpack else 1,
                    # Only the unpack shape has a vector destination, which is
                    # where the ISA puts the sink symbol.
                    sinkable=unpack,
                ),
                OperandSlot(
                    "a",
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
    # Complete scalar `ld` per PTX ISA 9.7.9.8 + the 9.7.9.9 ld.global.nc forms.
    # NOT REGISTERED: each is an operand the helper-function ABI cannot carry,
    # because it is a *symbol* rather than a value --
    # - .unified (asserts its address names a variable declared with that
    #   attribute: a declaration property, invisible from a register)
    # - .param/.const spaces (kernel-parameter and const addresses are named,
    #   not computed; taking `&param` in CUDA C already materializes a generic
    #   pointer, so no C expression can deliver one)
    # Registering these needs a second rendering model that emits at the site
    # where the symbol is in scope, not a slot field.
    # The ISA permits @p on this instruction; ptx does not, because it writes a
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
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_ld,
        operands=(
            OperandSlot("d", rw="w", dtypes=_ld_dst_dtypes),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
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
    # NOT REGISTERED: `.mmio`, whose syntax line carries no `{.vec}`.
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
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_ld_vec,
        operands=(
            OperandSlot("d", rw="w", lanes=_vec_lanes),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
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
            # `_` means this element is not read from memory.
            OperandSlot("d", rw="w", lanes=_vec_lanes, sinkable=_sink256),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
        ),
    ),
    # Complete scalar `st` per PTX ISA 9.7.9.11, at parity with `ld`.
    # NOT REGISTERED: .param::func -- a kernel-parameter address is a symbol,
    # not a value, so it cannot flow through the helper-function ABI (see the
    # note on `ld` above).
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
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_st,
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
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
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_st_vec,
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value", lanes=_vec_lanes),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
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
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            # ISA 9.7.9.11 puts the sink in "vector expression b" -- the data
            # being stored -- so here `_` means this element is not written to
            # memory. Sink is not a destination-only spelling.
            OperandSlot("value", lanes=_vec_lanes, sinkable=_sink256),
        ),
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
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("size", dtype="u64"),
            OperandSlot("initval", kind="imm", literal="0"),
        ),
    ),
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
        operands=(OperandSlot("addr", kind="addr", allow_imm_offset=True),),
    ),
    # st.async per PTX ISA 9.7.9.12. Two syntax blocks that share nothing but
    # the mnemonic: one signals completion through an mbarrier, the other is a
    # release store to global memory.
    #   st.async{.weak}{.ss}.completion_mechanism{.vec}.type  [a], b, [mbar];
    #   st.async{.mmio}.sem.scope{.ss}.type                   [a], b;
    # The first is sm_90; the second needs sm_100 ("Feature 'st.async/red.async
    # with .global state space' requires .target sm_100").
    *[
        InstructionEntry(
            name="st_async_vec" if vec else "st_async",
            mnemonic="st.async",
            slots=(
                ModifierSlot("weak", ("weak",), optional=True),
                ModifierSlot("space", ("shared::cluster",), optional=True),
                ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
                *((ModifierSlot("vec", ("v2", "v4")),) if vec else ()),
                ModifierSlot("type", _ST_ASYNC_V2 if vec else _ST_ASYNC_TYPES),
            ),
            check=_check_st_async if vec else None,
            cert_arch="sm_90",
            operands=(
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("b", lanes=_vec_lanes if vec else 1),
                # The mbarrier lives in the same state space as the destination
                # ("`.ss` specifies the state space of the destination operand
                # a and the mbarrier operand mbar").
                OperandSlot("mbar", kind="addr", allow_imm_offset=True),
            ),
        )
        for vec in (False, True)
    ],
    InstructionEntry(
        name="st_async_release",
        mnemonic="st.async",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("gpu", "sys")),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("type", _ST_ASYNC_REL_TYPES),
        ),
        check=_check_st_async_rel,
        cert_arch="sm_100",
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("b"),
        ),
    ),
    # multimem per PTX ISA 9.7.9.15: operations on a multimem address, which
    # names one location on each of several GPUs at once. Three mnemonics --
    # ld_reduce reduces across the copies into a register, st writes all of
    # them, red reduces into all of them -- and each splits into an integer and
    # a floating-point entry, because only the floating lines carry `.vec` and
    # `.acc::f32`, and their op x type tables are different.
    # NOT REGISTERED: the FP8 types (.e4m3*/.e5m2*) and `.acc::f16`, whose
    # Target Notes scope them to the sm_100a/sm_120a family architectures
    # rather than to sm_90 where the rest of this subsection lives; and
    # `multimem.st.async` (9.7.9.13), which is PTX ISA 9.3 and does not
    # assemble on this toolchain.
    InstructionEntry(
        name="multimem_ld_reduce",
        mnemonic="multimem.ld_reduce",
        slots=(
            ModifierSlot("sem", ("weak", "relaxed", "acquire"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("op", tuple(_MM_INT_OPS)),
            ModifierSlot("type", _MM_INT_TYPES),
        ),
        check=_check_multimem_int,
        cert_arch="sm_90",
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
        ),
    ),
    InstructionEntry(
        name="multimem_st",
        mnemonic="multimem.st",
        slots=(
            ModifierSlot("sem", ("weak", "relaxed", "release"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("type", _MM_INT_TYPES),
        ),
        check=_check_multimem_int,
        cert_arch="sm_90",
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("b"),
        ),
    ),
    InstructionEntry(
        name="multimem_red",
        mnemonic="multimem.red",
        slots=(
            # No `.weak` here: ptxas answers "Illegal modifier '.weak' for
            # instruction 'multimem.red'", and the ISA gives red a `.redsem`
            # line without it -- the same pair the scalar `red` takes.
            ModifierSlot("sem", _RED_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("op", tuple(_MM_INT_OPS)),
            ModifierSlot("type", _MM_INT_TYPES),
        ),
        check=_check_multimem_int,
        cert_arch="sm_90",
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("b"),
        ),
    ),
    # The floating-point multimem lines. Each is two entries rather than one
    # with an optional `.vec`, because the qualifier changes the operand's
    # *shape*: written, the value is a brace-enclosed register group; omitted,
    # it is a plain register. That is the table's rule for where one entry
    # ends, and it is also what ptxas insists on ("Vector is not expected for
    # argument 1").
    *[
        InstructionEntry(
            name=name + ("_vec" if vec else ""),
            mnemonic=f"multimem.{mnem}",
            slots=(
                ModifierSlot("sem", sems, optional=True),
                ModifierSlot("scope", _ATOM_SCOPES, optional=True),
                ModifierSlot("space", ("global",), optional=True),
                *((ModifierSlot("op", ops),) if ops else ()),
                *((ModifierSlot("acc", ("acc::f32",), optional=True),) if acc else ()),
                *((ModifierSlot("vec", ("v2", "v4", "v8")),) if vec else ()),
                ModifierSlot("type", _MM_FLOAT_TYPES),
            ),
            check=_check_multimem_f,
            cert_arch="sm_90",
            operands=(
                # ld_reduce reads the multimem address into a register; st and
                # red send a value the other way.
                *(
                    (
                        OperandSlot("d", rw="w", lanes=_vec_lanes if vec else 1),
                        OperandSlot("addr", kind="addr", allow_imm_offset=True),
                    )
                    if mnem == "ld_reduce"
                    else (
                        OperandSlot("addr", kind="addr", allow_imm_offset=True),
                        OperandSlot("b", lanes=_vec_lanes if vec else 1),
                    )
                ),
            ),
        )
        # `multimem.st` names no operation (it just writes), and `multimem.red`
        # on floats offers only `.add` -- the ISA gives it a `.redop` line of
        # its own for exactly that reason. `.acc::f32` is a load-side
        # qualifier, so only ld_reduce carries it.
        for name, mnem, sems, ops, acc in (
            (
                "multimem_ld_reduce_f",
                "ld_reduce",
                ("weak", "relaxed", "acquire"),
                ("min", "max", "add"),
                True,
            ),
            ("multimem_st_f", "st", ("weak", "relaxed", "release"), (), False),
            ("multimem_red_f", "red", ("relaxed", "release"), ("add",), False),
        )
        for vec in (False, True)
    ],
    # createpolicy per PTX ISA 9.7.9.19: build the opaque 64-bit cache-eviction
    # policy that the `.level::cache_hint` operand of ld/st/cp takes. Three
    # syntax lines, three shapes:
    #   createpolicy.range{.global}.pri{.sec}.b64  policy, [a], primary, total;
    #   createpolicy.fractional.pri{.sec}.b64      policy{, fraction};
    #   createpolicy.cvt.L2.b64                    policy, access-property;
    # The fractional line's `{, fraction}` is an optional *operand*, so it is
    # two entries split by arity, as elsewhere in this table.
    *[
        InstructionEntry(
            name=name,
            mnemonic="createpolicy",
            slots=(
                ModifierSlot("kind", (kind,)),
                *((ModifierSlot("space", ("global",), optional=True),) if kind == "range" else ()),
                ModifierSlot("pri", _CACHE_PRIORITIES),
                ModifierSlot("sec", ("L2::evict_first", "L2::evict_unchanged"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            cert_arch="sm_90",  # the subsection's floor is sm_80
            operands=(
                OperandSlot("cache_policy", rw="w", dtype="u64"),
                *ops,
            ),
        )
        for name, kind, ops in (
            (
                "createpolicy_range",
                "range",
                (
                    OperandSlot("addr", kind="addr", allow_imm_offset=True),
                    OperandSlot("primary_size", dtype="u32"),
                    OperandSlot("total_size", dtype="u32"),
                ),
            ),
            ("createpolicy_fractional", "fractional", ()),
            ("createpolicy_fraction", "fractional", (OperandSlot("fraction", dtype="f32"),)),
        )
    ],
    # The third line converts a CUDA-API access property instead of describing
    # one, so it carries neither priority slot.
    InstructionEntry(
        name="createpolicy_cvt",
        mnemonic="createpolicy",
        slots=(
            ModifierSlot("kind", ("cvt",)),
            ModifierSlot("level", ("L2",)),
            ModifierSlot("type", ("b64",)),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("cache_policy", rw="w", dtype="u64"),
            OperandSlot("access_property", dtype="u64"),
        ),
    ),
    # cp.async.bulk.prefetch per PTX ISA 9.7.9.26.4.3 -- the non-tensor bulk
    # prefetch, beside the tensor one further down:
    #   cp.async.bulk.prefetch.L2.src{.level::cache_hint} [srcMem], size{, policy};
    InstructionEntry(
        name="cp_async_bulk_prefetch",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("op", ("prefetch",)),
            ModifierSlot("level", ("L2",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="global"),
            OperandSlot("size", dtype="u32"),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
        ),
    ),
    # tensormap.replace per PTX ISA 9.7.9.27: overwrite one field of a 1024-bit
    # tensor-map object in place. Three field groups, and they differ in shape
    # rather than in qualifier, so they are three entries:
    #   .field1 = {global_address, rank}                      [addr], new_val
    #   .field2 = {box_dim, global_dim, global_stride, ...}   [addr], ord, new_val
    #   .field3 = {elemtype, interleave_layout, ...}          [addr], <imm>
    # For field3 "the operand new_val must be an immediate" -- ptxas agrees
    # ("Integer constant expression expected"), and each field has its own
    # closed value table, so they are `imm` operands with per-field choices.
    # `ord` is likewise an immediate the ISA bounds to [0..4], which ptxas
    # enforces by name.
    # NOT REGISTERED: `.swizzle_atomicity` (sm_100a and later only, while the
    # rest of this instruction is sm_90a), and the `.elemtype` values 13-15 and
    # `.swizzle_mode` value 4, which need newer architectures still.
    *[
        InstructionEntry(
            name=f"tensormap_replace_{group}",
            mnemonic="tensormap.replace",
            slots=(
                ModifierSlot("mode", ("tile",)),
                ModifierSlot("field", fields),
                ModifierSlot("space", ("global", "shared::cta"), optional=True),
                ModifierSlot("width", ("b1024",)),
                ModifierSlot("type", (ty,)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                *ops,
            ),
        )
        for group, fields, ty, ops in (
            ("address", ("global_address",), "b64", (OperandSlot("new_val", dtype="b64"),)),
            ("rank", ("rank",), "b32", (OperandSlot("new_val", dtype="b32"),)),
            (
                "stride",
                ("global_stride",),
                "b64",
                (
                    OperandSlot("ord", kind="imm", choices=("0", "1", "2", "3", "4")),
                    OperandSlot("new_val", dtype="b64"),
                ),
            ),
            (
                "dim",
                ("box_dim", "global_dim", "element_stride"),
                "b32",
                (
                    OperandSlot("ord", kind="imm", choices=("0", "1", "2", "3", "4")),
                    # `b32i`: measured at sm_90a over all four 32-bit classes
                    # -- b32, u32 and s32 assemble here and `.f32` does not,
                    # while `rank` and the 64-bit fields take all four and so
                    # stay on the full axis.
                    OperandSlot("new_val", dtype="b32i"),
                ),
            ),
        )
    ],
    # The field3 group, one entry per field because each names its own closed
    # set of immediate values (ISA Table 33).
    *[
        InstructionEntry(
            name=f"tensormap_replace_{field}",
            mnemonic="tensormap.replace",
            slots=(
                ModifierSlot("mode", ("tile",)),
                ModifierSlot("field", (field,)),
                ModifierSlot("space", ("global", "shared::cta"), optional=True),
                ModifierSlot("width", ("b1024",)),
                ModifierSlot("type", ("b32",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("new_val", kind="imm", choices=values),
            ),
        )
        for field, values in (
            ("elemtype", tuple(str(i) for i in range(13))),  # 13-15 need newer archs
            ("interleave_layout", ("0", "1", "2")),
            ("swizzle_mode", ("0", "1", "2", "3")),  # 4 needs sm_103a
            ("fill_mode", ("0", "1")),
        )
    ],
    # mov, scalar form (PTX ISA 9.7.9.3): `mov.type d, a;`. A different
    # instruction from the vector pack/unpack lines above that share the
    # mnemonic -- this one just copies a register -- and dispatch tells them
    # apart by arity, since every pack/unpack shape takes at least three.
    # NOT REGISTERED: the other four source forms of the syntax block (`sreg`,
    # a variable's address, `avar+imm`, a function name). Each names a *symbol*
    # rather than a value, and a symbol cannot cross the helper-function ABI --
    # the same reason `ld` leaves out `.unified` and the `.param` spaces.
    InstructionEntry(
        name="mov",
        slots=(ModifierSlot("type", _MOV_TYPES),),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
        ),
        asm_volatile=False,  # a register copy: let nvcc common it up
    ),
    # prmt per PTX ISA 9.7.9.7: pick four arbitrary bytes out of the eight in
    # {b, a} and reassemble them. Without `.mode`, operand c is four 4-bit
    # selectors; with one, its two low bits pick a row of the mode's table.
    # The mode comes *after* the type in the text (`prmt.b32{.mode}`), which is
    # why the slots are in that order.
    InstructionEntry(
        name="prmt",
        slots=(
            ModifierSlot("type", ("b32",)),
            ModifierSlot("mode", ("f4e", "b4e", "rc8", "ecl", "ecr", "rc16"), optional=True),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("b"),
            OperandSlot("c"),
        ),
    ),
    # ldu per PTX ISA 9.7.9.10: a load whose address is uniform across the
    # warp. Two lines, scalar and vector, split the way ld's are.
    # Only `.global` or generic addressing: ptxas answers "State space
    # incorrect for instruction 'ldu'" to anything else, matching the ISA's
    # one-element `.ss` list.
    InstructionEntry(
        name="ldu",
        slots=(
            ModifierSlot("space", ("global",), optional=True),  # omitted = generic
            ModifierSlot("type", _LD_TYPES),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
        ),
    ),
    InstructionEntry(
        name="ldu_vec",
        mnemonic="ldu",
        slots=(
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        # The same 128-bit ceiling ld's vector lines have (ISA 5.4.2), and
        # measured the same way: ptxas answers "Vector type too large, exceeds
        # 128 bit limit" to `.v4` with a 64-bit type. ldu has no 256-bit line
        # to send those to, so here the check is the whole rule.
        check=_check_vec128,
        operands=(
            OperandSlot("d", rw="w", lanes=_vec_lanes),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
        ),
    ),
    # prefetchu per PTX ISA 9.7.9.16, the fourth line of that subsection: the
    # uniform cache rather than the data cache, so a different mnemonic and no
    # state space (it "requires a generic address").
    InstructionEntry(
        name="prefetchu",
        slots=(ModifierSlot("level", ("L1",)),),
        operands=(OperandSlot("addr", kind="addr", allow_imm_offset=True),),
    ),
    # applypriority / discard per PTX ISA 9.7.9.17, 9.7.9.18. Same shape: an
    # address range and a cache level, one hinting how to evict it and the
    # other saying it need not be written back at all.
    #   applypriority{.global}.L2::evict_normal  [a], size;
    #   discard{.global}.L2                      [a], size;
    # `size` is fixed at 128 by both ISA lines ("The only supported value for
    # the size operand is 128"), so it is a table-owned literal: it takes no
    # call argument and cannot be got wrong.
    *[
        InstructionEntry(
            name=name,
            slots=(
                ModifierSlot("space", ("global",), optional=True),
                ModifierSlot("level", (level,)),
            ),
            operands=(
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("size", kind="imm", literal="128"),
            ),
        )
        for name, level in (("applypriority", "L2::evict_normal"), ("discard", "L2"))
    ],
    # isspacep per PTX ISA 9.7.9.20: does this generic address fall inside the
    # window of a given state space? A `.pred` result, so it rides the bridge.
    # The address is a plain pointer rather than an `addr` operand: the
    # instruction reads the value, it does not dereference it, so there is
    # nothing to bracket and no memory to clobber.
    InstructionEntry(
        name="isspacep",
        slots=(ModifierSlot("space", _CVTA_SPACES),),
        operands=(
            OperandSlot("p", rw="w", dtype="pred"),
            OperandSlot("a", kind="ptr"),
        ),
        asm_volatile=False,  # a pure query of an address value
    ),
    # getctarank per PTX ISA 9.7.9.25: which CTA of the cluster owns this
    # address. Two lines by whether the address is a shared one or a generic
    # one; `.type` describes the *source*, while "destination d is always a
    # 32-bit register".
    *[
        InstructionEntry(
            name=name,
            mnemonic="getctarank",
            slots=(
                *((ModifierSlot("space", ("shared::cluster",)),) if shared else ()),
                ModifierSlot("type", ("u32", "u64")),
            ),
            cert_arch="sm_90",
            operands=(
                OperandSlot("d", rw="w", dtype="u32"),
                OperandSlot("a"),
            ),
            asm_volatile=False,
        )
        for name, shared in (("getctarank", True), ("getctarank_generic", False))
    ],
    # cvt.pack per PTX ISA 9.7.9.23: convert two .s32 values with saturation
    # and pack them into one register. Two syntax lines, and the operand count
    # is what separates them -- the sub-byte line needs `c` to supply the bits
    # the pair does not fill, and the 16-bit line has none left over.
    #   cvt.pack.sat.convertType.abType        d, a, b;      16-bit halves
    #   cvt.pack.sat.convertType.abType.cType  d, a, b, c;   sub-byte
    *[
        InstructionEntry(
            name=name,
            mnemonic="cvt.pack",
            slots=(
                ModifierSlot("sat", ("sat",)),
                ModifierSlot("convert", convert),
                ModifierSlot("abtype", ("s32",)),
                *((ModifierSlot("ctype", ("b32",)),) if has_c else ()),
            ),
            # `.u4`/`.s4` and `.u2`/`.s2` need sm_75, the rest sm_72.
            cert_arch="sm_90",
            operands=(
                OperandSlot("d", rw="w", dtype="u32"),
                OperandSlot("a", dtype="s32"),
                OperandSlot("b", dtype="s32"),
                *((OperandSlot("c", dtype="b32"),) if has_c else ()),
            ),
        )
        for name, convert, has_c in (
            ("cvt_pack", ("u16", "s16"), False),
            ("cvt_pack_c", ("u2", "s2", "u4", "s4", "u8", "s8"), True),
        )
    ],
    # shfl.sync per PTX ISA 9.7.9.6: exchange a register between the lanes of a
    # warp. `d[|p]` is optional in the syntax and adds a destination, so the
    # two shapes are two entries; p reports whether the computed source lane
    # was in range.
    # NOT REGISTERED: the non-sync `shfl` of 9.7.9.5, which the ISA removed for
    # sm_70 and higher in PTX ISA 6.4 -- ptxas agrees ("Instruction 'shfl'
    # without '.sync' is not supported on .target sm_70 and higher"), so it
    # assembles on no architecture this dialect targets.
    *[
        InstructionEntry(
            name="shfl_sync_p" if pq else "shfl_sync",
            mnemonic="shfl.sync",
            slots=(
                ModifierSlot("mode", ("up", "down", "bfly", "idx")),
                ModifierSlot("type", ("b32",)),
            ),
            operands=(
                OperandSlot("d", rw="w", pipe="dp" if pq else None),
                *((OperandSlot("p", rw="w", dtype="pred", pipe="dp"),) if pq else ()),
                OperandSlot("a"),
                OperandSlot("b", dtype="u32"),  # source lane or offset
                OperandSlot("c", dtype="u32"),  # packed segmask and clamp
                OperandSlot("membermask", dtype="u32"),
            ),
        )
        for pq in (False, True)
    ],
    # cvta per PTX ISA 9.7.9.21, both directions over all eight state spaces.
    # The `.to` direction converts a generic address into a space-specific one
    # and the bare direction does the reverse; they are two entries because the
    # bare one takes a plain value where `.to` takes a pointer.
    # NOT REGISTERED: `.u32`. The 32-bit ABI is unusable at sm_90 and higher --
    # ptxas rejects the program with "Program uses 32-bit address" -- and the
    # ISA's own advice for narrowing is to follow cvta with a cvt.
    InstructionEntry(
        name="cvta",
        slots=(
            ModifierSlot("dir", ("to",)),
            ModifierSlot("space", _CVTA_SPACES),
            ModifierSlot("type", ("u64",)),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("ptr", kind="ptr"),
        ),
        asm_volatile=False,  # legacy cvta carried no barrier
    ),
    InstructionEntry(
        name="cvta_generic",
        mnemonic="cvta",
        slots=(
            ModifierSlot("space", _CVTA_SPACES),
            ModifierSlot("type", ("u64",)),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            # The source is an address *within* its state space, which is an
            # ordinary 64-bit value rather than a pointer the C side owns.
            OperandSlot("a", dtype="u64"),
        ),
        asm_volatile=False,
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="atype"),
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
            OperandSlot("d", rw="w", dtype="f16x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="tf32"),
            OperandSlot("a", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="f16x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
            OperandSlot("rbits", dtype="b32"),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
            OperandSlot("rbits", dtype="b32"),
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
            OperandSlot("d", rw="w", dtype="ue8m0x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="ue8m0x2"),
            OperandSlot("a", dtype="bf16x2"),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="ue8m0x2"),
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="atype"),
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
            OperandSlot("d", rw="w", dtype="f16x2"),
            OperandSlot("a", dtype="atype"),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="atype"),
            OperandSlot(
                "scale_factor",
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
            OperandSlot("d", rw="w", dtype="dtype"),
            # The four sources are one brace-enclosed group in the operand
            # list, which is what `lanes` renders; the name is the ISA's own
            # spelling of the group, `{a, b, e, f}`.
            OperandSlot("abef", dtype="f32", lanes=4),
            OperandSlot("rbits", dtype="b32"),
        ),
    ),
    # The four `.f4x2type = { .e2m1x2 };` lines. Their one operand is typed
    # .b8 by the ISA -- :92 "When converting to .e2m1x2 data formats, the
    # destination operand d has .b8 type." and :101 "When converting from
    # .e2m1x2 to .f16x2/.bf16x2, source operand a has .b8 type." -- a register
    # class inline asm cannot bind, so it is staged through a block-local
    # `.reg .b8`. That staging is not written here: `.e2m1x2` carries a row in
    # `render.BRIDGE`, which is where the measured toolchain facts live, and
    # the derived path emits it exactly as `mov`'s `.pred` operands get their
    # setp. These entries are therefore ordinary entries.
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
            OperandSlot("d", rw="w", dtype="e2m1x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
        ),
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
            OperandSlot("d", rw="w", dtype="e2m1x2"),
            OperandSlot("a", dtype="atype"),
        ),
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
            OperandSlot("d", rw="w", dtype="f16x2"),
            OperandSlot("a", dtype="e2m1x2"),
        ),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="e2m1x2"),
            # ISA:106-107 "For .bf16x2 destination type optional scale-factor
            # operand of type .b16 can be specified along with
            # .scaled::n2::ue8m0 qualifier." -- .b16, not .b8, so only `a`
            # needs the staging register.
            OperandSlot(
                "scale_factor",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
        ),
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
            OperandSlot("d", rw="w", dtype="e2m1x4"),
            OperandSlot("abef", dtype="f32", lanes=4),
            OperandSlot("rbits", dtype="b32"),
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("a", dtype="atype"),
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
            OperandSlot("d", rw="w", dtype="f16x2"),
            OperandSlot("a", dtype="atype"),
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="atype"),
            OperandSlot(
                "scale_factor",
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
            OperandSlot("d", rw="w", dtype="dtype"),
            OperandSlot("abef", dtype="f32", lanes=4),
            OperandSlot("rbits", dtype="b32"),
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
            OperandSlot("d", rw="w", dtype="s2f6x2"),
            OperandSlot("a", dtype="f32"),
            OperandSlot("b", dtype="f32"),
            # ISA:162-163 "Optional operand scale-factor has type .b16 and
            # stores two packed scaling factors of type .ue8m0."
            OperandSlot(
                "scale_factor",
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
            OperandSlot("d", rw="w", dtype="s2f6x2"),
            OperandSlot("a", dtype="bf16x2"),
            OperandSlot(
                "scale_factor",
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
            OperandSlot("d", rw="w", dtype="bf16x2"),
            OperandSlot("a", dtype="s2f6x2"),
            OperandSlot(
                "scale_factor",
                dtype="ue8m0x2",
                lanes=_cvt_scale_lanes,
                vector=False,
            ),
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
            OperandSlot("d", rw="w"),
            OperandSlot("a", kind="ptr"),
            OperandSlot("b", dtype="u32"),
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
            OperandSlot("d", rw="w"),
            OperandSlot("a", dtype="u32"),
            OperandSlot("b", dtype="u32"),
        ),
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
            operands=(OperandSlot("group", kind="imm", choices=tuple(str(n) for n in range(8))),),
        )
        for bulk in (False, True)
    ],
    # cp.async per PTX ISA 9.7.9.26.3.1: the non-bulk asynchronous copy.
    # cp-size is an integer constant the ISA closes to {4, 8, 16} (and to 16
    # alone under .cg) -- a choices immediate. The src-size arity zero-fills
    # the destination tail; it is a separate entry told apart by arity, the
    # mbarrier.arrive precedent.
    #
    # The `{, src-size}` and `{, ignore-src}` lines are separate entries: they
    # change the operand list, which is what an entry is. Both add one operand
    # at the same position, so arity cannot tell them apart -- the ISA tells
    # them apart by register class, "The optional and non-immediate PREDICATE
    # argument ignore-src" against "a 32-bit INTEGER operand src-size", and so
    # does this table: `.pred` and `.u32` are different acceptance classes, and
    # the caller writes `T.ptx.pred(...)` for the one PTX would write `%p` for.
    # Before `.pred` was a dtype these two were indistinguishable and the
    # ignore-src lines could not be registered at all.
    *[
        InstructionEntry(
            name=f"cp_async_{cop}{'_' + tail if tail else ''}",
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
                OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared"),
                OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="global"),
                OperandSlot(
                    "cp_size", kind="imm", choices=("4", "8", "16") if cop == "ca" else ("16",)
                ),
                # src-size zero-fills the destination tail; ignore-src skips
                # the read entirely and zero-fills all of it ("If the source
                # data is ignored then zeros will be copied to destination
                # dst"), which is why it is a predicate and not a count.
                *(
                    (OperandSlot("src_size", dtype="u32"),)
                    if tail == "src_size"
                    else (OperandSlot("ignore_src", dtype="pred"),)
                    if tail
                    else ()
                ),
                OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
            ),
        )
        for cop in ("ca", "cg")
        for tail in (None, "src_size", "ignore_src")
    ],
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
    # cp.async.bulk (non-tensor), per PTX ISA 9.7.9.26.4.1: four directions.
    #
    # NOT REGISTERED:
    # - the `.sem.scope`/`.type` lines: "Support for .weak and .relaxed
    #   semantics, .scope and .type qualifiers are introduced in PTX ISA version
    #   9.3", which ptxas 13.2 in this toolchain (9.2) cannot assemble, and no
    #   call site uses them.
    # - the `{.sem}` (`.weak`) position on the plain lines, for the same reason:
    #   omitting an optional qualifier renders the same instruction, and the
    #   token itself is 9.3. Same situation `_check_ld` records for
    #   ld.mmio.acquire.
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="global"),
            OperandSlot("size", dtype="u32"),
            OperandSlot(
                "ignore_bytes_left",
                dtype="u32",
                lanes=_ignore_oob_lanes,
                vector=False,
            ),
            OperandSlot(
                "ignore_bytes_right",
                dtype="u32",
                lanes=_ignore_oob_lanes,
                vector=False,
            ),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared::cluster"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="global"),
            OperandSlot("size", dtype="u32"),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared"),
            OperandSlot("cta_mask", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared::cluster"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("size", dtype="u32"),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared"),
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="global"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("size", dtype="u32"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
            # "the 16-bit wide byteMask operand" -- the legacy helper bound it
            # "r", but that form was unreachable and ptxas rejects the 32-bit
            # register here.
            OperandSlot("byte_mask", dtype="u16", lanes=_cp_mask_lanes, vector=False),
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared::cluster"),
            OperandSlot("tmap", kind="addr", space="global", bracket="src"),
            OperandSlot("coords", dtype="s32", lanes=_tma_coords_lanes, bracket="src"),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared"),
            OperandSlot("cta_mask", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("tmap", kind="addr", space="global", bracket="src"),
            OperandSlot("coords", dtype="s32", lanes=_tma_coords_lanes, bracket="src"),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("tmap", kind="addr", space="global", bracket="dst"),
            OperandSlot("coords", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("tmap", kind="addr", space="global", bracket="dst"),
            OperandSlot("coords", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
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
            OperandSlot("tmap", kind="addr", space="global", bracket="src"),
            OperandSlot("coords", dtype="s32", lanes=_tma_coords_lanes, bracket="src"),
            OperandSlot("cache_policy", dtype="u64", lanes=_tma_cache_lanes, vector=False),
        ),
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
    # ------------------------------------------------------------------
    # PTX ISA 9.7.14 — Parallel Synchronization and Communication Instructions
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
    # The `.red` lines are registered below, after the plain ones.
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
            operands=(OperandSlot("a", dtype="u32"),),
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
            OperandSlot("a", dtype="u32"),
            OperandSlot("b", dtype="u32"),
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
            OperandSlot("a", dtype="u32"),
            OperandSlot("b", dtype="u32"),
        ),
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
            OperandSlot("a", dtype="u32"),
            OperandSlot("b", dtype="u32"),
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
            OperandSlot("a", dtype="u32"),
            OperandSlot("b", dtype="u32"),
        ),
    ),
    *[
        InstructionEntry(  # barrier.cluster.arrive{.sem}{.aligned} / .wait{.acquire}{.aligned}
            name=f"barrier_cluster_{act}",
            # Shares the `barrier` mnemonic with 9.7.14.1 so the surface reads
            # `T.ptx.barrier.cluster.arrive(...)`; `cluster` is the token that
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
        operands=(OperandSlot("membermask", dtype="u32"),),
    ),
    # The reduction forms of bar / barrier (PTX ISA 9.7.14.1), which combine a
    # predicate across the barrier's threads instead of only waiting:
    #   bar{.cta}.red.popc.u32       d, a{, b}, {!}c;
    #   bar{.cta}.red.op.pred        p, a{, b}, {!}c;   .op = {.and, .or}
    #   barrier{.cta}.red...{.aligned} likewise
    # Four shapes per mnemonic: the count operand `b` is a separate syntax
    # line, and `popc` returns how many predicates were true while `.and`/`.or`
    # return a predicate. `.xor` is not offered on either -- ptxas: "Illegal
    # reduction operation for instruction 'bar.red.pred'".
    # NOT REGISTERED: the `{!}c` negation, per the policy stated at `setp`.
    *[
        InstructionEntry(
            name=f"{mnem}_red_{kind}" + ("_count" if count else ""),
            mnemonic=mnem,
            slots=(
                ModifierSlot("cta", ("cta",), optional=True),
                ModifierSlot("action", ("red",)),
                ModifierSlot("op", ("popc",) if kind == "popc" else ("and", "or")),
                *(
                    (ModifierSlot("aligned", ("aligned",), optional=True),)
                    if mnem == "barrier"
                    else ()
                ),
                ModifierSlot("type", ("u32",) if kind == "popc" else ("pred",)),
            ),
            orders_memory=True,
            operands=(
                OperandSlot("d", rw="w", dtype="u32" if kind == "popc" else "pred"),
                OperandSlot("a", dtype="u32"),  # barrier id
                *((OperandSlot("b", dtype="u32"),) if count else ()),  # thread count
                OperandSlot("c", dtype="pred"),  # the predicate being reduced
            ),
        )
        for mnem in ("bar", "barrier")
        for kind in ("popc", "pred")
        for count in (False, True)
    ],
    # vote.sync per PTX ISA 9.7.14.10: reduce a predicate across a warp. The
    # `.ballot` line returns one bit per lane instead of a single answer, so it
    # is a second entry rather than a fourth `.mode` token.
    # NOT REGISTERED: `vote` without `.sync` (9.7.14.9), deprecated in PTX ISA
    # 6.0 and rejected outright by ptxas at sm_70 and higher, exactly as the
    # non-sync `shfl` is; and the `{!}a` negation, per the usual policy.
    *[
        InstructionEntry(
            name=name,
            mnemonic="vote.sync",
            slots=(
                ModifierSlot("mode", modes),
                ModifierSlot("type", (ty,)),
            ),
            operands=(
                OperandSlot("d", rw="w", dtype="pred" if ty == "pred" else None),
                OperandSlot("a", dtype="pred"),
                OperandSlot("membermask", dtype="u32"),
            ),
        )
        for name, modes, ty in (
            ("vote_sync", ("all", "any", "uni"), "pred"),
            ("vote_sync_ballot", ("ballot",), "b32"),
        )
    ],
    # match.sync per PTX ISA 9.7.14.11: which lanes of the warp hold the same
    # value. `.all` optionally reports, through a second predicate, whether
    # every lane agreed -- `.any` has no such answer to give, and ptxas says so
    # ("Predicate output not allowed for instruction 'match.any'"), so the
    # pipe form belongs to `.all` alone.
    # The destination is a lane mask, so it is `.b32` whatever the type of the
    # value being compared.
    *[
        InstructionEntry(
            name=name,
            # The mnemonic is `match`: the ISA spells `match.any.sync.b32`, so
            # `.sync` follows the mode rather than preceding it. (ptxas takes
            # both orders, but the syntax line is what the table registers.)
            mnemonic="match",
            slots=(
                ModifierSlot("mode", (mode,)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("type", ("b32", "b64")),
            ),
            cert_arch="sm_90",  # the subsection's floor is sm_70
            operands=(
                # The lane mask is `b32i`: a float register is rejected here
                # (measured), while the value being compared -- `a`, on the
                # entry's own type slot -- takes one happily.
                OperandSlot("d", rw="w", dtype="b32i", pipe="dp" if pq else None),
                *((OperandSlot("p", rw="w", dtype="pred", pipe="dp"),) if pq else ()),
                OperandSlot("a"),
                OperandSlot("membermask", dtype="u32"),
            ),
        )
        for name, mode, pq in (
            ("match_any_sync", "any", False),
            ("match_all_sync", "all", False),
            ("match_all_sync_p", "all", True),
        )
    ],
    # activemask per PTX ISA 9.7.14.12: the one instruction in the table that
    # writes a destination and reads nothing at all.
    InstructionEntry(
        name="activemask",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(OperandSlot("d", rw="w"),),
    ),
    # elect.sync per PTX ISA 9.7.14.15: pick one leader lane out of the member
    # mask. Its `d|p` is not optional the way match's is -- ptxas answers
    # "Predicate output expected for instruction 'elect'" to a bare
    # destination -- so this family has exactly one shape, and it is the pipe.
    InstructionEntry(
        name="elect_sync",
        mnemonic="elect.sync",
        slots=(),
        cert_arch="sm_90",
        operands=(
            # `b32i`: ptxas answers "Arguments mismatch for instruction
            # 'elect'" to a float register here, though `.b32` would admit one.
            OperandSlot("d", rw="w", dtype="b32i", pipe="dp"),
            OperandSlot("p", rw="w", dtype="pred", pipe="dp"),
            OperandSlot("membermask", dtype="u32"),
        ),
    ),
    # redux.sync per PTX ISA 9.7.14.13: reduce a value across the warp.
    *[
        InstructionEntry(
            name=name,
            mnemonic="redux.sync",
            slots=(
                ModifierSlot("op", ops),
                ModifierSlot("type", types),
            ),
            cert_arch="sm_90",  # the integer lines need sm_80
            operands=(
                OperandSlot("d", rw="w", dtype=dt),
                OperandSlot("a", dtype=dt),
                OperandSlot("membermask", dtype="u32"),
            ),
        )
        # The ISA writes the arithmetic and bitwise reductions as two syntax
        # lines, and they really are two: the arithmetic one carries the
        # signedness in its type, while the bitwise one is untyped. They also
        # differ in register class -- ptxas takes no float register on the
        # `.b32` line in either position, so that line's operands say `b32i`.
        for name, ops, types, dt in (
            ("redux_sync", ("add", "min", "max"), ("u32", "s32"), None),
            ("redux_sync_bitwise", ("and", "or", "xor"), ("b32",), "b32i"),
        )
    ],
    # The `.f32` line of the same subsection, which adds `{.abs}{.NaN}` and is
    # scoped to one architecture family: ptxas answers "Instruction
    # 'redux.f32' not supported on .target 'sm_100'" and takes it only at
    # sm_100a, so it is certified there rather than beside its integer sibling.
    InstructionEntry(
        name="redux_sync_f32",
        mnemonic="redux.sync",
        slots=(
            ModifierSlot("op", ("min", "max")),
            ModifierSlot("abs", ("abs",), optional=True),
            ModifierSlot("nan", ("NaN",), optional=True),
            ModifierSlot("type", ("f32",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("a"),
            OperandSlot("membermask", dtype="u32"),
        ),
    ),
    # fence / membar per PTX ISA 9.7.14.4, griddepcontrol per 9.7.14.14.
    #
    # These name no address yet constrain everyone else's memory order, so they
    # set `orders_memory=True` -- see InstructionEntry for why `asm volatile`
    # alone is not enough. Each syntax line whose token sequence differs is its
    # own entry, all sharing the `fence` mnemonic, so the surface stays
    # `T.ptx.fence...` and the chain narrows on the tokens themselves.
    # NOT REGISTERED: the `.sync_restrict` lines and the fabric-proxy line
    # (fixed token sequences with no users yet), and the deprecated `membar`
    # spellings, which the ISA itself marks as the old style for `fence`.
    InstructionEntry(  # fence{.sem}.scope;
        name="fence",
        slots=(
            ModifierSlot("sem", ("sc", "acq_rel", "acquire", "release"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES),
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
            ModifierSlot("scope", _ATOM_SCOPES),
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
            ModifierSlot("scope", _ATOM_SCOPES),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            # "The only supported value for the size operand is 128, which must
            # be a constant integer literal" -- ISA 9.7.14.4.
            OperandSlot("size", kind="imm", literal="128"),
        ),
    ),
    # The ISA permits @p on this instruction; ptx does not, because it writes a
    # destination -- see InstructionEntry.has_dst for why that needs a "+"
    # constraint first.
    InstructionEntry(
        name="atom",
        slots=(
            ModifierSlot("sem", _ATOM_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
        ),
    ),
    # atom's other syntax lines (PTX ISA 9.7.14.5), each its own entry because
    # each differs in shape or in type domain from the `.op` line above.
    #
    # `.cas` compares and swaps, so it takes a fourth value; it is also the one
    # atom line with no `{.level::cache_hint}` (ptxas rejects the qualifier
    # there) and the only one that reaches down to `.b16`.
    InstructionEntry(
        name="atom_cas",
        mnemonic="atom",
        slots=(
            ModifierSlot("sem", _ATOM_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", ("cas",)),
            ModifierSlot("type", ("b16", "b32", "b64", "b128")),
        ),
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("compare"),
            OperandSlot("value"),
        ),
    ),
    # `.exch` just writes, so it keeps the plain shape; its type line starts at
    # 32 bits (ptxas: "Arguments mismatch" for `.b16`) and reaches `.b128`.
    InstructionEntry(
        name="atom_exch",
        mnemonic="atom",
        slots=(
            ModifierSlot("sem", _ATOM_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", ("exch",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("type", ("b32", "b64", "b128")),
        ),
        check=_check_cache_hint,
        operands=(
            OperandSlot("d", rw="w"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
        ),
    ),
    # The half-precision add lines of atom and red. `.noftz` is not optional on
    # them -- ptxas: "'.noftz' modifier required for instruction 'atom' with
    # type '.f16'" -- so it is a single-choice slot rather than an optional one.
    *[
        InstructionEntry(
            name=f"{mnem}_half",
            mnemonic=mnem,
            slots=(
                ModifierSlot("sem", _ATOM_SEM if mnem == "atom" else _RED_SEM, optional=True),
                ModifierSlot("scope", _ATOM_SCOPES, optional=True),
                ModifierSlot("space", _ATOM_SPACES, optional=True),
                ModifierSlot("op", ("add",)),
                ModifierSlot("noftz", ("noftz",)),
                ModifierSlot("cache", ("L2::cache_hint",), optional=True),
                ModifierSlot("type", (*_ATOM_VEC_HALF, *_HALF_X2)),
            ),
            check=_check_cache_hint,
            operands=(
                *((OperandSlot("d", rw="w"),) if mnem == "atom" else ()),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("value"),
                OperandSlot(
                    "cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False
                ),
            ),
        )
        for mnem in ("atom", "red")
    ],
    # atom's two vector lines. Both operands are register groups, so this is a
    # different shape again; `_check_atom_vec` holds which widths pair with
    # which element. `.f32` has no `.noftz` (there is no half to flush), which
    # is why the two lines are two entries rather than one.
    # The ISA's own examples spell these tokens in a different order than its
    # syntax line (`atom.global.v8.f16.max.noftz` against
    # `atom{...}.op.noftz{.cache}.vec.type`); ptxas accepts both, and the
    # normative syntax line is what is registered here.
    *[
        InstructionEntry(
            name=name,
            mnemonic="atom",
            slots=(
                ModifierSlot("sem", _ATOM_SEM, optional=True),
                ModifierSlot("scope", _ATOM_SCOPES, optional=True),
                ModifierSlot("space", ("global",), optional=True),
                ModifierSlot("op", ops),
                *((ModifierSlot("noftz", ("noftz",)),) if noftz else ()),
                ModifierSlot("cache", ("L2::cache_hint",), optional=True),
                ModifierSlot("vec", ("v2", "v4", "v8") if noftz else ("v2", "v4")),
                ModifierSlot("type", types),
            ),
            check=_check_atom_vec,
            operands=(
                OperandSlot("d", rw="w", lanes=_vec_lanes),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("value", lanes=_vec_lanes),
                OperandSlot(
                    "cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False
                ),
            ),
        )
        for name, ops, noftz, types in (
            ("atom_vec_half", ("add", "min", "max"), True, (*_ATOM_VEC_HALF, *_HALF_X2)),
            ("atom_vec_f32", ("add",), False, ("f32",)),
        )
    ],
    # red / atom scalar `.op` forms per PTX ISA 9.7.14.6 and 9.7.14.5.
    InstructionEntry(
        name="red",
        slots=(
            ModifierSlot("sem", _RED_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
            OperandSlot("cache_policy", dtype="u64", lanes=_present_lanes("cache"), vector=False),
        ),
    ),
    InstructionEntry(  # griddepcontrol.action;
        name="griddepcontrol",
        slots=(ModifierSlot("action", ("launch_dependents", "wait")),),
        orders_memory=True,
        operands=(),
    ),
    # mbarrier, per the PTX ISA 9.7.14.16 chapter: init 9.7.14.16.12, inval
    # 9.7.14.16.13, expect_tx 9.7.14.16.14, complete_tx 9.7.14.16.15, arrive
    # 9.7.14.16.16, arrive_drop 9.7.14.16.17, test_wait / try_wait 9.7.14.16.19,
    # pending_count 9.7.14.16.20.
    #
    # The arrive lines come in a `state`-returning form and a sink form
    # (`_, [addr]`); the sink is what a void helper can express, so `_` is an
    # ISA-fixed immediate the way `st.bulk`'s initval is. That also leaves the
    # instruction without a destination, so `pred=` works.
    #
    # This is the same ISA facility `OperandSlot.sinkable` models elsewhere,
    # spelled differently on purpose: here the sink is not a choice. The
    # state-returning form needs a destination nothing reads today, so only the
    # sunk spelling is registered, and an operand that is ALWAYS `_` is a fixed
    # instruction-text value -- which is what a literal imm is. `sinkable`
    # would let a caller ask for the other form, and there is no other form
    # here. Unify the two the day the state-returning line is registered. arrive_drop's five
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
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("count", dtype="u32"),
        ),
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
                OperandSlot("state", kind="imm", literal="_"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
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
                OperandSlot("state", kind="imm", literal="_"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("count", dtype="u32"),
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
                OperandSlot("state", kind="imm", literal="_"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("tx_count", dtype="u32"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    # mbarrier.<act>.noComplete{.release.cta}{.shared{::cta}}.b64 state, [addr], count;
    *[
        InstructionEntry(
            # This line has no sink form, so `state` is a real register result.
            # It rides rw="rw" rather than "w": "+" keeps the old value live
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
                # `b64i`, not `b64`: the bit-type dtype axis would offer an f64
                # carrier, and ptxas rejects an .f64 register as the state operand
                # ("Arguments mismatch for instruction 'mbarrier.arrive'").
                OperandSlot("state", rw="rw", dtype="b64i"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("count", dtype="u32"),
            ),
        )
        for act in ("arrive", "arrive_drop")
    ],
    # mbarrier.test_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity;
    # mbarrier.try_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity, timeHint;
    #
    # waitComplete is a `.pred` result -- rw="w", dtype="pred", the in-block selp
    # materialization. try_wait supports both PTX arities: timeHint is optional.
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
                OperandSlot("wait_complete", rw="w", dtype="pred"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("phase", dtype="u32"),
                *((OperandSlot("time_hint", dtype="u32"),) if act == "try_wait" else ()),
            ),
        )
        for act in ("test_wait", "try_wait")
    ],
    InstructionEntry(
        name="mbarrier_try_wait_parity_no_hint",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("try_wait",)),
            ModifierSlot("parity", ("parity",)),
            ModifierSlot("sem", ("acquire", "relaxed"), optional=True),
            ModifierSlot("scope", ("cta", "cluster"), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        check=_check_mbarrier_sem_scope,
        operands=(
            OperandSlot("wait_complete", rw="w", dtype="pred"),
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("phase", dtype="u32"),
        ),
    ),
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
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("tx_count", dtype="u32"),
            ),
        )
        for act in ("expect_tx", "complete_tx")
    ],
    InstructionEntry(  # mbarrier.inval{.shared{::cta}}.b64 [addr];
        name="mbarrier_inval",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("inval",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(OperandSlot("addr", kind="addr", allow_imm_offset=True),),
    ),
    # The state-returning arrive lines (PTX ISA 9.7.14.16.16 / .17). The
    # entries above bake `_` into the text, which is the only spelling the
    # `.shared::cluster` lines offer; these return the phase state instead, and
    # the ISA gives them `.shared{::cta}` alone.
    *[
        InstructionEntry(
            name=f"mbarrier_{act}{suffix}_state",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                *((ModifierSlot("expect_tx", ("expect_tx",)),) if suffix == "_expect_tx" else ()),
                ModifierSlot("sem", ("release", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                # The phase state is an opaque 64-bit token: ptxas takes an
                # integer register of either signedness and rejects a float one
                # ("Arguments mismatch"), which is what `b64i` names.
                OperandSlot("state", rw="w", dtype="b64i"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                *((OperandSlot("count", dtype="u32"),) if count else ()),
            ),
        )
        for act in ("arrive", "arrive_drop")
        # Three shapes per action, mirroring the `_` family above: bare, with a
        # thread count, and with a transaction count. The noComplete line needs
        # no twin -- it has no `_` spelling, so `mbarrier_*_no_complete` above
        # is already the state-returning entry.
        for suffix, count in (("", False), ("_count", True), ("_expect_tx", True))
    ],
    # The non-parity waits (PTX ISA 9.7.14.16.19), which take the state token
    # an arrive returned rather than a phase parity. try_wait may also carry a
    # sleep hint, which is a separate syntax line and so a separate entry.
    # NOT REGISTERED: the `.phase_type::*` qualifiers and the
    # `waitComplete|reportPredicate{, reportValue}` report forms -- both are
    # PTX ISA 9.3, and ptxas 13.2 (9.2) rejects the token outright.
    *[
        InstructionEntry(
            name=f"mbarrier_{act}{'_hint' if hint else ''}",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sem", ("acquire", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("wait_complete", rw="w", dtype="pred"),
                OperandSlot("addr", kind="addr", allow_imm_offset=True),
                OperandSlot("state", dtype="b64i"),  # the token an arrive returned
                *((OperandSlot("time_hint", dtype="u32"),) if hint else ()),
            ),
        )
        for act, hint in (("test_wait", False), ("try_wait", False), ("try_wait", True))
    ],
    # tensormap.cp_fenceproxy per PTX ISA 9.7.14.17: copy a tensor-map object
    # from shared to global and fence the tensormap proxy over it, in one
    # instruction. `size` is the literal 128 -- ptxas both refuses a register
    # there ("Arguments mismatch") and names the only legal value ("unexpected
    # value '64', expected to be 128"), so it is table-owned.
    InstructionEntry(
        name="tensormap_cp_fenceproxy",
        mnemonic="tensormap.cp_fenceproxy",
        slots=(
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("proxy", ("tensormap::generic",)),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", _ATOM_SCOPES),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", kind="addr", allow_imm_offset=True, space="global"),
            OperandSlot("src_mem", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("size", kind="imm", literal="128"),
        ),
    ),
    # red.async per PTX ISA 9.7.14.7: an asynchronous reduction whose
    # completion is signalled through an mbarrier. The ISA writes one syntax
    # line per op group; `_check_red_async` is that grid.
    InstructionEntry(
        name="red_async",
        mnemonic="red.async",
        slots=(
            ModifierSlot("sem", ("relaxed",)),
            ModifierSlot("scope", ("cluster",)),
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("op", tuple(_RED_ASYNC_OPS)),
            ModifierSlot("type", ("u32", "s32", "u64", "b32")),
        ),
        check=_check_red_async,
        cert_arch="sm_90",
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True),
        ),
    ),
    # The release line of the same subsection, which reduces straight into
    # global memory and needs no mbarrier. Like st.async's release line it is
    # sm_100, and `.mmio` is system-scoped only.
    # NOT REGISTERED: `multimem.red.async` (9.7.14.8) -- ptxas 13.2 rejects the
    # mnemonic at sm_90 and sm_100 alike, as it does `multimem.st.async`.
    InstructionEntry(
        name="red_async_release",
        mnemonic="red.async",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("gpu", "sys")),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("op", ("add",)),
            ModifierSlot("type", ("u32", "s32", "u64", "s64")),
        ),
        check=_check_st_async_rel,
        cert_arch="sm_100",
        operands=(
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("value"),
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
    # (The `cp.async` ca/cg copy lines this note once excluded are registered
    # in the 9.7.9 group above, ignore-src operand and all.)
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
        operands=(OperandSlot("addr", kind="addr", allow_imm_offset=True),),
    ),
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
            OperandSlot("count", rw="w", dtype="u32"),
            OperandSlot("state", dtype="u64"),
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
            OperandSlot("addr", kind="addr", allow_imm_offset=True),
            OperandSlot("mbar", kind="addr", allow_imm_offset=True),
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
    # The `.v4` line's sink spelling IS registered: the ISA's own example is
    # "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    # {xctaid, _, _, _}, handle;", and "The contents of the 4th element are
    # unspecified" makes the discard the point rather than an optimization.
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
            OperandSlot("p", rw="w", dtype="pred"),
            OperandSlot("response", dtype="b128"),
        ),
    ),
    InstructionEntry(
        # `d` is rw="rw", not "w": the caller seeds it with a sentinel and
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
            OperandSlot("d", rw="rw", dtype="b32"),
            OperandSlot("response", dtype="b128"),
        ),
    ),
    InstructionEntry(
        # The `.v4` line, which is also the `{::dimension}`-omitted spelling
        # (see the family comment). `d` is one operand of four registers, not
        # four operands: PTX writes it as the brace group `{xdim, ydim, zdim,
        # ignr}`. rw="rw" for the same reason as the per-dimension entry --
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
            # "The contents of the 4th element are unspecified", and the
            # ISA's example discards three of the four: `_` is registered here
            # rather than described.
            OperandSlot("d", rw="rw", dtype="b32", lanes=4, sinkable=True),
            OperandSlot("response", dtype="b128"),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.15 — Warp Level Matrix Multiply-Accumulate Instructions
    # ------------------------------------------------------------------
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
            OperandSlot("d", rw="w", dtype="f32", lanes=_mma_lanes("d")),
            OperandSlot("a", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", dtype="f32", lanes=_mma_lanes("c")),
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
            OperandSlot("d", rw="w", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", dtype="u32", lanes=_mma_lanes("c")),
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
            OperandSlot("d", rw="w", dtype="f32", lanes=_mma_lanes("d")),
            OperandSlot("a", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", dtype="u32", lanes=_mma_lanes("c")),
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
            OperandSlot("d", rw="w", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", dtype="u32", lanes=_mma_lanes("c")),
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
            OperandSlot("d", rw="w", dtype="f64", lanes=_mma_lanes("d")),
            OperandSlot("a", dtype="f64", lanes=_mma_lanes("a")),
            OperandSlot("b", dtype="f64", lanes=_mma_lanes("b")),
            OperandSlot("c", dtype="f64", lanes=_mma_lanes("c")),
        ),
    ),
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
                OperandSlot("d", rw="w", dtype="u32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", dtype="u32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", dtype="u32"),
                OperandSlot("f", kind="imm", choices=selector),
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
                OperandSlot("d", rw="w", dtype="u32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", dtype="u32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", dtype="u32"),
                OperandSlot("f", kind="imm", choices=selector),
            ),
        )
        for suffix, shapes, selector, check in (
            ("_pair", ("m16n8k32", "m16n8k64"), ("0", "1"), _check_mma_sp_int_pair),
            ("_all", ("m16n8k64", "m16n8k128"), ("0",), _check_mma_sp_int_all),
        )
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
            OperandSlot("r", rw="w", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", kind="addr", allow_imm_offset=True),
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
            OperandSlot("r", rw="w", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", kind="addr", allow_imm_offset=True),
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
            OperandSlot("r", rw="w", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", kind="addr", allow_imm_offset=True),
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
            OperandSlot("p", kind="addr", allow_imm_offset=True),
            OperandSlot("r", dtype="b32", lanes=_matrix_num_lanes),
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
            OperandSlot("p", kind="addr", allow_imm_offset=True),
            OperandSlot("r", dtype="b32", lanes=_matrix_num_lanes),
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
                OperandSlot("d", rw="w", dtype="f32", lanes=_mma_sp_lanes("d")),
                OperandSlot("a", dtype="u32", lanes=_mma_sp_lanes("a")),
                OperandSlot("b", dtype="u32", lanes=_mma_sp_lanes("b")),
                OperandSlot("c", dtype="f32", lanes=_mma_sp_lanes("c")),
                OperandSlot("e", dtype="u32"),
                OperandSlot("f", kind="imm", choices=selector),
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
    # ------------------------------------------------------------------
    # PTX ISA 9.7.16 — Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions
    # ------------------------------------------------------------------
    # wgmma.mma_async per PTX ISA 9.7.16.5.2 (sm_90a). Six type groups, each
    # with an ss line (both A and B from shared memory, named by 64-bit matrix
    # descriptors) and an rs line (A from registers -- always four .b32, see
    # the fragment note above -- which also drops imm-trans-a). Like mma, one
    # entry per accumulator register type so every operand dtype is pinned.
    #
    # The accumulator is read and written in place: D = A*B + D, so `d` is
    # rw="rw", the "+" constraint. The trailing arguments all live in the
    # instruction text as caller immediates:
    #   - scale-d: "the operation of the form D = A*B is issued when the input
    #     predicate argument scale-d is false". The syntax calls it a
    #     predicate, but ptxas accepts the literals 0 and 1 in that position
    #     (certified), and every call site passes a compile-time constant --
    #     so it is a choices immediate, not a runtime operand. (The legacy
    #     helper burned a setp + predicate register on it per call.)
    #   - imm-scale-a/b: "the valid values ... are -1 and 1". The legacy
    #     helper emitted 0 for "no negate", outside the ISA's domain; ptx
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
                OperandSlot("d", rw="rw", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", kind="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", kind="imm", choices=("0", "1")),
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
                OperandSlot("d", rw="rw", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", kind="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", kind="imm", choices=("0", "1")),
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
                OperandSlot("d", rw="rw", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", kind="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", kind="imm", choices=("0", "1")),
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
                OperandSlot("d", rw="rw", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
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
                OperandSlot("d", rw="rw", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
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
                OperandSlot("d", rw="rw", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
                OperandSlot("scale_a", kind="imm", choices=("-1", "1")),
                OperandSlot("scale_b", kind="imm", choices=("-1", "1")),
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
                OperandSlot("d", rw="rw", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
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
                OperandSlot("d", rw="rw", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("scale_d", kind="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
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
        operands=(OperandSlot("group", kind="imm", choices=tuple(str(n) for n in range(8))),),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.17 — TensorCore 5th Generation Family Instructions
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
            OperandSlot("dst", kind="addr", allow_imm_offset=True, space="shared::cta"),
            OperandSlot("ncols", dtype="u32"),
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
            OperandSlot("taddr", dtype="u32"),
            OperandSlot("ncols", dtype="u32"),
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
            OperandSlot("r", rw="w", dtype="b32", lanes=_tcgen05_ldst_lanes),
            OperandSlot("taddr", kind="addr", space="tmem"),
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
            OperandSlot("r", rw="w", dtype="b32", lanes=_tcgen05_ldst_lanes),
            OperandSlot("taddr", kind="addr", space="tmem"),
            OperandSlot("imm_half_splitoff", kind="imm"),
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
            OperandSlot("taddr", kind="addr", space="tmem"),
            OperandSlot("r", dtype="b32", lanes=_tcgen05_ldst_lanes),
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
            OperandSlot("taddr", kind="addr", space="tmem"),
            OperandSlot("imm_half_splitoff", kind="imm"),
            OperandSlot("r", dtype="b32", lanes=_tcgen05_ldst_lanes),
        ),
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
            OperandSlot("taddr", kind="addr", space="tmem"),
            OperandSlot("s_desc", dtype="b64"),
        ),
    ),
    # tcgen05.mma per PTX ISA 9.7.17.10.9.1 (sm_100a): D = A*B + D where D
    # lives in Tensor Memory (an address, not registers -- so the instruction
    # has no dst and @p stays available). Two entries per family split on the
    # A operand's home: a 64-bit shared-memory descriptor ("ss") or a tmem
    # address ("ts"), the same split the syntax lines draw. enable-input-d is
    # a runtime .pred argument -- dtype="pred", the in-block setp
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
    # - block_scale's scale_vec-omitted spelling without a .block16/.block32
    #   size: no call site uses that form.
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
                OperandSlot("d_tmem", kind="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", kind="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("idesc", dtype="u32"),
                OperandSlot(
                    "disable_output_lane",
                    dtype="u32",
                    lanes=_tcgen05_mma_mask_lanes,
                ),
                OperandSlot("enable_input_d", dtype="pred"),
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
                OperandSlot("d_tmem", kind="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", kind="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("idesc", dtype="u32"),
                OperandSlot("sfa_tmem", kind="addr", space="tmem"),
                OperandSlot("sfb_tmem", kind="addr", space="tmem"),
                OperandSlot("enable_input_d", dtype="pred"),
            ),
        )
        for form in ("ss", "ts")
    ],
    *[
        InstructionEntry(  # block-scaled with an explicit scale block size
            name=f"tcgen05_mma_block_scale_block_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
                ModifierSlot("kind", ("kind::mxf8f6f4", "kind::mxf4", "kind::mxf4nvf4")),
                ModifierSlot("block_scale", ("block_scale",)),
                ModifierSlot("block_size", ("block16", "block32")),
            ),
            cert_arch="sm_100a",
            check=_check_tcgen05_mma_block_scale_block,
            operands=(
                OperandSlot("d_tmem", kind="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", kind="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("idesc", dtype="u32"),
                OperandSlot("sfa_tmem", kind="addr", space="tmem"),
                OperandSlot("sfb_tmem", kind="addr", space="tmem"),
                OperandSlot("enable_input_d", dtype="pred"),
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
                OperandSlot("d_tmem", kind="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", kind="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", dtype="u64"),
                OperandSlot("idesc", dtype="u32"),
                OperandSlot("enable_input_d", dtype="pred"),
                OperandSlot("zero_col_mask", dtype="u64"),
            ),
        )
        for form in ("ss", "ts")
    ],
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
        operands=(
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared::cluster"),
        ),
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
            OperandSlot("mbar", kind="addr", allow_imm_offset=True, space="shared::cluster"),
            OperandSlot("mask", dtype="u16"),
        ),
    ),
    # ------------------------------------------------------------------
    # PTX ISA 9.7.20 — Miscellaneous Instructions
    # ------------------------------------------------------------------
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
            OperandSlot("nreg", kind="imm", choices=tuple(str(n) for n in range(24, 257, 8))),
        ),
    ),
]


def _validate_imm_offset_slots(entries) -> None:
    """Reject capability bits on operand classes that cannot spell ``[addr+imm]``."""
    errors = []
    for entry in entries:
        for slot in entry.operands:
            if not slot.allow_imm_offset:
                continue
            reasons = []
            if slot.kind != "addr":
                reasons.append(f"kind={slot.kind!r}, expected 'addr'")
            if slot.bracket is not None:
                reasons.append("is a composite bracket member")
            if slot.space == "tmem" or (
                slot.space is None
                and any(
                    modifier.name == "space" and "tmem" in modifier.choices
                    for modifier in entry.slots
                )
            ):
                reasons.append("is a tmem address")
            if reasons:
                errors.append(f"{entry.name}.{slot.name}: " + "; ".join(reasons))
    if errors:
        raise ValueError("invalid allow_imm_offset slots:\n  " + "\n  ".join(errors))


_validate_imm_offset_slots(_ENTRIES)
TABLE: dict[str, InstructionEntry] = {e.name: e for e in _ENTRIES}
# Keying by name silently drops a duplicate, and a dropped entry is an ISA line
# that stops being reachable. Two entries never legitimately share a name.
assert len(TABLE) == len(_ENTRIES), "duplicate InstructionEntry name in _ENTRIES"
