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
"""Generic engine for the ``T.ptx`` table-driven PTX dialect.

One engine interprets every :class:`~.table.InstructionEntry` — there is no
per-instruction generated or hand-written code:

- :func:`register_table` registers each family as a TVM Op
  (``tirx.ptx.<name>``) with effect/printer attrs, plus one generic codegen
  closure that renders the ``asm volatile`` helper from the table.
- :class:`PTXNamespace` (surfaced as ``T.ptx``) resolves attribute chains
  such as ``T.ptx.ld.global_.acquire.gpu.b32(addr)`` against the table:
  the first token names the family, every further token fills a modifier
  slot (order-free). Python keywords are escaped with a trailing underscore
  (``global_``); ``::`` is written as a double underscore (``shared__cta``).
  The string form ``T.ptx["st.weak.shared::cta.b32"]`` preserves exact PTX
  text.
- Modifiers travel as trailing positional string args of the traced Call
  (never ``Call.attrs`` — that would break TVMScript pretty-printing). Call
  arg layout: ``[operands..., pred?] [slot tokens ("" = omitted)]``; the
  codegen derives predication from the arg count. Destinations are ordinary
  leading operands, so a call is always a statement:
  ``T.ptx.ld.acquire.gpu.global_.b32(val, ptr)``.
"""

from tvm.backend.cuda.codegen.registry import register_codegen
from tvm.backend.cuda.codegen.utils import parse_str
from tvm.backend.cuda.op import cuda_cvta_generic_to_shared, cuda_func_call
from tvm.ir import Call
from tvm.ir.op import register_op_attr
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import const
from tvm.tirx.expr import BufferLoad, CallEffectKind, IntImm
from tvm.tirx.op import call_intrin, reinterpret

from .render import render_variant
from .table import (
    InstructionEntry,
    escape_token,
    mods,
    operand_dtypes,
    operand_layout,
    operand_space,
    operand_type,
    unescape_token,
)

# Every ptx call is a void statement, and RemoveNoOp deletes any Evaluate()
# whose value is <= kReadState, so kOpaque is what keeps the instruction alive.
# It is also the honest answer: "do not touch my instruction" is exactly the
# contract a hand-written PTX call wants.
_EFFECT_OPAQUE = CallEffectKind.Opaque.value
_EFFECT_PURE = CallEffectKind.Pure.value
_ADDR_OP_NAME = "tirx.ptx.addr"
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_INTEGER_DTYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)

# ---------------------------------------------------------------------------
# Registration (import time)
# ---------------------------------------------------------------------------


def register_table(table: dict[str, InstructionEntry]) -> None:
    """Register every table entry as a TVM Op + generic codegen."""
    for entry in table.values():
        # First attr call implicitly creates the Op registry entry. Effect
        # kind must exist before any side-effect analysis sees the op.
        register_op_attr(entry.op_name, "TCallEffectKind", _EFFECT_OPAQUE)
        # The printer name is the *surface* path a user can type, which is the
        # mnemonic (several `mov_*` entries all answer to `T.ptx.mov`), not the
        # table key. Reparsing re-dispatches on the operand shape.
        #
        # Escaped, because "a user can type it" is the whole requirement and
        # three PTX mnemonics are Python keywords: `and`, `or` and `not` (ISA
        # 9.7.8) print as `T.ptx.and_` and are read back by `unescape_token` in
        # `PTXNamespace.__getattr__`. The escape is the identity for every
        # other family, and `gen_stubs` already spells the attribute this way.
        family = escape_token(entry.family)
        register_op_attr(entry.op_name, "TScriptPrinterName", f"ptx.{family}", level=20)
        register_op_attr(entry.op_name, "TIRxOpCategory", "device_intrin")
        register_op_attr(entry.op_name, "TDeviceIntrinsicNamespace", "ptx")
        register_codegen(f"ptx.{entry.name}")(_make_codegen(entry))


def register_addr() -> None:
    """Register the pure address-expression op consumed by PTX instructions."""
    register_op_attr(_ADDR_OP_NAME, "TCallEffectKind", _EFFECT_PURE)
    register_op_attr(_ADDR_OP_NAME, "TScriptPrinterName", "ptx.addr", level=20)
    register_op_attr(_ADDR_OP_NAME, "TIRxOpCategory", "device_intrin")
    register_op_attr(_ADDR_OP_NAME, "TDeviceIntrinsicNamespace", "ptx")
    register_codegen("ptx.addr")(_unconsumed_addr_codegen)


def _unconsumed_addr_codegen(*_args):
    raise ValueError(
        "T.ptx.addr(...) must be consumed by a PTX address operand that supports "
        "immediate byte offsets"
    )


# ---------------------------------------------------------------------------
# Codegen (compile time): table -> asm volatile helper
# ---------------------------------------------------------------------------


def arg_dtype(value) -> str:
    """The TVM dtype one argument carries.

    One definition, used by both layers: trace time decides acceptance with it
    and codegen selects the helper with it, so the two cannot drift.
    """
    buf = getattr(value, "buffer", None)
    if buf is not None:
        return buf.dtype  # scalar or buffer element: the element type
    ty = getattr(value, "ty", None)
    return ty.dtype if isinstance(ty, PrimType) else type(value).__name__


def _is_addr_call(value) -> bool:
    return isinstance(value, Call) and getattr(value.op, "name", None) == _ADDR_OP_NAME


def _codegen_addr_offset(entry, slot, value) -> tuple[object, int]:
    """Unpack one nested ``tirx.ptx.addr`` call at CUDA codegen time."""
    if not slot.allow_imm_offset:
        raise ValueError(f"{entry.name}: operand '{slot.name}' does not support T.ptx.addr(...)")
    if len(value.args) != 2:
        raise ValueError("malformed tirx.ptx.addr call: expected base and byte_offset")
    base, offset = value.args
    if _is_addr_call(base):
        raise ValueError("T.ptx.addr(...) cannot be nested")
    if not isinstance(offset, IntImm) or arg_dtype(offset) not in _INTEGER_DTYPES:
        raise ValueError(
            f"{entry.name}: T.ptx.addr byte_offset must become a compile-time "
            "signed int32 constant before CUDA codegen (use an explicitly-unrolled loop)"
        )
    offset = int(offset)
    if not _INT32_MIN <= offset <= _INT32_MAX:
        raise ValueError(
            f"{entry.name}: T.ptx.addr byte_offset {offset} is outside signed int32 range"
        )
    return base, offset


def _make_codegen(entry: InstructionEntry):
    n_slots = len(entry.slots)

    def codegen(*args):
        # Layout: [operands..., pred?] [n_slots tokens] [pred marker]. The token
        # count is fixed per entry, so the tokens are parsed first and the
        # operand layout -- which may depend on them, a register group's length
        # being a function of the modifiers -- is looked up from them (memoized
        # per token combination in the table).
        # The marker is a comma-joined flag set ("pred", destination policy,
        # and/or "p<i>"); codegen only cares about @p and the destination
        # binding policy, since the register classes are already in the table.
        # See the arg-layout note in `_emit`.
        flags = parse_str(args[-1]).split(",")
        predicated = "pred" in flags
        preserve_dst = "keep" in flags
        tokens = [parse_str(a) for a in args[len(args) - n_slots - 1 : -1]]
        rest = list(args[: len(args) - n_slots - 1])  # operands, plus pred when present
        mod_map = mods(entry, tokens)
        layout = operand_layout(entry, mod_map)
        n_operands = sum(n for _, _, n in layout)
        # Sunk lanes left no argument behind, so the layout's operand indices
        # and the Call's argument positions no longer line up; `at` maps one to
        # the other. Everything below indexes through it.
        sunk = {int(f[1:]) for f in flags if f.startswith("s") and f != "sink"}
        at, pos = {}, 0
        for i in range(n_operands):
            if i not in sunk:
                at[i] = pos
                pos += 1
        n_present = pos
        sinks = frozenset(
            (slot.name, lane)
            for slot, i, lanes in layout
            for lane in range(lanes)
            if i + lane in sunk
        )

        # Which dtype each typed operand actually carries. It is already in the
        # Call -- the same channel PTX uses, where a register's type lives in its
        # .reg declaration rather than in the instruction text. An operand whose
        # length function resolved to zero has no argument to read; it still
        # needs its dtype slot filled (the render aligns dtypes with *every*
        # typed operand), so it reports its canonical dtype.
        def _slot_dtype(slot, i, n):
            # A dtype belongs to the operand, so any surviving lane reports it;
            # an operand with none left (zero lanes, or every lane sunk) falls
            # back to its canonical choice, which is the only one it can have
            # when there is nothing to read it from.
            live = [i + lane for lane in range(n) if i + lane not in sunk]
            if live and len(operand_dtypes(slot, mod_map)) > 1:
                return arg_dtype(rest[at[live[0]]])
            return operand_dtypes(slot, mod_map)[0]

        dtypes = tuple(_slot_dtype(slot, i, n) for slot, i, n in layout if slot.kind == "reg")
        # Caller-chosen immediates ride the Call until device codegen so an
        # explicitly-unrolled loop may specialize them. They must be IntImm by
        # this point, then are baked into the instruction text and NOT
        # forwarded to the helper (which has no parameter for them).
        imm_at = {i for slot, i, _ in layout if slot.kind == "imm"}
        imm_values = [rest[at[i]] for i in sorted(imm_at)]
        if any(not isinstance(value, IntImm) for value in imm_values):
            raise ValueError(
                f"{entry.name}: immediate operands must become compile-time constants "
                "before CUDA codegen (use an explicitly-unrolled loop)"
            )
        imms = tuple(str(int(value)) for value in imm_values)
        # ``tirx.ptx.addr`` is an expression only in the outer PTX call's IR.
        # The helper still receives the coerced base, while the signed byte
        # displacement becomes renderer metadata baked into ``[%N+imm]``.
        addr_offsets = []
        logical_addr_slot = 0
        for slot, i, lanes in layout:
            if slot.kind != "addr":
                continue
            if lanes != 1:
                raise AssertionError(
                    f"{entry.name}: address operand '{slot.name}' must occupy one register"
                )
            value = rest[at[i]]
            if _is_addr_call(value):
                base, offset = _codegen_addr_offset(entry, slot, value)
                rest[at[i]] = base
                if offset:
                    addr_offsets.append((logical_addr_slot, offset))
            logical_addr_slot += 1
        _, helper, source = render_variant(
            entry,
            tokens,
            predicated,
            dtypes,
            imms,
            sinks,
            preserve_dst=preserve_dst,
            addr_offsets=tuple(addr_offsets),
        )
        # Every helper is void; a destination is an ordinary argument, printed
        # by the C codegen as the lvalue it binds the reference parameter to.
        # A predicate rides after the operands, so everything past n_operands
        # is forwarded as-is.
        forwarded = [rest[at[i]] for i in range(n_operands) if i not in imm_at and i not in sunk]
        forwarded += list(rest[n_present:])
        return cuda_func_call(helper, *forwarded, source_code=source)

    return codegen


# ---------------------------------------------------------------------------
# Trace time: operand coercion + Call emission
# ---------------------------------------------------------------------------


class _Sink:
    """``T.ptx.SINK`` -- the ISA's sink symbol ``_`` at one destination lane.

    PTX writes the discard at the operand position (`mov.b64 {_, %0}, %1;`),
    so this is written there too. It is a per-call choice rather than a
    property of the instruction, which is why the resulting mask is part of
    the variant: a sunk lane has no C parameter, so it is a different helper.

    Only a lane of a `sinkable` operand accepts it; anywhere else is a
    trace-time error. Which operands those are is read off each syntax line
    and is not a property of the direction: `ld` sinks an element it does not
    write, `st` sinks one it does not store.
    """

    __slots__ = ()

    def __repr__(self):
        return "T.ptx.SINK"


SINK = _Sink()


class AddrArg:
    """Temporary trace-time wrapper for ``T.ptx.addr(base, byte_offset)``.

    It deliberately is not a TIR expression. An eligible outer PTX address
    operand supplies the state space, coerces ``base``, and only then creates
    the nested pure ``tirx.ptx.addr`` Call.
    """

    __slots__ = ("base", "byte_offset")

    def __init__(self, base, byte_offset):
        if isinstance(base, AddrArg):
            raise ValueError("T.ptx.addr(...) cannot be nested")
        self.base = base
        self.byte_offset = _coerce_addr_offset(byte_offset)

    def __repr__(self):
        return f"T.ptx.addr({self.base!r}, {self.byte_offset!r})"


def _coerce_addr_offset(value):
    """Validate the byte displacement while preserving unrollable expressions."""
    if isinstance(value, bool):
        raise ValueError("T.ptx.addr byte_offset must be a signed int32 integer, not bool")
    if isinstance(value, IntImm):
        if arg_dtype(value) not in _INTEGER_DTYPES:
            raise ValueError(
                f"T.ptx.addr byte_offset must be a signed int32 integer, got {arg_dtype(value)}"
            )
        value = int(value)
    if isinstance(value, int):
        if not _INT32_MIN <= value <= _INT32_MAX:
            raise ValueError(f"T.ptx.addr byte_offset {value} is outside signed int32 range")
        return const(value, "int32")
    dtype = arg_dtype(value)
    if dtype in _INTEGER_DTYPES:
        # An explicitly-unrolled loop may specialize this expression later.
        # CUDA codegen performs the final IntImm and range checks.
        return value
    raise ValueError(f"T.ptx.addr byte_offset must be a scalar integer expression, got {dtype}")


class PredArg:
    """``T.ptx.pred(x)`` -- "this operand is a ``.pred`` register".

    A trace-time tag, never a conversion: ``value`` is handed to the helper
    untouched, and the ``setp`` that turns the carrier into a predicate is
    emitted inside the asm block exactly as before. The tag exists for
    dispatch. PTX distinguishes a predicate operand from an integer one by the
    declared register class, but both cross the C boundary as a uint32, so
    without the tag the two `cp.async` syntax lines that differ only in
    ``{, src-size}`` vs ``{, ignore-src}`` would be indistinguishable -- the
    same shape of defect as two entries with identical discriminators.

    Being a plain Python object rather than a PrimExpr, it cannot be a branch
    of a TVMScript conditional -- ``a if c else b`` lowers to
    ``tirx.if_then_else``, which takes ``ir.Expr`` operands. Wrap the whole
    expression instead::

        T.ptx.pred(accumulate if k == 0 else 1)   # not T.ptx.pred(x) if ... else ...

    A bool-typed expression needs no tag at all: its dtype already names the
    class, so ``tx == 0`` and ``True`` are accepted as they are.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"T.ptx.pred({self.value!r})"


def _coerce_operand(entry, slot, values, mod_map):
    """Coerce one whole operand: ``slot.lanes`` arguments in, that many out.

    Per operand rather than per argument, because a dtype belongs to the
    operand: a register group is ONE PTX operand occupying N registers, and ISA
    6.4.3 calls the brace list "similarly typed scalars". Choosing the dtype
    once for the group *is* the lane-agreement rule, so there is nothing left to
    re-check afterwards -- and a bare literal lane can take its dtype from the
    group instead of being typed on its own.
    """
    # T.local_scalar / `x: T.float32` hand back a wrapper around the BufferLoad;
    # unwrap once here so every kind sees the node itself. A PredArg is NOT
    # unwrapped here: whether the tag is present is the discriminator, so each
    # branch below has to be able to see it.
    values = [v if isinstance(v, PredArg) else getattr(v, "scalar", v) for v in values]
    addr_args = [v for v in values if isinstance(v, AddrArg)]
    if addr_args:
        if slot.kind != "addr" or not slot.allow_imm_offset:
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' does not support T.ptx.addr(...)"
            )
        if len(addr_args) != len(values):
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' cannot mix offset and plain addresses"
            )
        return [_coerce_addr_arg(entry, slot, value, mod_map) for value in values]
    is_pred = slot.kind == "reg" and operand_type(slot, mod_map) == "pred"
    tagged = [v for v in values if isinstance(v, PredArg)]
    if tagged and not is_pred:
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' is not a .pred operand, so it cannot "
            f"take a T.ptx.pred(...) value"
        )
    if is_pred:
        return _coerce_pred_operand(entry, slot, values)
    if slot.kind == "imm":
        return [_coerce_imm(entry, slot, v) for v in values]
    if slot.kind in ("addr", "ptr"):
        return [_coerce_address(entry, slot, v, mod_map) for v in values]
    return _coerce_typed(entry, slot, values, mod_map)


def _coerce_addr_arg(entry, slot, value, mod_map):
    base = getattr(value.base, "scalar", value.base)
    base = _coerce_address(entry, slot, base, mod_map)
    return call_intrin(base.ty, _ADDR_OP_NAME, base, value.byte_offset)


def _coerce_pred_operand(entry, slot, values):
    """Coerce a ``.pred`` operand -- the one register class the C boundary cannot bind.

    PTX tells a ``.pred`` operand from an integer one by the declared register
    class (``%p`` vs ``%r``); the carrier that gets it through inline asm is a
    uint32, which is also every integer operand's carrier, so the class has to
    be *evidenced* at the call instead of inferred from the value. That is what
    ``T.ptx.pred(x)`` is: the declaration, lifted to the call site. Without it
    the two `cp.async` syntax lines that differ only in whether their fourth
    operand is a predicate or a byte count would be indistinguishable.

    A tag is not needed on the way out: a ``.pred`` result is a destination
    like any other, and no syntax line offers a non-predicate alternative at
    the same position.
    """
    (value,) = values
    if slot.rw != "r":
        # The 0/1 materialization of a .pred result: a "=r" uint32 the caller
        # receives through a reference parameter, so it needs a writable
        # uint32 lvalue exactly like any other destination.
        if not isinstance(value, BufferLoad) or arg_dtype(value) != "uint32":
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' is a .pred result and must be "
                f"a writable uint32 scalar or buffer element (declare it first, "
                f'e.g. `ok = T.local_scalar("uint32")`)'
            )
        return values
    if isinstance(value, PredArg):
        value = getattr(value.value, "scalar", value.value)
        if isinstance(value, bool | int):
            return [const(int(value), "uint32")]
        ty = getattr(value, "ty", None)
        if isinstance(ty, PrimType) and ty.dtype in ("bool", "uint32", "int32"):
            return [value]
        raise ValueError(
            f"{entry.name}: T.ptx.pred(...) takes a bool/uint32/int32 expression, "
            f"got dtype {getattr(ty, 'dtype', type(value).__name__)}"
        )
    # A truth value needs no tag: it already says what it is.
    if isinstance(value, bool):
        return [const(int(value), "uint32")]
    ty = getattr(value, "ty", None)
    if isinstance(ty, PrimType) and ty.dtype == "bool":
        return [value]
    raise ValueError(
        f"{entry.name}: operand '{slot.name}' is a .pred argument. Pass a bool "
        f"expression, or wrap an integer one so the register class is explicit: "
        f"T.ptx.pred(x)"
    )


def _coerce_typed(entry, slot, values, mod_map):
    """Coerce a dtype-carrying register operand (``rw`` any)."""
    allowed = operand_dtypes(slot, mod_map)
    token = operand_type(slot, mod_map)
    if slot.rw in ("w", "rw"):
        # A PTX destination is a register the caller declared, so every lane has
        # to be a writable lvalue: a scalar (`x: T.float32`) or a buffer element.
        # Both are BufferLoad nodes, which the C codegen prints as the lvalue
        # bound to the helper's reference parameter.
        for value in values:
            if not isinstance(value, BufferLoad):
                raise ValueError(
                    f"{entry.name}: destination '{slot.name}' must be a writable scalar or "
                    f"buffer element (declare it first, e.g. `d: T.{allowed[0]}`), got "
                    f"{type(value).__name__} — a T.let binding is immutable and cannot be one"
                )
    role = "destination" if slot.rw in ("w", "rw") else "operand"
    # A bare Python literal names no dtype, so it does not get a vote.
    carried = [None if isinstance(v, int | float) else arg_dtype(v) for v in values]
    named = sorted({d for d in carried if d is not None})
    if len(named) > 1:
        raise ValueError(
            f"{entry.name}: {role} '{slot.name}' is one {len(values)}-register group, so all "
            f"its lanes must have one dtype, got {', '.join(d or 'literal' for d in carried)}"
        )
    dtype = named[0] if named else allowed[0]
    if dtype not in allowed:
        raise ValueError(
            f"{entry.name}: {role} '{slot.name}' must have dtype "
            f"{' / '.join(allowed)} (from .{token}), got {dtype}"
        )
    for value in values:
        # An integer literal is unambiguous -- its bits are its bits. A float one
        # is not, and which complaint it earns depends on why: an operand with a
        # dtype axis cannot tell which same-width type was meant, while a
        # concretely typed one would simply truncate.
        if isinstance(value, float) and not dtype.startswith("float"):
            if len(allowed) > 1:
                raise ValueError(
                    f"{entry.name}: {role} '{slot.name}' is .{token}, which accepts "
                    f"{' / '.join(allowed)}, so the float literal {value!r} is ambiguous — "
                    f"write the constant you mean, e.g. T.float32({value!r}) for its bits "
                    f"or T.{dtype}(...) for a number"
                )
            raise ValueError(
                f"{entry.name}: {role} '{slot.name}' is .{token}, so the float literal "
                f"{value!r} would be truncated — write T.{dtype}(...) if that is what you mean"
            )
    return [const(v, dtype) if isinstance(v, int | float) else v for v in values]


def _coerce_address(entry, slot, value, mod_map):
    """Coerce an address-like operand (``role`` addr or ptr)."""
    ty = getattr(value, "ty", None)
    if slot.kind == "ptr":
        if isinstance(ty, PointerType):
            return value
        raise ValueError(f"{entry.name}: operand '{slot.name}' must be a pointer")
    space = operand_space(slot, mod_map)
    if space == "tmem":
        # A tmem address is a packed (row << 16 | col) 32-bit value, not a
        # pointer into any address space the host language can name -- there is
        # nothing to convert, and a pointer here would be a category error.
        if isinstance(ty, PrimType) and ty.dtype == "uint32":
            return value
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' is a tmem address and must be a uint32 "
            f"(compose it with T.cuda.get_tmem_addr)"
        )
    if space.startswith("shared"):
        if isinstance(ty, PointerType):
            # Any pointer is accepted and converted, which is what the legacy
            # helpers did. The pointer's storage_scope is not a reliable
            # discriminator here: a shared buffer's ptr_to() reports 'global',
            # so gating on it rejects correct code.
            return cuda_cvta_generic_to_shared(value)
        if isinstance(ty, PrimType) and ty.dtype == "uint32":
            return value  # trusted raw shared-window address
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' must be a shared-scope pointer "
            f"or a uint32 shared address"
        )
    if isinstance(ty, PointerType):
        if ty.storage_scope.startswith("shared"):
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' is a {space or 'generic'} address "
                f"but got a shared-scope pointer"
            )
        return value
    if isinstance(ty, PrimType) and ty.dtype == "uint64":
        # A 64-bit address handle, e.g. T.address_of(tensormap). The helper
        # parameter is `const void*` and PTX binds it to the same "l" register
        # either way, so make the conversion an explicit, visible IR node
        # rather than a silent type pun.
        return reinterpret("handle", value)
    if isinstance(ty, PrimType) and ty.dtype == "uint32":
        raise ValueError(f"{entry.name}: uint32 address requires shared state space")
    raise ValueError(f"{entry.name}: operand '{slot.name}' must be a pointer or uint64 handle")


def _coerce_imm(entry, slot, value):
    """A caller-passed immediate: a compile-time constant.

    The value lands in the instruction *text* -- the ISA gives these operands
    no register form (verified for immHalfSplitoff: ptxas answers "Arguments
    mismatch" to a register there) -- so a runtime expression has nothing to
    lower to and is rejected outright rather than silently materialized.
    With `choices` the value is additionally checked against the declared set;
    an open slot declares no domain because the ISA declares none, so any
    constant passes.
    """
    if isinstance(value, IntImm):
        value = value.value
    if not isinstance(value, int) or isinstance(value, bool):
        # Open immediates may be produced by an explicitly-unrolled TIR loop.
        # Keep the integer expression in the Call; the unroll/simplify pipeline
        # must turn it into IntImm before the codegen hook bakes it into text.
        if slot.choices is None and arg_dtype(value) in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            return value
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' is an immediate in the instruction "
            f"text; it needs a compile-time integer constant, got {type(value).__name__}"
        )
    if slot.choices is not None and str(value) not in slot.choices:
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' must be one of "
            f"{', '.join(slot.choices)}, got {value}"
        )
    return const(value, "int32")


def _coerce_pred(entry, pred):
    # A Python bool names no dtype but is unambiguous, so type it here rather
    # than making every call site spell `T.bool(True)`.
    if isinstance(pred, bool):
        return const(pred, "bool")
    ty = getattr(pred, "ty", None)
    if isinstance(ty, PrimType) and ty.dtype in ("bool", "uint32", "int32"):
        return pred
    raise ValueError(f"{entry.name}: pred must be a bool/uint32/int32 expression")


def _emit(entry, filled, operands, pred=None, preserve_dst=False):
    # Modifiers resolve before the operands are looked at: the attribute chain
    # is parsed before the call happens, and a register group's length may be a
    # function of the modifiers, so the expected arity needs the modifier map.
    missing = [
        slot.name for slot, tok in zip(entry.slots, filled) if tok is None and not slot.optional
    ]
    if missing:
        raise ValueError(f"{entry.name}: missing required modifier(s): {', '.join(missing)}")
    mod_map = mods(entry, filled)
    if entry.check is not None:
        error = entry.check(mod_map)
        if error:
            raise ValueError(f"{entry.name}: {error}")
    layout = operand_layout(entry, mod_map)
    n_args = sum(n for _, _, n in layout)
    if len(operands) != n_args:
        names = ", ".join(f"{slot.name}[{n}]" if n > 1 else slot.name for slot, _, n in layout)
        raise ValueError(f"{entry.name} expects {n_args} operand(s) ({names}), got {len(operands)}")
    if preserve_dst and not entry.has_dst:
        raise ValueError(f"{entry.name}: preserve_dst=True requires a written destination")
    if pred is not None:
        pred = _coerce_pred(entry, pred)
    elif preserve_dst:
        raise ValueError(f"{entry.name}: preserve_dst=True requires pred=...")
    # The sink symbol `_`: a lane the caller discards. It is checked here
    # rather than in `_coerce_operand` because it is not a value at all -- the
    # instruction names the symbol, so the lane leaves no Call argument, and
    # its index has to be recorded for the codegen and the round trip.
    sunk = set()
    for slot, i, lanes in layout:
        for lane in range(lanes):
            if operands[i + lane] is not SINK:
                continue
            sinkable = slot.sinkable(mod_map) if callable(slot.sinkable) else slot.sinkable
            if not (sinkable and slot.kind == "reg"):
                raise ValueError(
                    f"{entry.name}: operand '{slot.name}' is not sinkable here, "
                    f"so it cannot take T.ptx.SINK"
                )
            sunk.add(i + lane)
        if lanes and all(i + lane in sunk for lane in range(lanes)):
            # ISA 9.7.9.4 states it for mov ("provided that at least one
            # element is a scalar register"); an instruction whose every
            # destination is discarded has nothing left to do anyway.
            raise ValueError(
                f"{entry.name}: at least one lane of '{slot.name}' must be a real register"
            )
    coerced = [
        value
        for slot, i, lanes in layout
        for value in _coerce_operand(
            entry,
            slot,
            [operands[i + lane] for lane in range(lanes) if i + lane not in sunk],
            mod_map,
        )
    ]
    # Call arg layout: [operands..., pred?] [slot tokens ("" = omitted)] [marker].
    # The trailing marker states what the argument list alone cannot: whether a
    # predicate operand is present ("pred"), and which operand positions carry
    # a `.pred` register class ("p<i>"), comma-joined. Without it a printed call
    # would have to be re-parsed by guessing -- in a family whose syntax lines
    # differ by one optional operand a predicated short form is
    # indistinguishable from an unpredicated long one, and a `.pred` operand is
    # indistinguishable from the integer that shares its carrier. The marker
    # makes the round trip exact; a call with neither prints "" as before.
    flags = ["pred"] if pred is not None else []
    if preserve_dst:
        flags.append("keep")
    flags += [
        f"p{i}"
        for slot, i, lanes in layout
        if lanes and slot.kind == "reg" and slot.rw == "r" and operand_type(slot, mod_map) == "pred"
    ]
    # A sunk lane leaves no argument behind, so only the marker can say it was
    # there -- without it a printed call would re-parse at the wrong arity.
    flags += [f"s{i}" for i in sorted(sunk)]
    return call_intrin(
        "",  # every ptx call is a void statement; destinations are operands
        entry.op_name,
        *coerced,
        *((pred,) if pred is not None else ()),
        *mod_map.values(),
        ",".join(flags),
    )


# ---------------------------------------------------------------------------
# Namespace surface: T.ptx attribute chains + string form
# ---------------------------------------------------------------------------


def _fill(entry, filled, token):
    """Assign ``token`` to the first open modifier slot listing it, or None.

    Slot membership only; whether the final combination is legal is decided
    once at call time by the entry's ``check`` function. Returning None rather
    than raising keeps the one error message in :func:`_narrow`, which is the
    only caller that knows the full candidate set.
    """
    for i, slot in enumerate(entry.slots):
        if filled[i] is None and token in slot.choices:
            return (*filled[:i], token, *filled[i + 1 :])
    return None


class _InstrChain:
    """A PTX instruction family with a partially-filled modifier tuple.

    Holds *candidate* ``(entry, filled)`` pairs rather than a single entry,
    because PTX puts some of an instruction's shape in the operand list rather
    than in the dotted modifier text: ``mov.b64 d, {lo, hi}`` and
    ``mov.b64 {lo, hi}, a`` are the same opcode with different operand shapes.
    Each modifier token narrows the candidates; the call's argument count and
    operand dtypes -- the same information the assembler resolves them by --
    select the final one.
    """

    __slots__ = ("_cands",)

    def __init__(self, cands):
        self._cands = tuple(cands)

    def __getattr__(self, name):
        if name.startswith("_"):  # keep copy/pickle/IPython dunder probes out
            raise AttributeError(name)
        return _InstrChain(_narrow(self._cands, unescape_token(name)))

    def __call__(self, *args, pred=None, preserve_dst=False):
        # Also accepts the printed round-trip form: trailing modifier-token
        # strings in slot order ("" = omitted slot) followed by the pred
        # marker, and, when the marker says "pred", the predicate as the last
        # expression operand. The marker is what makes this exact -- see the
        # arg-layout note in `_emit`.
        split = len(args)
        while split > 0 and isinstance(args[split - 1], str):
            split -= 1
        trailing = args[split:]
        args = args[:split]
        if trailing:  # printed form: last trailing string is the marker
            *tokens, marker = trailing
            flags = marker.split(",") if marker else []
            if "pred" in flags and pred is None and args:
                args, pred = args[:-1], args[-1]
            if "keep" in flags:
                preserve_dst = True
            # Put back what the printed text cannot carry: the sunk lanes
            # (which left no argument at all, so they are re-inserted first, in
            # ascending order, to restore the operand positions) and then the
            # register-class tags, whose indices are those same positions.
            args = list(args)
            for flag in sorted(f for f in flags if f.startswith("s")):
                args.insert(int(flag[1:]), SINK)
            for flag in flags:
                if flag.startswith("p") and flag != "pred":
                    args[int(flag[1:])] = PredArg(args[int(flag[1:])])
            args = tuple(args)
        else:
            tokens = ()
        cands = self._cands
        for token in tokens:
            if token:
                cands = _narrow(cands, token)
        operands = args

        hits, errors = [], []
        # `pred` is keyword-only. There used to be a fallback that read one
        # extra positional argument as the predicate; it made every entry also
        # accept arity+1 calls, so a no-count and a counted syntax line could
        # never share a mnemonic (both matched the two-operand call). Nothing
        # ever passed the predicate positionally, and removing the guess is
        # what lets optional trailing operands dispatch by arity.
        for entry, filled in cands:
            try:
                hits.append(
                    (
                        entry,
                        _emit(
                            entry,
                            filled,
                            operands,
                            pred=pred,
                            preserve_dst=preserve_dst,
                        ),
                    )
                )
            except ValueError as err:
                # Keep the exception; a lone candidate re-raises it untouched,
                # and only the aggregate view needs entry names in front.
                errors.append((entry, err))
        if len(hits) == 1:
            return hits[0][1]
        if not hits:
            if len(errors) == 1:
                raise errors[0][1]
            raise ValueError(
                "no ptx instruction matches these operands; candidates rejected it as:\n  "
                + "\n  ".join(f"{e.name}: {err}" for e, err in errors)
            )
        raise AssertionError(  # a table bug, not a user error
            # The entries that ACCEPTED, not the ones that merely survived
            # narrowing: naming a candidate that `_emit` rejected sends whoever
            # reads this after the wrong row.
            f"ambiguous ptx table: {len(hits)} entries accept the same call "
            f"({', '.join(e.name for e, _ in hits)})"
        )

    def __dir__(self):
        """Valid next tokens — drives tab completion in IPython/Jupyter."""
        return sorted(
            {
                escape_token(tok)
                for entry, filled in self._cands
                for i, slot in enumerate(entry.slots)
                if filled[i] is None
                for tok in slot.choices
            }
        )

    def __repr__(self):
        entry, filled = self._cands[0]
        mods_str = [tok for tok in filled if tok]
        return f"<T.ptx.{'.'.join([entry.ptx_name, *mods_str])}>"


def _narrow(cands, token):
    """Keep the candidates that can still take ``token`` in an open slot."""
    out = [(e, filled) for e, f in cands if (filled := _fill(e, f, token)) is not None]
    if not out:
        entry, filled = cands[0]
        open_choices = [
            f"{slot.name}∈{{{','.join(slot.choices)}}}"
            for i, slot in enumerate(entry.slots)
            if filled[i] is None
        ]
        raise AttributeError(
            f"'{token}' is not a valid modifier for '{entry.ptx_name}'; "
            f"open slots: {'; '.join(open_choices) or '(none)'}"
        )
    return out


class PTXNamespace:
    """``T.ptx`` — table-driven PTX instruction namespace."""

    def __init__(self, table=None):
        if table is None:
            from .table import TABLE as table  # pylint: disable=import-outside-toplevel
        self._table = table
        # Grouped once: the table is fixed here, and this lookup is the first
        # step of every traced call. The key is the mnemonic with dots folded to
        # underscores -- identical to the table key for every single-shape
        # family (st.bulk -> st_bulk), and the shared surface name for families
        # whose shapes are separate entries (all `mov_*` answer to `mov`).
        self._by_family = {}
        for entry in table.values():
            self._by_family.setdefault(entry.family, []).append((entry, (None,) * len(entry.slots)))

    #: The sink symbol -- see :class:`_Sink`.
    SINK = SINK

    @staticmethod
    def pred(value):
        """Tag an operand as a ``.pred`` register -- see :class:`PredArg`."""
        return PredArg(value)

    @staticmethod
    def addr(base, byte_offset):
        """Form ``[base+byte_offset]`` for an eligible PTX address operand."""
        return AddrArg(base, byte_offset)

    def _family(self, token):
        cands = self._by_family.get(token)
        return _InstrChain(list(cands)) if cands else None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        chain = self._family(unescape_token(name))
        if chain is None:
            raise AttributeError(
                f"'{name}' is not a ptx instruction; known families: "
                f"{', '.join(sorted(self._family_names()))}"
            )
        return chain

    def __getitem__(self, text):
        """Exact-PTX-text form, e.g. ``T.ptx["st.weak.shared::cta.b32"]``."""
        first, _, rest = text.partition(".")
        chain = self._family(first)
        if chain is None:
            raise KeyError(
                f"'{text}' does not start with a ptx instruction family; "
                f"known: {', '.join(sorted(self._family_names()))}"
            )
        for token in rest.split(".") if rest else []:
            try:
                chain = getattr(chain, escape_token(token))
            except AttributeError as err:
                raise KeyError(str(err)) from None
        return chain

    def _family_names(self):
        return set(self._by_family)

    def __dir__(self):
        """Family names — drives tab completion."""
        return sorted(self._family_names() | {"addr"} | set(super().__dir__()))

    def __repr__(self):
        return f"<T.ptx: {len(self._family_names())} instruction families>"
