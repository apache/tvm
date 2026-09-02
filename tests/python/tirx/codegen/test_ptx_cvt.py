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
"""The ptx cvt entries: one case per registered syntax line of ISA 9.7.10.24."""

import pytest

from tvm.backend.cuda.ptx.render import render_variant
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_dtypes, renderings, tokens_for

# (entry name, the modifier slots to write, the instruction that combination emits).
# Slots are named, not positional: `tokens_for` shifts nothing when a slot is
# inserted, rejects an unknown slot name, and refuses a combination the entry's
# own check would reject.
_FORM_CASES = [
    # generic scalar line: one case per rule the check enforces
    ("cvt", dict(rnd="rzi", dtype="s32", atype="f32"), "cvt.rzi.s32.f32"),
    ("cvt", dict(rnd="rn", dtype="f32", atype="s32"), "cvt.rn.f32.s32"),
    ("cvt", dict(rnd="rn", dtype="f16", atype="f32"), "cvt.rn.f16.f32"),
    ("cvt", dict(dtype="f32", atype="f16"), "cvt.f32.f16"),
    ("cvt", dict(sat="sat", dtype="s8", atype="s32"), "cvt.sat.s8.s32"),
    ("cvt", dict(rnd="rzi", ftz="ftz", dtype="s32", atype="f32"), "cvt.rzi.ftz.s32.f32"),
    # the two frnd2 scalar lines, which share that entry's shape and types
    (
        "cvt",
        dict(rnd="rn", relu="relu", satfinite="satfinite", dtype="f16", atype="f32"),
        "cvt.rn.relu.satfinite.f16.f32",
    ),
    (
        "cvt",
        dict(rnd="rz", satfinite="satfinite", dtype="bf16", atype="f32"),
        "cvt.rz.satfinite.bf16.f32",
    ),
    # the frnd2 lines that pack two .f32 sources into one register
    ("cvt_f16x2_f32", dict(rnd="rn", dtype="f16x2", atype="f32"), "cvt.rn.f16x2.f32"),
    (
        "cvt_f16x2_f32",
        dict(rnd="rz", relu="relu", satfinite="satfinite", dtype="f16x2", atype="f32"),
        "cvt.rz.relu.satfinite.f16x2.f32",
    ),
    (
        "cvt_bf16x2_f32",
        dict(rnd="rn", relu="relu", dtype="bf16x2", atype="f32"),
        "cvt.rn.relu.bf16x2.f32",
    ),
    # both .tf32 lines
    (
        "cvt_tf32_f32",
        dict(rnd="rna", satfinite="satfinite", dtype="tf32", atype="f32"),
        "cvt.rna.satfinite.tf32.f32",
    ),
    (
        "cvt_tf32_f32",
        dict(rnd="rn", satfinite="satfinite", relu="relu", dtype="tf32", atype="f32"),
        "cvt.rn.satfinite.relu.tf32.f32",
    ),
    # the .rs lines: a trailing rbits operand, and {a, b, e, f} on the x4 forms
    ("cvt_rs_f16x2_f32", dict(rnd="rs", dtype="f16x2", atype="f32"), "cvt.rs.f16x2.f32"),
    (
        "cvt_rs_bf16x2_f32",
        dict(rnd="rs", relu="relu", satfinite="satfinite", dtype="bf16x2", atype="f32"),
        "cvt.rs.relu.satfinite.bf16x2.f32",
    ),
    (
        "cvt_rs_f8x4_f32",
        dict(rnd="rs", satfinite="satfinite", dtype="e4m3x4", atype="f32"),
        "cvt.rs.satfinite.e4m3x4.f32",
    ),
    (
        "cvt_rs_f4x4_f32",
        dict(rnd="rs", relu="relu", satfinite="satfinite", dtype="e2m1x4", atype="f32"),
        "cvt.rs.relu.satfinite.e2m1x4.f32",
    ),
    (
        "cvt_rs_f6x4_f32",
        dict(rnd="rs", satfinite="satfinite", dtype="e2m3x4", atype="f32"),
        "cvt.rs.satfinite.e2m3x4.f32",
    ),
    ("cvt_ue8m0x2_f32", dict(rnd="rz", dtype="ue8m0x2", atype="f32"), "cvt.rz.ue8m0x2.f32"),
    (
        "cvt_ue8m0x2_f32",
        dict(rnd="rp", satfinite="satfinite", dtype="ue8m0x2", atype="f32"),
        "cvt.rp.satfinite.ue8m0x2.f32",
    ),
    (
        "cvt_ue8m0x2_bf16x2",
        dict(rnd="rz", dtype="ue8m0x2", atype="bf16x2"),
        "cvt.rz.ue8m0x2.bf16x2",
    ),
    (
        "cvt_bf16x2_ue8m0x2",
        dict(rnd="rn", dtype="bf16x2", atype="ue8m0x2"),
        "cvt.rn.bf16x2.ue8m0x2",
    ),
    (
        "cvt_f8x2_f32",
        dict(rnd="rn", satfinite="satfinite", dtype="e4m3x2", atype="f32"),
        "cvt.rn.satfinite.e4m3x2.f32",
    ),
    (
        "cvt_f8x2_f32",
        dict(rnd="rn", satfinite="satfinite", relu="relu", dtype="e5m2x2", atype="f32"),
        "cvt.rn.satfinite.relu.e5m2x2.f32",
    ),
    (
        "cvt_f8x2_fp16x2",
        dict(rnd="rn", satfinite="satfinite", dtype="e4m3x2", atype="f16x2"),
        "cvt.rn.satfinite.e4m3x2.f16x2",
    ),
    ("cvt_f16x2_f8x2", dict(rnd="rn", dtype="f16x2", atype="e4m3x2"), "cvt.rn.f16x2.e4m3x2"),
    (
        "cvt_f16x2_f8x2",
        dict(rnd="rn", relu="relu", dtype="f16x2", atype="e5m2x2"),
        "cvt.rn.relu.f16x2.e5m2x2",
    ),
    ("cvt_bf16x2_f8x2", dict(rnd="rn", dtype="bf16x2", atype="e4m3x2"), "cvt.rn.bf16x2.e4m3x2"),
    (
        "cvt_bf16x2_f8x2",
        dict(rnd="rn", relu="relu", satfinite="satfinite", dtype="bf16x2", atype="e5m2x2"),
        "cvt.rn.relu.satfinite.bf16x2.e5m2x2",
    ),
    (
        "cvt_bf16x2_f8x2",
        dict(
            rnd="rn",
            satfinite="satfinite",
            scaled="scaled::n2::ue8m0",
            dtype="bf16x2",
            atype="e4m3x2",
        ),
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2",
    ),
    # the fp4 lines, whose `.b8` operand puts them on the raw_render path
    (
        "cvt_f4x2_f32",
        dict(rnd="rn", satfinite="satfinite", relu="relu", dtype="e2m1x2", atype="f32"),
        "cvt.rn.satfinite.relu.e2m1x2.f32",
    ),
    (
        "cvt_f4x2_fp16x2",
        dict(rnd="rn", satfinite="satfinite", dtype="e2m1x2", atype="bf16x2"),
        "cvt.rn.satfinite.e2m1x2.bf16x2",
    ),
    (
        "cvt_f16x2_f4x2",
        dict(rnd="rn", relu="relu", dtype="f16x2", atype="e2m1x2"),
        "cvt.rn.relu.f16x2.e2m1x2",
    ),
    (
        "cvt_bf16x2_f4x2",
        dict(
            rnd="rn",
            satfinite="satfinite",
            scaled="scaled::n2::ue8m0",
            dtype="bf16x2",
            atype="e2m1x2",
        ),
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2",
    ),
    # the fp6 lines
    (
        "cvt_f6x2_f32",
        dict(rnd="rn", satfinite="satfinite", dtype="e2m3x2", atype="f32"),
        "cvt.rn.satfinite.e2m3x2.f32",
    ),
    (
        "cvt_f6x2_fp16x2",
        dict(rnd="rn", satfinite="satfinite", relu="relu", dtype="e3m2x2", atype="bf16x2"),
        "cvt.rn.satfinite.relu.e3m2x2.bf16x2",
    ),
    (
        "cvt_f16x2_f6x2",
        dict(rnd="rn", relu="relu", dtype="f16x2", atype="e2m3x2"),
        "cvt.rn.relu.f16x2.e2m3x2",
    ),
    (
        "cvt_bf16x2_f6x2",
        dict(
            rnd="rn",
            satfinite="satfinite",
            scaled="scaled::n2::ue8m0",
            dtype="bf16x2",
            atype="e3m2x2",
        ),
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e3m2x2",
    ),
    # the .s2f6x2 lines, with and without the scale-factor operand
    (
        "cvt_s2f6x2_f32",
        dict(rnd="rn", satfinite="satfinite", dtype="s2f6x2", atype="f32"),
        "cvt.rn.satfinite.s2f6x2.f32",
    ),
    (
        "cvt_s2f6x2_f32",
        dict(
            rnd="rn",
            satfinite="satfinite",
            relu="relu",
            scaled="scaled::n2::ue8m0",
            dtype="s2f6x2",
            atype="f32",
        ),
        "cvt.rn.satfinite.relu.scaled::n2::ue8m0.s2f6x2.f32",
    ),
    (
        "cvt_s2f6x2_bf16x2",
        dict(
            rnd="rn",
            satfinite="satfinite",
            scaled="scaled::n2::ue8m0",
            dtype="s2f6x2",
            atype="bf16x2",
        ),
        "cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2",
    ),
    (
        "cvt_bf16x2_s2f6x2",
        dict(rnd="rn", relu="relu", dtype="bf16x2", atype="s2f6x2"),
        "cvt.rn.relu.bf16x2.s2f6x2",
    ),
    # PTX ISA 9.4: pzo, narrow rz/n1 scaling, and UE5M3.
    (
        "cvt_pzo_scalar_f32",
        dict(rnd="rn", pzo="pzo", dtype="f16", atype="f32"),
        "cvt.rn.pzo.f16.f32",
    ),
    (
        "cvt_pzo_fp16x2_f32",
        dict(rnd="rz", satfinite="satfinite", pzo="pzo", dtype="bf16x2", atype="f32"),
        "cvt.rz.satfinite.pzo.bf16x2.f32",
    ),
    (
        "cvt_pzo_tf32_f32",
        dict(rnd="rn", relu="relu", pzo="pzo", dtype="tf32", atype="f32"),
        "cvt.rn.relu.pzo.tf32.f32",
    ),
    (
        "cvt_94_narrow_f32",
        dict(
            rnd="rz",
            satfinite="satfinite",
            scaled="scaled::n1::ue8m0",
            dtype="e4m3x2",
            atype="f32",
        ),
        "cvt.rz.satfinite.scaled::n1::ue8m0.e4m3x2.f32",
    ),
    (
        "cvt_94_narrow_fp16x2",
        dict(
            rnd="rn",
            satfinite="satfinite",
            pzo="pzo",
            dtype="e2m1x2",
            atype="bf16x2",
        ),
        "cvt.rn.satfinite.pzo.e2m1x2.bf16x2",
    ),
    (
        "cvt_ue5m3x2_f32",
        dict(rnd="rp", satfinite="satfinite", dtype="ue5m3x2", atype="f32"),
        "cvt.rp.satfinite.ue5m3x2.f32",
    ),
    (
        "cvt_ue5m3x2_f32_scaled",
        dict(
            rnd="rz",
            scaled="scaled::n1::ue8m0",
            dtype="ue5m3x2",
            atype="f32",
        ),
        "cvt.rz.scaled::n1::ue8m0.ue5m3x2.f32",
    ),
    (
        "cvt_ue5m3x2_fp16x2",
        dict(rnd="rn", dtype="ue5m3x2", atype="f16x2"),
        "cvt.rn.ue5m3x2.f16x2",
    ),
    (
        "cvt_ue5m3x2_fp16x2_scaled",
        dict(
            rnd="rn",
            satfinite="satfinite",
            scaled="scaled::n1::ue8m0",
            dtype="ue5m3x2",
            atype="bf16x2",
        ),
        "cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.bf16x2",
    ),
    (
        "cvt_f16x2_ue5m3x2",
        dict(rnd="rn", dtype="f16x2", atype="ue5m3x2"),
        "cvt.rn.f16x2.ue5m3x2",
    ),
    (
        "cvt_bf16x2_ue5m3x2",
        dict(
            rnd="rn",
            scaled="scaled::n2::ue8m0",
            dtype="bf16x2",
            atype="ue5m3x2",
        ),
        "cvt.rn.scaled::n2::ue8m0.bf16x2.ue5m3x2",
    ),
]

# The generic scalar line is one entry named plain "cvt"; the packed lines are
# the "cvt_*" family.
# Every entry of the `cvt` instruction (ISA 9.7.10.24). Keyed off the mnemonic
# rather than the table name: `cvt.pack` (9.7.10.25) is a different instruction
# that happens to sort under the same prefix, and its forms are not points on
# this conversion grid.
_CVT_ENTRIES = {name for name, entry in TABLE.items() if entry.ptx_name == "cvt"}


@pytest.mark.parametrize("entry_name,slots,instruction", _FORM_CASES)
def test_cvt_form_renders_its_instruction(entry_name, slots, instruction):
    entry = TABLE[entry_name]
    opcode, helper, source = render_variant(entry, tokens_for(entry, **slots))
    assert opcode == instruction
    # A trailing space, so a shorter opcode cannot match as a prefix. Not
    # anchored to the opening quote: the `.e2m1x2` lines stage a `.b8` operand
    # in a block-local `.reg .b8` (see `_cvt_f4x2_raw`), so there the
    # instruction is a statement inside the block rather than the whole asm
    # text. That those bodies emit it as their own statement is the
    # single-instruction invariant's job, not this test's.
    assert f"{instruction} " in source
    assert helper.startswith("tvm_builtin_ptx_cvt_")
    # Every ptx helper is void: the destination is an operand, not a return.
    assert source.startswith("__forceinline__ __device__ void ")


def test_cvt_cases_cover_every_registered_entry():
    """Every cvt entry has a case here, so a newly transcribed syntax line has
    to be given one rather than sliding in untested."""
    assert {case[0] for case in _FORM_CASES} == _CVT_ENTRIES


def test_cvt_packed_operands_bind_their_carrier():
    """A packed format names a lane layout; the register it binds is the
    carrier, not one register per lane."""
    _, _, source = render_variant(TABLE["cvt_ue8m0x2_f32"], ("rz", "", "ue8m0x2", "f32"))
    # .ue8m0x2 is two 8-bit exponents in one 16-bit register, from two floats.
    assert "uint16_t& __d" in source
    assert source.count("float __") == 2

    _, _, source = render_variant(TABLE["cvt_bf16x2_ue8m0x2"], ("rn", "bf16x2", "ue8m0x2"))
    assert "uint32_t& __d" in source
    assert "uint16_t __a" in source


_CVT_RELAXED_DTYPE_CASES = [
    (
        "u8",
        (
            "uint8",
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "uint64",
            "int64",
            "uint128",
            "int128",
        ),
        None,
    ),
    (
        "s8",
        (
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "int128",
            "uint128",
        ),
        None,
    ),
    (
        "u16",
        ("uint16", "int16", "uint32", "int32", "uint64", "int64", "uint128", "int128"),
        None,
    ),
    (
        "s16",
        ("int16", "uint16", "int32", "uint32", "int64", "uint64", "int128", "uint128"),
        None,
    ),
    ("u32", ("uint32", "int32", "uint64", "int64", "uint128", "int128"), None),
    ("s32", ("int32", "uint32", "int64", "uint64", "int128", "uint128"), None),
    ("u64", ("uint64", "int64", "uint128", "int128"), None),
    ("s64", ("int64", "uint64", "int128", "uint128"), None),
    (
        "f16",
        (
            "uint16",
            "int16",
            "float16",
            "bfloat16",
            "uint32",
            "int32",
            "uint64",
            "int64",
            "uint128",
            "int128",
        ),
        ("uint16", "int16", "float16", "bfloat16"),
    ),
    ("bf16", ("uint16",), ("uint16",)),
    (
        "f32",
        ("float32", "uint32", "int32", "uint64", "int64", "uint128", "int128"),
        ("float32", "uint32", "int32"),
    ),
    (
        "f64",
        ("float64", "uint64", "int64", "uint128", "int128"),
        ("float64", "uint64", "int64"),
    ),
]


@pytest.mark.parametrize("ptx_type,dst_expected,src_expected", _CVT_RELAXED_DTYPE_CASES)
def test_cvt_generic_scalar_relaxed_carriers(ptx_type, dst_expected, src_expected):
    """Keep documented relaxation except where ptxas rejects a floating source.

    The native spelling remains first to preserve canonical helper names.
    Integer instruction types never admit a floating register, and the cvt
    section explicitly exempts `.bf16` from widening in both directions. PTX
    9.2 permits wider bit registers for the other floating sources, but ptxas
    rejects them, so they are not callable interfaces in this backend.
    """
    entry = TABLE["cvt"]
    mod_map = mods(entry, tokens_for(entry, dtype=ptx_type, atype=ptx_type))
    assert operand_dtypes(entry.operands[0], mod_map) == dst_expected
    assert operand_dtypes(entry.operands[1], mod_map) == (src_expected or dst_expected)


def test_cvt_generic_scalar_cuda_13_4_bf16_opposite_operand_gap():
    """CUDA 13.4 ptxas also rejects widening the operand opposite `.bf16`."""
    entry = TABLE["cvt"]
    to_bf16 = mods(entry, tokens_for(entry, rnd="rn", dtype="bf16", atype="u32"))
    assert operand_dtypes(entry.operands[0], to_bf16) == ("uint16",)
    assert operand_dtypes(entry.operands[1], to_bf16) == ("uint32", "int32")

    from_bf16 = mods(entry, tokens_for(entry, rnd="rni", dtype="u32", atype="bf16"))
    assert operand_dtypes(entry.operands[0], from_bf16) == ("uint32", "int32")
    assert operand_dtypes(entry.operands[1], from_bf16) == ("uint16",)

    # The gap is about width, not signedness or register interpretation.
    from_bf16_f16 = mods(entry, tokens_for(entry, rnd="rn", dtype="f16", atype="bf16"))
    assert operand_dtypes(entry.operands[0], from_bf16_f16) == (
        "uint16",
        "int16",
        "float16",
        "bfloat16",
    )


def test_cvt_generic_scalar_cuda_13_4_ftz_destination_gaps():
    """Pin the two destination-only `.ftz` gaps measured on CUDA 13.4 ptxas."""
    entry = TABLE["cvt"]

    u64_from_f32 = mods(entry, tokens_for(entry, rnd="rni", ftz="ftz", dtype="u64", atype="f32"))
    assert operand_dtypes(entry.operands[0], u64_from_f32) == ("uint64", "int64")

    f64_from_f32 = mods(entry, tokens_for(entry, ftz="ftz", dtype="f64", atype="f32"))
    assert operand_dtypes(entry.operands[0], f64_from_f32) == (
        "float64",
        "uint64",
        "int64",
    )

    f32_from_f64_rz = mods(entry, tokens_for(entry, rnd="rz", ftz="ftz", dtype="f32", atype="f64"))
    assert operand_dtypes(entry.operands[0], f32_from_f64_rz) == (
        "float32",
        "uint32",
        "int32",
    )

    # ptxas accepts the 64-bit destination on `.rn`, but still rejects q.
    f32_from_f64_rn = mods(entry, tokens_for(entry, rnd="rn", ftz="ftz", dtype="f32", atype="f64"))
    assert operand_dtypes(entry.operands[0], f32_from_f64_rn) == (
        "float32",
        "uint32",
        "int32",
        "uint64",
        "int64",
    )


def test_cvt_generic_scalar_keeps_supported_wide_carriers():
    """Do not turn CUDA 13.4's narrow gaps into a blanket cvt restriction."""
    entry = TABLE["cvt"]

    no_ftz = mods(entry, tokens_for(entry, rnd="rn", dtype="f32", atype="f64"))
    assert operand_dtypes(entry.operands[0], no_ftz)[-2:] == ("uint128", "int128")

    narrow_ftz = mods(entry, tokens_for(entry, rnd="rn", ftz="ftz", dtype="f16", atype="f32"))
    assert operand_dtypes(entry.operands[0], narrow_ftz)[-2:] == ("uint128", "int128")

    wide_integer_source = mods(
        entry, tokens_for(entry, rnd="rn", ftz="ftz", dtype="f32", atype="u64")
    )
    assert operand_dtypes(entry.operands[1], wide_integer_source)[-2:] == (
        "uint128",
        "int128",
    )


def test_cvt_generic_scalar_relaxed_carriers_render_exact_instruction():
    entry = TABLE["cvt"]

    tokens = tokens_for(entry, dtype="s16", atype="u16")
    opcode, helper, source = render_variant(entry, tokens, dtypes=("int64", "uint64"))
    assert opcode == "cvt.s16.u16"
    assert helper == "tvm_builtin_ptx_cvt_s16_u16_s64_u64"
    assert "(int64_t& __d, uint64_t __a)" in source
    assert '"cvt.s16.u16 %0, %1;" : "=l"(__d) : "l"(__a)' in source

    # Destination widening remains supported, while the floating source is
    # exact-width because ptxas rejects the wider source form permitted by ISA
    # section 9.4.1, Table 27.
    tokens = tokens_for(entry, rnd="rn", dtype="f16", atype="f32")
    opcode, helper, source = render_variant(entry, tokens, dtypes=("uint128", "uint32"))
    assert opcode == "cvt.rn.f16.f32"
    assert helper == "tvm_builtin_ptx_cvt_rn_f16_f32_u128_u32"
    assert "(__uint128_t& __d, uint32_t __a)" in source
    assert '"cvt.rn.f16.f32 %0, %1;" : "=q"(__d) : "r"(__a)' in source


def test_cvt_packed_and_bf16_carriers_are_not_blanket_widened():
    generic_bf16 = TABLE["cvt"]
    bf16_map = mods(generic_bf16, tokens_for(generic_bf16, dtype="bf16", atype="bf16"))
    assert operand_dtypes(generic_bf16.operands[0], bf16_map) == ("uint16",)
    assert operand_dtypes(generic_bf16.operands[1], bf16_map) == ("uint16",)

    packed = TABLE["cvt_f16x2_f32"]
    packed_map = mods(packed, tokens_for(packed, rnd="rn", dtype="f16x2", atype="f32"))
    assert operand_dtypes(packed.operands[0], packed_map) == ("uint32",)
    assert operand_dtypes(packed.operands[1], packed_map) == ("float32",)

    tf32 = TABLE["cvt_tf32_f32"]
    tf32_map = mods(tf32, tokens_for(tf32, rnd="rna", dtype="tf32", atype="f32"))
    assert operand_dtypes(tf32.operands[0], tf32_map) == ("uint32",)
    assert operand_dtypes(tf32.operands[1], tf32_map) == ("float32",)


def test_cvt_bf16_sat_diagnostics_distinguish_isa_and_toolchain():
    entry = TABLE["cvt"]
    with pytest.raises(ValueError, match=r"ISA limits floating-point \.sat destinations"):
        tokens_for(entry, rnd="rn", sat="sat", dtype="bf16", atype="f32")
    with pytest.raises(ValueError, match=r"toolchain assembles no \.sat when \.bf16 is the source"):
        tokens_for(entry, sat="sat", dtype="f32", atype="bf16")


def test_cvt_e2m1x2_stages_its_b8_operand():
    """`.e2m1x2` is the one cvt format with no register of its own width.

    ISA 9.7.10.24:92 "When converting to .e2m1x2 data formats, the destination
    operand d has .b8 type." and :101 "When converting from .e2m1x2 to
    .f16x2/.bf16x2, source operand a has .b8 type." Inline asm has no 8-bit
    constraint letter, so both directions declare the register inside the block
    and bridge it to the 16-bit "h" carrier; uint8_t is what the caller sees.
    """
    _, _, source = render_variant(TABLE["cvt_f4x2_f32"], ("rn", "satfinite", "", "e2m1x2", "f32"))
    assert "void tvm_builtin_ptx_cvt_f4x2_f32_rn_satfinite_e2m1x2_f32(" in source
    assert "(uint8_t& __d, float __a, float __b)" in source
    assert (
        '"{ .reg .b8 raw_d; cvt.rn.satfinite.e2m1x2.f32 raw_d, %1, %2;'
        ' cvt.u16.u8 %0, raw_d; }" : "=h"(__d_reg)' in source
    )
    assert "__d = (uint8_t)__d_reg;" in source

    _, _, source = render_variant(TABLE["cvt_f16x2_f4x2"], ("rn", "", "f16x2", "e2m1x2"))
    assert "(uint32_t& __d, uint8_t __a)" in source
    assert (
        '"{ .reg .b8 raw_a; cvt.u8.u16 raw_a, %1; cvt.rn.f16x2.e2m1x2 %0, raw_a; }"'
        ' : "=r"(__d) : "h"((uint16_t)__a)' in source
    )

    # The scale-factor operand is .b16, so it binds a register directly and
    # rides alongside the staged source.
    scaled = ("rn", "", "", "scaled::n2::ue8m0", "bf16x2", "e2m1x2")
    _, _, source = render_variant(TABLE["cvt_bf16x2_f4x2"], scaled)
    assert "(uint32_t& __d, uint8_t __a, uint16_t __scale_factor)" in source
    assert "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, raw_a, %2;" in source


# The cvt type tokens ISA 9.7.10.24's Target ISA Notes list architecture by
# architecture (:527-639), plus the .rs rounding mode (":602 .rs rounding mode
# is supported on following architectures:", listing sm_100a and sm_103a).
_CVT_BLACKWELL_TOKENS = {
    "ue8m0x2",
    "s2f6x2",
    "e2m1x2",
    "e2m3x2",
    "e3m2x2",
    "e2m1x4",
    "e2m3x4",
    "e3m2x4",
    "e4m3x4",
    "e5m2x4",
    "rs",
}


def _needs_sm100a(written: set[str]) -> bool:
    """Whether one rendering's tokens put it on a Blackwell-floor syntax line.

    `.e4m3x2`/`.e5m2x2` alone are sm_89 lines. Pairing either with `.bf16x2`
    needs a floor whichever side it sits on: as the destination it is the PTX
    9.2 line at :634-639, and as the source it is :612-617
    ("cvt.rn.satfinite{.relu}{.e5m2x2/.e4m3x2}{.bf16x2} is supported on
    following family-specific architectures:"). The second clause covers both
    directions.
    """
    return bool(written & _CVT_BLACKWELL_TOKENS) or (
        "bf16x2" in written and bool(written & {"e4m3x2", "e5m2x2"})
    )


_PTX94_CVT_ENTRIES = {
    "cvt_pzo_scalar_f32",
    "cvt_pzo_fp16x2_f32",
    "cvt_pzo_tf32_f32",
    "cvt_94_narrow_f32",
    "cvt_94_narrow_fp16x2",
    "cvt_ue5m3x2_f32",
    "cvt_ue5m3x2_f32_scaled",
    "cvt_ue5m3x2_fp16x2",
    "cvt_ue5m3x2_fp16x2_scaled",
    "cvt_f16x2_ue5m3x2",
    "cvt_bf16x2_ue5m3x2",
}


def test_cvt_blackwell_lines_carry_their_arch_floor():
    """Blackwell-only cvt lines carry their maximum documented family floor.

    The baseline lines certify at sm_100a and PTX 9.4's SM107 lines at sm_107f;
    certifying either at the sm_90 default would report legal forms as illegal.
    The floor rides the narrow format, not `.bf16x2` on its own: ISA
    9.7.10.24:517-518 puts `.bf16x2` as a destination format at "sm_80 or
    higher", which is where cvt.frnd2{.relu}{.satfinite}.bf16x2.f32 lives,
    while :634-639 restrict `.bf16x2` *from* an fp8/fp6/fp4 format to
    family-specific architectures.
    """
    for name in _CVT_ENTRIES:
        entry = TABLE[name]
        if any(_needs_sm100a(set(tokens)) for tokens, *_ in renderings(entry)):
            expected = "sm_107f" if name in _PTX94_CVT_ENTRIES else "sm_100a"
            assert entry.cert_arch == expected, name


def test_cvt_tf32_satfinite_carries_its_sm_100_floor():
    """ISA 9.7.10.24:526 "cvt.{rn/rz}.satfinite.tf32.f32 requires sm_100 or
    higher." -- the maximum floor over the entry, whose other spellings sit at
    sm_80/sm_90."""
    entry = TABLE["cvt_tf32_f32"]
    assert entry.cert_arch == "sm_100"
    opcodes = {render_variant(entry, tokens)[0] for tokens, *_ in renderings(entry)}
    assert "cvt.rn.satfinite.tf32.f32" in opcodes


def test_cvt_rs_and_scale_factor_shapes():
    """The two operand shapes this family added: the .rs lines' trailing rbits
    (with a grouped ``{a, b, e, f}`` source on the x4 forms), and the
    scale-factor operand that exists exactly when .scaled::n2::ue8m0 is
    written (ISA 9.7.10.24:180-182 "Operand scale-factor and qualifier
    .scaled::n2::ue8m0 must be used together.")."""
    _, _, source = render_variant(TABLE["cvt_rs_f16x2_f32"], ("rs", "", "", "f16x2", "f32"))
    assert "uint32_t& __d, float __a, float __b, uint32_t __rbits" in source
    assert '"cvt.rs.f16x2.f32 %0, %1, %2, %3;"' in source

    _, _, source = render_variant(
        TABLE["cvt_rs_f8x4_f32"], ("rs", "", "satfinite", "e4m3x4", "f32")
    )
    # One operand, four registers: PTX writes the group in the operand list.
    assert "float __abef0, float __abef1, float __abef2, float __abef3" in source
    assert '"cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"' in source

    plain = ("rn", "satfinite", "", "", "s2f6x2", "f32")
    scaled = ("rn", "satfinite", "", "scaled::n2::ue8m0", "s2f6x2", "f32")
    _, _, source = render_variant(TABLE["cvt_s2f6x2_f32"], plain)
    assert '"cvt.rn.satfinite.s2f6x2.f32 %0, %1, %2;"' in source
    assert "__scale_factor" not in source
    _, _, source = render_variant(TABLE["cvt_s2f6x2_f32"], scaled)
    assert '"cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32 %0, %1, %2, %3;"' in source
    assert "uint16_t __scale_factor" in source


if __name__ == "__main__":
    pytest.main([__file__])
