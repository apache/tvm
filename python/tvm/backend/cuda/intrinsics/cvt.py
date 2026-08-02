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
# pylint: disable=invalid-name, too-many-arguments, too-many-branches, too-many-locals
"""PTX ``cvt`` instruction forms.

The PTX ISA specifies ``cvt`` as a family of grammar forms.  The public
``tirx.ptx.cvt`` op carries concrete PTX types and optional modifiers as attrs.
Its codegen classifies the invocation into one of the forms below.  Each form
then receives only its register operands followed by the attrs admitted by that
specific grammar line.

Alternate and packed floating-point formats use their PTX register carriers:
``uint8`` for ``.b8`` values, ``uint16`` for ``.b16`` values, and ``uint32``
for ``.b32`` values.
"""

from dataclasses import dataclass

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str

_BASIC_TYPES = {
    "u8",
    "u16",
    "u32",
    "u64",
    "s8",
    "s16",
    "s32",
    "s64",
    "bf16",
    "f16",
    "f32",
    "f64",
}
_INTEGER_ROUNDING = {"rni", "rzi", "rmi", "rpi"}
_FLOAT_ROUNDING = {"rn", "rz", "rm", "rp"}
_F8X2_TYPES = {"e4m3x2", "e5m2x2"}
_F4X2_TYPES = {"e2m1x2"}
_F6X2_TYPES = {"e2m3x2", "e3m2x2"}
_F8X4_TYPES = {"e4m3x4", "e5m2x4"}
_F4X4_TYPES = {"e2m1x4"}
_F6X4_TYPES = {"e2m3x4", "e3m2x4"}
_FP16X2_TYPES = {"f16x2", "bf16x2"}


def _as_bool(value) -> bool:
    return bool(int(value)) if hasattr(value, "value") else bool(value)


def _safe(value: str) -> str:
    return value.replace("::", "_").replace(".", "_")


@dataclass(frozen=True)
class _CvtForm:
    """One syntax line from the PTX ``cvt`` grammar."""

    attrs: tuple[str, ...]
    primary_count: int
    dtype: str
    atype: str
    modifiers: tuple[str, ...]
    rbits: bool = False
    grouped_primary: bool = False


# Modifier descriptors:
#   literal          -> mandatory modifier, e.g. ``rn``
#   ``$rounding``    -> string-valued modifier attr
#   ``?relu``        -> boolean modifier attr
#   ``~scaled``      -> ``.scaled::<attr>`` and an extra scale-factor operand
_CVT_FORMS = {
    # Generic scalar forms.
    "scalar": _CvtForm(("dtype", "atype", "ftz", "sat"), 1, "$dtype", "$atype", ("?ftz", "?sat")),
    "irnd": _CvtForm(
        ("dtype", "atype", "rounding", "ftz", "sat"),
        1,
        "$dtype",
        "$atype",
        ("$rounding", "?ftz", "?sat"),
    ),
    "frnd": _CvtForm(
        ("dtype", "atype", "rounding", "ftz", "sat"),
        1,
        "$dtype",
        "$atype",
        ("$rounding", "?ftz", "?sat"),
    ),
    # f16 / bf16 / tf32 destination forms.
    "frnd2_f16_f32": _CvtForm(
        ("rounding", "relu", "satfinite"),
        1,
        "f16",
        "f32",
        ("$rounding", "?relu", "?satfinite"),
    ),
    "frnd2_f16x2_f32": _CvtForm(
        ("rounding", "relu", "satfinite"),
        2,
        "f16x2",
        "f32",
        ("$rounding", "?relu", "?satfinite"),
    ),
    "rs_f16x2_f32": _CvtForm(
        ("relu", "satfinite"),
        2,
        "f16x2",
        "f32",
        ("rs", "?relu", "?satfinite"),
        rbits=True,
    ),
    "frnd2_bf16_f32": _CvtForm(
        ("rounding", "relu", "satfinite"),
        1,
        "bf16",
        "f32",
        ("$rounding", "?relu", "?satfinite"),
    ),
    "frnd2_bf16x2_f32": _CvtForm(
        ("rounding", "relu", "satfinite"),
        2,
        "bf16x2",
        "f32",
        ("$rounding", "?relu", "?satfinite"),
    ),
    "rs_bf16x2_f32": _CvtForm(
        ("relu", "satfinite"),
        2,
        "bf16x2",
        "f32",
        ("rs", "?relu", "?satfinite"),
        rbits=True,
    ),
    "rna_tf32_f32": _CvtForm(("satfinite",), 1, "tf32", "f32", ("rna", "?satfinite")),
    "frnd2_tf32_f32": _CvtForm(
        ("rounding", "satfinite", "relu"),
        1,
        "tf32",
        "f32",
        ("$rounding", "?satfinite", "?relu"),
    ),
    # f8 forms.
    "rn_satfinite_f8x2_f32": _CvtForm(
        ("f8x2type", "relu"),
        2,
        "$f8x2type",
        "f32",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_satfinite_f8x2_fp16x2": _CvtForm(
        ("f8x2type", "fp16x2type", "relu"),
        1,
        "$f8x2type",
        "$fp16x2type",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_f16x2_f8x2": _CvtForm(("f8x2type", "relu"), 1, "f16x2", "$f8x2type", ("rn", "?relu")),
    "rn_bf16x2_f8x2": _CvtForm(
        ("f8x2type", "relu", "satfinite", "scaled"),
        1,
        "bf16x2",
        "$f8x2type",
        ("rn", "?relu", "?satfinite", "~scaled"),
    ),
    "rs_satfinite_f8x4_f32": _CvtForm(
        ("f8x4type", "relu"),
        4,
        "$f8x4type",
        "f32",
        ("rs", "?relu", "satfinite"),
        rbits=True,
        grouped_primary=True,
    ),
    # f4 forms.
    "rn_satfinite_f4x2_f32": _CvtForm(
        ("f4x2type", "relu"),
        2,
        "$f4x2type",
        "f32",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_satfinite_f4x2_fp16x2": _CvtForm(
        ("f4x2type", "fp16x2type", "relu"),
        1,
        "$f4x2type",
        "$fp16x2type",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_f16x2_f4x2": _CvtForm(("f4x2type", "relu"), 1, "f16x2", "$f4x2type", ("rn", "?relu")),
    "rn_bf16x2_f4x2": _CvtForm(
        ("f4x2type", "relu", "satfinite", "scaled"),
        1,
        "bf16x2",
        "$f4x2type",
        ("rn", "?relu", "?satfinite", "~scaled"),
    ),
    "rs_satfinite_f4x4_f32": _CvtForm(
        ("f4x4type", "relu"),
        4,
        "$f4x4type",
        "f32",
        ("rs", "?relu", "satfinite"),
        rbits=True,
        grouped_primary=True,
    ),
    # f6 forms.
    "rn_satfinite_f6x2_f32": _CvtForm(
        ("f6x2type", "relu"),
        2,
        "$f6x2type",
        "f32",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_satfinite_f6x2_fp16x2": _CvtForm(
        ("f6x2type", "fp16x2type", "relu"),
        1,
        "$f6x2type",
        "$fp16x2type",
        ("rn", "satfinite", "?relu"),
    ),
    "rn_f16x2_f6x2": _CvtForm(("f6x2type", "relu"), 1, "f16x2", "$f6x2type", ("rn", "?relu")),
    "rn_bf16x2_f6x2": _CvtForm(
        ("f6x2type", "relu", "satfinite", "scaled"),
        1,
        "bf16x2",
        "$f6x2type",
        ("rn", "?relu", "?satfinite", "~scaled"),
    ),
    "rs_satfinite_f6x4_f32": _CvtForm(
        ("f6x4type", "relu"),
        4,
        "$f6x4type",
        "f32",
        ("rs", "?relu", "satfinite"),
        rbits=True,
        grouped_primary=True,
    ),
    # ue8m0 forms.
    "frnd3_ue8m0x2_f32": _CvtForm(
        ("rounding", "satfinite"),
        2,
        "ue8m0x2",
        "f32",
        ("$rounding", "?satfinite"),
    ),
    "frnd3_ue8m0x2_bf16x2": _CvtForm(
        ("rounding", "satfinite"),
        1,
        "ue8m0x2",
        "bf16x2",
        ("$rounding", "?satfinite"),
    ),
    "rn_bf16x2_ue8m0x2": _CvtForm((), 1, "bf16x2", "ue8m0x2", ("rn",)),
    # s2f6 forms.
    "rn_satfinite_s2f6x2_f32": _CvtForm(
        ("relu", "scaled"),
        2,
        "s2f6x2",
        "f32",
        ("rn", "satfinite", "?relu", "~scaled"),
    ),
    "rn_satfinite_s2f6x2_bf16x2": _CvtForm(
        ("relu", "scaled"),
        1,
        "s2f6x2",
        "bf16x2",
        ("rn", "satfinite", "?relu", "~scaled"),
    ),
    "rn_bf16x2_s2f6x2": _CvtForm(
        ("satfinite", "relu", "scaled"),
        1,
        "bf16x2",
        "s2f6x2",
        ("rn", "?satfinite", "?relu", "~scaled"),
    ),
}


_STRING_ATTRS = {
    "dtype",
    "atype",
    "rounding",
    "f8x2type",
    "f8x4type",
    "f4x2type",
    "f4x4type",
    "f6x2type",
    "f6x4type",
    "fp16x2type",
    "scaled",
}


def _parse_form_attrs(form: _CvtForm, args):
    raw_attrs = args[-len(form.attrs) :] if form.attrs else ()
    attrs = {}
    for name, value in zip(form.attrs, raw_attrs):
        attrs[name] = parse_str(value) if name in _STRING_ATTRS else _as_bool(value)
    operands = args[: -len(form.attrs)] if form.attrs else args
    return operands, attrs


def _resolve(value: str, attrs: dict) -> str:
    return attrs[value[1:]] if value.startswith("$") else value


def _modifier_string(form: _CvtForm, attrs: dict) -> str:
    out = ""
    for modifier in form.modifiers:
        if modifier.startswith("$"):
            value = attrs[modifier[1:]]
            if value:
                out += f".{value}"
        elif modifier.startswith("?"):
            name = modifier[1:]
            if attrs[name]:
                out += f".{name}"
        elif modifier.startswith("~"):
            value = attrs[modifier[1:]]
            if value:
                out += f".scaled::{value}"
        else:
            out += f".{modifier}"
    return out


def _result_info(dtype: str):
    """Return (C return type, result temporary type, constraint, TVM dtype)."""
    if dtype == "u8":
        return "uint8_t", "uint32_t", "r", "uint8"
    if dtype == "s8":
        return "int8_t", "int32_t", "r", "int8"
    if dtype == "u16":
        return "uint16_t", "uint16_t", "h", "uint16"
    if dtype == "s16":
        return "int16_t", "int16_t", "h", "int16"
    if dtype == "u32":
        return "uint32_t", "uint32_t", "r", "uint32"
    if dtype == "s32":
        return "int32_t", "int32_t", "r", "int32"
    if dtype == "u64":
        return "uint64_t", "uint64_t", "l", "uint64"
    if dtype == "s64":
        return "int64_t", "int64_t", "l", "int64"
    if dtype == "f32":
        return "float", "float", "f", "float32"
    if dtype == "f64":
        return "double", "double", "d", "float64"
    if dtype == "e2m1x2":
        # CUDA extended asm has no 8-bit register constraint.  The helper
        # stages the PTX .b8 result through a .b16 output register.
        return "uint8_t", "uint16_t", "h", "uint8"
    if dtype in {"f16", "bf16", "e4m3x2", "e5m2x2", "e2m1x4"}:
        return "uint16_t", "uint16_t", "h", "uint16"
    if dtype in {"e2m3x2", "e3m2x2", "ue8m0x2", "s2f6x2"}:
        return "uint16_t", "uint16_t", "h", "uint16"
    if dtype in {
        "tf32",
        "f16x2",
        "bf16x2",
        "e4m3x4",
        "e5m2x4",
        "e2m3x4",
        "e3m2x4",
    }:
        return "uint32_t", "uint32_t", "r", "uint32"
    raise ValueError(f"Unsupported PTX cvt destination type {dtype!r}")


def _input_info(atype: str):
    """Return (C parameter type, asm constraint) for one source operand."""
    if atype == "u8":
        return "uint32_t", "r"
    if atype == "s8":
        return "int32_t", "r"
    if atype == "u16":
        return "uint16_t", "h"
    if atype == "s16":
        return "int16_t", "h"
    if atype == "u32":
        return "uint32_t", "r"
    if atype == "s32":
        return "int32_t", "r"
    if atype == "u64":
        return "uint64_t", "l"
    if atype == "s64":
        return "int64_t", "l"
    if atype == "f32":
        return "float", "f"
    if atype == "f64":
        return "double", "d"
    if atype in {
        "f16",
        "bf16",
        "e4m3x2",
        "e5m2x2",
        "e2m3x2",
        "e3m2x2",
        "ue8m0x2",
        "s2f6x2",
    }:
        return "uint16_t", "h"
    if atype == "e2m1x2":
        # CUDA extended asm has no 8-bit register constraint.  PTX permits a
        # .b8 value to be staged through a wider register.
        return "uint16_t", "h"
    if atype in {
        "tf32",
        "f16x2",
        "bf16x2",
        "e4m3x4",
        "e5m2x4",
        "e2m1x4",
        "e2m3x4",
        "e3m2x4",
    }:
        return "uint32_t", "r"
    raise ValueError(f"Unsupported PTX cvt source type {atype!r}")


def _cvt_form_parts(form_name: str, *args):
    form = _CVT_FORMS[form_name]
    operands, attrs = _parse_form_attrs(form, args)
    dtype = _resolve(form.dtype, attrs)
    atype = _resolve(form.atype, attrs)
    has_scale = bool(attrs.get("scaled", ""))
    expected_operands = form.primary_count + int(form.rbits) + int(has_scale)
    if len(operands) != expected_operands:
        raise ValueError(
            f"PTX cvt form {form_name!r} expects {expected_operands} operands, got {len(operands)}"
        )

    source_types = [atype] * form.primary_count
    if form.rbits:
        source_types.append("u32")
    if has_scale:
        source_types.append("u16")
    source_info = [_input_info(source_type) for source_type in source_types]
    source_names = ["a", "b", "e", "f"][: form.primary_count]
    if form.rbits:
        source_names.append("rbits")
    if has_scale:
        source_names.append("scale_factor")

    return_type, result_type, out_constraint, tvm_return_type = _result_info(dtype)
    signature = (
        "("
        + ", ".join(
            f"{c_type} {name}" for name, (c_type, _constraint) in zip(source_names, source_info)
        )
        + ")"
    )

    instruction = f"cvt{_modifier_string(form, attrs)}.{dtype}.{atype}"
    name = f"tvm_builtin_ptx_{_safe(instruction)}"
    placeholders = [f"%{index}" for index in range(1, len(source_names) + 1)]
    if form.grouped_primary:
        primary = "{" + ", ".join(placeholders[: form.primary_count]) + "}"
        asm_operands = [primary, *placeholders[form.primary_count :]]
    else:
        asm_operands = placeholders
    asm_prefix = []
    asm_suffix = []
    result_operand = "%0"
    if atype == "e2m1x2":
        asm_prefix.extend((".reg .b8 raw_a;", "cvt.u8.u16 raw_a, %1;"))
        asm_operands[0] = "raw_a"
    if dtype == "e2m1x2":
        asm_prefix.append(".reg .b8 raw_result;")
        result_operand = "raw_result"
        asm_suffix.append("cvt.u16.u8 %0, raw_result;")
    operand_text = ", ".join([result_operand, *asm_operands])
    inputs = ", ".join(
        f'"{constraint}"({source_name})'
        for source_name, (_c_type, constraint) in zip(source_names, source_info)
    )
    asm_instructions = [*asm_prefix, f"{instruction} {operand_text};", *asm_suffix]
    if asm_prefix or asm_suffix:
        asm_text = "\\n\\t".join(asm_instructions)
        body = (
            f"    {result_type} result;\n"
            f'    asm volatile("{{\\n\\t{asm_text}\\n\\t}}"\n'
            f'                 : "={out_constraint}"(result)\n'
            f"                 : {inputs});\n"
            "    return result;"
        )
    else:
        body = (
            f"    {result_type} result;\n"
            f'    asm volatile("{instruction} {operand_text};"\n'
            f'                 : "={out_constraint}"(result)\n'
            f"                 : {inputs});\n"
            "    return result;"
        )
    return name, signature, return_type, tvm_return_type, body


def _register_cvt_form(form_name: str) -> None:
    form = _CVT_FORMS[form_name]
    device_intrinsic(
        f"_ptx_cvt_{form_name}",
        n_attrs=len(form.attrs),
        helper_name=lambda *a, _form=form_name: _cvt_form_parts(_form, *a)[0],
        c_signature=lambda *a, _form=form_name: _cvt_form_parts(_form, *a)[1],
        return_type=lambda *a, _form=form_name: _cvt_form_parts(_form, *a)[2],
        tvm_return_type=lambda *a, _form=form_name: _cvt_form_parts(_form, *a)[3],
        body=lambda *a, _form=form_name: _cvt_form_parts(_form, *a)[4],
    )


for _form_name in _CVT_FORMS:
    _register_cvt_form(_form_name)
del _form_name


def _reject_modifiers(form_name, *, ftz, sat, relu, satfinite, scaled):
    if ftz or sat:
        raise ValueError(f"PTX cvt form {form_name} does not accept .ftz or .sat")
    if relu and "relu" not in _CVT_FORMS[form_name].attrs:
        raise ValueError(f"PTX cvt form {form_name} does not accept .relu")
    if satfinite and "satfinite" not in _CVT_FORMS[form_name].attrs:
        raise ValueError(f"PTX cvt form {form_name} does not accept .satfinite")
    if scaled and "scaled" not in _CVT_FORMS[form_name].attrs:
        raise ValueError(f"PTX cvt form {form_name} does not accept .scaled")


def _classify_cvt_form(
    dtype,
    atype,
    rounding,
    ftz,
    sat,
    relu,
    satfinite,
    scaled,
    n_values,
    has_rbits,
):
    """Return ``(form_name, form_attrs)`` for one public ``ptx.cvt`` call."""

    def require_values(expected):
        if n_values != expected:
            raise ValueError(
                f"PTX cvt {dtype}.{atype} expects {expected} primary operands, got {n_values}"
            )

    # f16 / bf16 destination forms from f32.
    if atype == "f32" and dtype in {"f16", "f16x2", "bf16", "bf16x2"}:
        _reject_modifiers(
            "frnd2_f16_f32",
            ftz=ftz,
            sat=sat,
            relu=relu,
            satfinite=satfinite,
            scaled=scaled,
        )
        if rounding == "rs":
            if dtype not in {"f16x2", "bf16x2"}:
                raise ValueError("PTX cvt.rs from f32 is available only for packed x2 outputs")
            require_values(2)
            if not has_rbits:
                raise ValueError("PTX cvt.rs requires the rbits operand")
            form_name = f"rs_{dtype}_f32"
            return form_name, [relu, satfinite]
        if rounding not in {"rn", "rz"}:
            raise ValueError(
                "PTX f16/bf16 destination forms require rounding in {'rn', 'rz', 'rs'}"
            )
        if has_rbits:
            raise ValueError("rbits is valid only with rounding='rs'")
        require_values(2 if dtype.endswith("x2") else 1)
        form_name = f"frnd2_{dtype}_f32"
        return form_name, [rounding, relu, satfinite]

    # tf32 destination forms.
    if dtype == "tf32" and atype == "f32":
        require_values(1)
        if has_rbits or scaled or ftz or sat:
            raise ValueError("PTX tf32 destination forms do not accept rbits, scaled, ftz, or sat")
        if rounding == "rna":
            if relu:
                raise ValueError("cvt.rna.tf32.f32 does not accept .relu")
            return "rna_tf32_f32", [satfinite]
        if rounding not in {"rn", "rz"}:
            raise ValueError("PTX tf32 destination forms require rounding in {'rna', 'rn', 'rz'}")
        return "frnd2_tf32_f32", [rounding, satfinite, relu]

    # f8/f4/f6 families.
    families = (
        ("f8", _F8X2_TYPES, _F8X4_TYPES),
        ("f4", _F4X2_TYPES, _F4X4_TYPES),
        ("f6", _F6X2_TYPES, _F6X4_TYPES),
    )
    for family, x2_types, x4_types in families:
        if dtype in x2_types and atype == "f32":
            require_values(2)
            if rounding != "rn" or not satfinite:
                raise ValueError(f"PTX {family}x2.f32 requires rounding='rn' and satfinite=True")
            if has_rbits or scaled or ftz or sat:
                raise ValueError(f"PTX {family}x2.f32 does not accept rbits, scaled, ftz, or sat")
            return f"rn_satfinite_{family}x2_f32", [dtype, relu]
        if dtype in x2_types and atype in _FP16X2_TYPES:
            require_values(1)
            if rounding != "rn" or not satfinite:
                raise ValueError(
                    f"PTX {family}x2.{atype} requires rounding='rn' and satfinite=True"
                )
            if has_rbits or scaled or ftz or sat:
                raise ValueError(
                    f"PTX {family}x2.{atype} does not accept rbits, scaled, ftz, or sat"
                )
            return f"rn_satfinite_{family}x2_fp16x2", [dtype, atype, relu]
        if dtype == "f16x2" and atype in x2_types:
            require_values(1)
            if rounding != "rn":
                raise ValueError(f"PTX f16x2.{family}x2 requires rounding='rn'")
            if has_rbits or scaled or ftz or sat or satfinite:
                raise ValueError(
                    f"PTX f16x2.{family}x2 does not accept rbits, scaled, ftz, sat, or satfinite"
                )
            return f"rn_f16x2_{family}x2", [atype, relu]
        if dtype == "bf16x2" and atype in x2_types:
            require_values(1)
            if rounding != "rn":
                raise ValueError(f"PTX bf16x2.{family}x2 requires rounding='rn'")
            if has_rbits or ftz or sat:
                raise ValueError(f"PTX bf16x2.{family}x2 does not accept rbits, ftz, or sat")
            return f"rn_bf16x2_{family}x2", [atype, relu, satfinite, scaled]
        if dtype in x4_types and atype == "f32":
            require_values(4)
            if rounding != "rs" or not satfinite or not has_rbits:
                raise ValueError(
                    f"PTX {family}x4.f32 requires rounding='rs', satfinite=True, and rbits"
                )
            if scaled or ftz or sat:
                raise ValueError(f"PTX {family}x4.f32 does not accept scaled, ftz, or sat")
            return f"rs_satfinite_{family}x4_f32", [dtype, relu]

    # ue8m0 forms.
    if dtype == "ue8m0x2" and atype in {"f32", "bf16x2"}:
        require_values(2 if atype == "f32" else 1)
        if rounding not in {"rz", "rp"}:
            raise ValueError("PTX ue8m0x2 destination forms require rounding in {'rz', 'rp'}")
        if has_rbits or scaled or ftz or sat or relu:
            raise ValueError("PTX ue8m0x2 destination forms accept only optional .satfinite")
        return f"frnd3_ue8m0x2_{atype}", [rounding, satfinite]
    if dtype == "bf16x2" and atype == "ue8m0x2":
        require_values(1)
        if rounding != "rn" or has_rbits or scaled or ftz or sat or relu or satfinite:
            raise ValueError("PTX bf16x2.ue8m0x2 accepts only rounding='rn'")
        return "rn_bf16x2_ue8m0x2", []

    # s2f6 forms.
    if dtype == "s2f6x2" and atype in {"f32", "bf16x2"}:
        require_values(2 if atype == "f32" else 1)
        if rounding != "rn" or not satfinite:
            raise ValueError(f"PTX s2f6x2.{atype} requires rounding='rn' and satfinite=True")
        if has_rbits or ftz or sat:
            raise ValueError(f"PTX s2f6x2.{atype} does not accept rbits, ftz, or sat")
        return f"rn_satfinite_s2f6x2_{atype}", [relu, scaled]
    if dtype == "bf16x2" and atype == "s2f6x2":
        require_values(1)
        if rounding != "rn" or has_rbits or ftz or sat:
            raise ValueError("PTX bf16x2.s2f6x2 requires rounding='rn'")
        return "rn_bf16x2_s2f6x2", [satfinite, relu, scaled]

    # Generic scalar grammar forms.
    if dtype in _BASIC_TYPES and atype in _BASIC_TYPES:
        require_values(1)
        if has_rbits or relu or satfinite or scaled:
            raise ValueError(
                "generic scalar PTX cvt does not accept rbits, relu, satfinite, or scaled"
            )
        if rounding in _INTEGER_ROUNDING:
            return "irnd", [dtype, atype, rounding, ftz, sat]
        if rounding in _FLOAT_ROUNDING:
            return "frnd", [dtype, atype, rounding, ftz, sat]
        if not rounding:
            return "scalar", [dtype, atype, ftz, sat]
        raise ValueError(f"Unsupported generic scalar PTX cvt rounding {rounding!r}")

    raise ValueError(
        f"No PTX cvt instruction form for dtype={dtype!r}, atype={atype!r}, rounding={rounding!r}"
    )


@register_codegen("ptx_cvt")
def codegen_ptx_cvt(*args):
    """Classify one public ``ptx.cvt`` invocation into an ISA grammar form."""
    if len(args) < 11:
        raise ValueError("ptx.cvt is missing its trailing instruction attrs")
    (
        n_values,
        has_rbits,
        has_scale,
        dtype,
        atype,
        rounding,
        ftz,
        sat,
        relu,
        satfinite,
        scaled,
    ) = args[-11:]
    operands = list(args[:-11])
    n_values = int(n_values)
    has_rbits = _as_bool(has_rbits)
    has_scale = _as_bool(has_scale)
    dtype = parse_str(dtype)
    atype = parse_str(atype)
    rounding = parse_str(rounding)
    ftz = _as_bool(ftz)
    sat = _as_bool(sat)
    relu = _as_bool(relu)
    satfinite = _as_bool(satfinite)
    scaled = parse_str(scaled)

    if len(operands) != n_values + int(has_rbits) + int(has_scale):
        raise ValueError("ptx.cvt operand metadata does not match the provided operands")
    if has_rbits != (rounding == "rs"):
        raise ValueError("ptx.cvt rbits operand must be present exactly when rounding='rs'")
    if has_scale != bool(scaled):
        raise ValueError("ptx.cvt scale-factor operand must match the scaled attr")

    form_name, form_attrs = _classify_cvt_form(
        dtype,
        atype,
        rounding,
        ftz,
        sat,
        relu,
        satfinite,
        scaled,
        n_values,
        has_rbits,
    )
    result = CODEGEN_REGISTRY[f"tirx._ptx_cvt_{form_name}"](operands + form_attrs)
    return result[0] if isinstance(result, tuple) else result
