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
"""Tests for the PTX ``cvt`` instruction-family registration."""

import pytest

from tvm.backend.cuda.operator.intrinsics.cvt import _CVT_FORMS, _cvt_form_parts
from tvm.backend.cuda.operator.intrinsics.registry import CODEGEN_REGISTRY

# One concrete expansion for every syntax form in the PTX ``cvt`` grammar.
# Args are register placeholders followed by the form-specific attrs.
_FORM_CASES = [
    ("scalar", (0, "f32", "f32", 0, 0), "cvt.f32.f32 %0, %1;"),
    ("irnd", (0, "f32", "f32", "rni", 0, 0), "cvt.rni.f32.f32 %0, %1;"),
    ("frnd", (0, "f32", "f64", "rn", 0, 0), "cvt.rn.f32.f64 %0, %1;"),
    ("frnd2_f16_f32", (0, "rn", 0, 0), "cvt.rn.f16.f32 %0, %1;"),
    ("frnd2_f16x2_f32", (0, 0, "rz", 1, 1), "cvt.rz.relu.satfinite.f16x2.f32"),
    ("rs_f16x2_f32", (0, 0, 0, 1, 0), "cvt.rs.relu.f16x2.f32"),
    ("frnd2_bf16_f32", (0, "rn", 0, 1), "cvt.rn.satfinite.bf16.f32"),
    ("frnd2_bf16x2_f32", (0, 0, "rz", 0, 0), "cvt.rz.bf16x2.f32"),
    ("rs_bf16x2_f32", (0, 0, 0, 0, 1), "cvt.rs.satfinite.bf16x2.f32"),
    ("rna_tf32_f32", (0, 1), "cvt.rna.satfinite.tf32.f32"),
    ("frnd2_tf32_f32", (0, "rn", 1, 1), "cvt.rn.satfinite.relu.tf32.f32"),
    (
        "rn_satfinite_f8x2_f32",
        (0, 0, "e4m3x2", 1),
        "cvt.rn.satfinite.relu.e4m3x2.f32",
    ),
    (
        "rn_satfinite_f8x2_fp16x2",
        (0, "e5m2x2", "bf16x2", 0),
        "cvt.rn.satfinite.e5m2x2.bf16x2",
    ),
    ("rn_f16x2_f8x2", (0, "e4m3x2", 1), "cvt.rn.relu.f16x2.e4m3x2"),
    (
        "rn_bf16x2_f8x2",
        (0, 0, "e4m3x2", 1, 1, "n2::ue8m0"),
        "cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2",
    ),
    (
        "rs_satfinite_f8x4_f32",
        (0, 0, 0, 0, 0, "e4m3x4", 1),
        "cvt.rs.relu.satfinite.e4m3x4.f32",
    ),
    (
        "rn_satfinite_f4x2_f32",
        (0, 0, "e2m1x2", 0),
        "cvt.rn.satfinite.e2m1x2.f32",
    ),
    (
        "rn_satfinite_f4x2_fp16x2",
        (0, "e2m1x2", "f16x2", 1),
        "cvt.rn.satfinite.relu.e2m1x2.f16x2",
    ),
    ("rn_f16x2_f4x2", (0, "e2m1x2", 0), "cvt.rn.f16x2.e2m1x2"),
    (
        "rn_bf16x2_f4x2",
        (0, 0, "e2m1x2", 0, 0, "n2::ue8m0"),
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2",
    ),
    (
        "rs_satfinite_f4x4_f32",
        (0, 0, 0, 0, 0, "e2m1x4", 0),
        "cvt.rs.satfinite.e2m1x4.f32",
    ),
    (
        "rn_satfinite_f6x2_f32",
        (0, 0, "e2m3x2", 1),
        "cvt.rn.satfinite.relu.e2m3x2.f32",
    ),
    (
        "rn_satfinite_f6x2_fp16x2",
        (0, "e3m2x2", "bf16x2", 0),
        "cvt.rn.satfinite.e3m2x2.bf16x2",
    ),
    ("rn_f16x2_f6x2", (0, "e2m3x2", 1), "cvt.rn.relu.f16x2.e2m3x2"),
    (
        "rn_bf16x2_f6x2",
        (0, "e3m2x2", 0, 1, ""),
        "cvt.rn.satfinite.bf16x2.e3m2x2",
    ),
    (
        "rs_satfinite_f6x4_f32",
        (0, 0, 0, 0, 0, "e2m3x4", 1),
        "cvt.rs.relu.satfinite.e2m3x4.f32",
    ),
    ("frnd3_ue8m0x2_f32", (0, 0, "rz", 0), "cvt.rz.ue8m0x2.f32"),
    (
        "frnd3_ue8m0x2_bf16x2",
        (0, "rp", 1),
        "cvt.rp.satfinite.ue8m0x2.bf16x2",
    ),
    ("rn_bf16x2_ue8m0x2", (0,), "cvt.rn.bf16x2.ue8m0x2"),
    (
        "rn_satfinite_s2f6x2_f32",
        (0, 0, 0, 1, "n2::ue8m0"),
        "cvt.rn.satfinite.relu.scaled::n2::ue8m0.s2f6x2.f32",
    ),
    (
        "rn_satfinite_s2f6x2_bf16x2",
        (0, 0, ""),
        "cvt.rn.satfinite.s2f6x2.bf16x2",
    ),
    (
        "rn_bf16x2_s2f6x2",
        (0, 0, 1, 1, "n2::ue8m0"),
        "cvt.rn.satfinite.relu.scaled::n2::ue8m0.bf16x2.s2f6x2",
    ),
]


@pytest.mark.parametrize(("form_name", "args", "instruction"), _FORM_CASES)
def test_ptx_cvt_form_registration(form_name, args, instruction):
    helper_name, _signature, _c_type, _tvm_type, body = _cvt_form_parts(form_name, *args)

    assert instruction in body
    assert helper_name.startswith("tvm_builtin_ptx_cvt_")
    assert f"tirx._ptx_cvt_{form_name}" in CODEGEN_REGISTRY


def test_ptx_cvt_test_matrix_covers_every_registered_form():
    assert {case[0] for case in _FORM_CASES} == set(_CVT_FORMS)


def test_ptx_cvt_e2m1x2_uses_b8_staging():
    _name, signature, c_type, tvm_type, body = _cvt_form_parts(
        "rn_satfinite_f4x2_f32", 0, 0, "e2m1x2", 0
    )
    assert signature == "(float a, float b)"
    assert (c_type, tvm_type) == ("uint8_t", "uint8")
    assert ".reg .b8 raw_result;" in body
    assert "cvt.u16.u8 %0, raw_result;" in body

    _name, signature, _c_type, _tvm_type, body = _cvt_form_parts("rn_f16x2_f4x2", 0, "e2m1x2", 0)
    assert signature == "(uint16_t a)"
    assert ".reg .b8 raw_a;" in body
    assert "cvt.u8.u16 raw_a, %1;" in body
