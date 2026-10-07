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
"""The compile-time tcgen05 descriptor encoders must agree with the C ones.

``encode_smem_descriptor_base_uint64`` and
``encode_instr_descriptor_block_scaled_uint32`` re-derive, in Python, bit
layouts that otherwise live only in a C struct (``cuda/cpp/descriptors.py``). A
kernel whose descriptor inputs are all compile-time constants uses them to bake
a literal instead of calling the runtime encoder, which is an opaque helper in
the generated CUDA.

Re-deriving a bit layout by hand is exactly the kind of thing that is silently
wrong, so these run both encoders on the same inputs and compare.
"""

import shutil
import subprocess

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda import op as cuda_op
from tvm.backend.cuda.codegen.header import header_generator
from tvm.backend.cuda.codegen.registry import get_codegen
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_instr_descriptor_dense_uint32,
    encode_smem_descriptor_base_uint64,
)
from tvm.script import tirx as T
from tvm.testing import env

TARGET = tvm.target.Target("cuda")


def _compile_and_run(kernel, n_out):
    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=TARGET, tir_pipeline="tirx")
    result = {}

    def go():
        dev = tvm.cuda(0)
        out = tvm.runtime.tensor(np.zeros(n_out, dtype="uint64"), device=dev)
        mod(out)
        result["out"] = out.numpy()

    tvm.testing.run_with_gpu_lock(go)
    return result["out"]


# One case per axis the SMEM encoder branches on: every swizzle enum, and
# offsets that exercise both 14-bit fields.
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "ldo,sdo,swizzle",
    [(0, 8, 0), (0, 8, 3), (64, 128, 2), (32, 64, 1), (16, 16, 4)],
)
def test_smem_descriptor_matches_runtime_encoder(ldo, sdo, swizzle):
    """`base | (addr >> 4)` must reproduce the C bitfield fill exactly."""

    @T.prim_func
    def kernel(out: T.Tensor((2,), "uint64")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_tensor((64,), "uint32", scope="shared")
        smem[tx] = T.uint32(0)
        smem[tx + 32] = T.uint32(0)
        if tx == 0:
            desc = T.local_scalar("uint64")
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc), smem.ptr_to([0]), ldo=ldo, sdo=sdo, swizzle=swizzle
            )
            out[0] = desc
            # The shared address is the encoder's only runtime input; hand it
            # back so the comparison can OR it into the Python constant.
            out[1] = T.cast(T.cuda.cvta_generic_to_shared(smem.ptr_to([0])), "uint64")

    got = _compile_and_run(kernel, 2)
    want = encode_smem_descriptor_base_uint64(ldo, sdo, swizzle) | ((int(got[1]) >> 4) & 0x3FFF)
    assert int(got[0]) == want, (
        f"runtime {int(got[0]):#018x} != compile-time {want:#018x} "
        f"for ldo={ldo} sdo={sdo} swizzle={swizzle}"
    )


# (M, N, K, a_dtype, b_dtype, trans_a, trans_b, cta_group)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "m,n,k,a_dtype,b_dtype,trans_a,trans_b,cta_group",
    [
        (128, 224, 32, "float8_e4m3fn", "float8_e4m3fn", False, False, 1),
        (128, 128, 32, "float8_e4m3fn", "float8_e4m3fn", True, True, 1),
        (256, 256, 32, "float8_e4m3fn", "float8_e5m2", False, True, 2),
        (128, 64, 64, "float4_e2m1fn", "float4_e2m1fn", False, False, 1),
        (256, 128, 96, "float4_e2m1fn", "float4_e2m1fn", False, False, 2),
        (128, 64, 96, "float4_e2m1fn", "float4_e2m1fn", False, False, 1),
    ],
)
def test_instr_descriptor_block_scaled_matches_runtime_encoder(
    m, n, k, a_dtype, b_dtype, trans_a, trans_b, cta_group
):
    @T.prim_func
    def kernel(out: T.Tensor((1,), "uint64")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            desc = T.local_scalar("uint32")
            T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc),
                d_dtype="float32",
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                M=m,
                N=n,
                K=k,
                trans_a=trans_a,
                trans_b=trans_b,
                n_cta_groups=cta_group,
            )
            out[0] = T.cast(desc, "uint64")

    got = _compile_and_run(kernel, 1)
    want = encode_instr_descriptor_block_scaled_uint32(
        M=m,
        N=n,
        K=k,
        d_dtype="float32",
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype="float8_e8m0fnu",
        trans_a=trans_a,
        trans_b=trans_b,
        cta_group=cta_group,
    )
    assert int(got[0]) == want, (
        f"runtime {int(got[0]):#010x} != compile-time {want:#010x} "
        f"for M={m} N={n} a={a_dtype} b={b_dtype} trans=({trans_a},{trans_b}) "
        f"cta_group={cta_group}"
    )


@pytest.mark.parametrize("cta_group,m", [(2, 256), (1, 128)])
def test_instr_descriptor_block_scaled_k96_validation_and_bit(cta_group, m):
    """Dense K=96 (PTX ISA 9.4, 9.7.18.2.1.1: sm_103a / sm_107a) sets Table 53's bit 31."""
    common = dict(
        M=m,
        N=128,
        K=96,
        d_dtype="float32",
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        trans_a=False,
        trans_b=False,
        cta_group=cta_group,
    )
    assert encode_instr_descriptor_block_scaled_uint32(**common) & (1 << 31)

    for changed in (
        # K=96 pairs cta_group::2 with M=256 and cta_group::1 with M=128 only.
        {"M": 128 if m == 256 else 256},
        {"cta_group": 1 if cta_group == 2 else 2},
        {"K": 95},
        {"is_sparse": True},
        {"a_dtype": "float8_e4m3fn", "b_dtype": "float8_e4m3fn"},
    ):
        with pytest.raises(ValueError, match="Invalid matrix shape"):
            encode_instr_descriptor_block_scaled_uint32(**(common | changed))


@pytest.mark.parametrize("block_scaled", [False, True])
def test_instr_descriptor_attrs_cuda_codegen(block_scaled, monkeypatch):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    builder = (
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled
        if block_scaled
        else T.cuda.tcgen05.encode_instr_descriptor
    )
    settings = (_BLOCK_SCALED if block_scaled else _DENSE) | {"neg_a": True}

    @T.prim_func
    def kernel(out: T.Tensor((1,), "uint32")):
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            desc = T.local_scalar("uint32")
            builder(T.address_of(desc), **settings)
            out[0] = desc

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=TARGET, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source("cuda")
    helper = "ptx_tcgen05_encode_instr_descriptor" + ("_block_scaled" if block_scaled else "")
    assert helper in source
    assert "_desc.a_negate_ = static_cast<uint8_t>(neg_a)" in source
    if block_scaled:
        assert "_desc.a_sf_id_ = 0" in source
        assert "_desc.b_sf_id_ = 0" in source


# These helpers are plain C++ bitfield fills, so they can also be exercised
# without a GPU. Compile the actual lowered helper against the CUDA header's
# descriptor layout rather than reimplementing its field assignments here.
_DENSE = dict(
    d_dtype="float32",
    a_dtype="float16",
    b_dtype="float16",
    M=128,
    N=128,
    K=16,
    trans_a=False,
    trans_b=False,
)
_BLOCK_SCALED = dict(
    d_dtype="float32",
    a_dtype="float8_e4m3fn",
    b_dtype="float8_e4m3fn",
    sfa_dtype="float8_e8m0fnu",
    sfb_dtype="float8_e8m0fnu",
    M=128,
    N=128,
    K=32,
    trans_a=False,
    trans_b=False,
)


def _descriptor_call(block_scaled, **changed):
    name = "tcgen05_encode_instr_descriptor" + ("_block_scaled" if block_scaled else "")
    settings = (_BLOCK_SCALED if block_scaled else _DENSE) | changed
    return getattr(cuda_op, "cuda_" + name)(tvm.tirx.Var("desc", "handle"), **settings)


@pytest.mark.parametrize("block_scaled", [False, True])
def test_instr_descriptor_typed_attrs_and_roundtrip(block_scaled):
    call = _descriptor_call(block_scaled)
    assert len(call.args) == 1
    assert call.ty == tvm.ir.PrimType("")
    assert [info.name for info in call.op.args_info] == ["desc"]
    assert call.op.attrs_type_key == call.attrs.__tvm_ffi_type_info__.type_key
    assert call.attrs.n_cta_groups == 1
    assert call.attrs.neg_a is False
    assert call.attrs.neg_b is False
    assert call.attrs.is_sparse is False
    if not block_scaled:
        assert call.attrs.sat_d is False

    for current in (call, _descriptor_call(block_scaled, M=256, n_cta_groups=2, neg_b=True)):
        func = tvm.tirx.PrimFunc([current.args[0]], tvm.tirx.Evaluate(current))
        source = func.script()
        assert "attrs=" not in source and "ty=" not in source
        assert "trans_a=False" in source
        assert "neg_a=" not in source
        assert ("n_cta_groups=2" in source) == (current.attrs.n_cta_groups == 2)
        tvm.ir.assert_structural_equal(
            func, tvm.script.from_source(source, extra_vars={"T": T, "I": tvm.script.ir})
        )


@pytest.mark.parametrize("block_scaled", [False, True])
def test_instr_descriptor_preserves_explicit_result(block_scaled):
    call = _descriptor_call(block_scaled)
    explicit = tvm.ir.Call.unchecked(
        call.op, call.args, attrs=call.attrs, ty=tvm.ir.PrimType("uint32")
    )
    func = tvm.tirx.PrimFunc([call.args[0]], tvm.tirx.Evaluate(explicit))
    source = func.script()
    assert "T.cuda.tcgen05_encode_instr_descriptor" in source
    assert 'ty="uint32"' in source
    tvm.ir.assert_structural_equal(
        func, tvm.script.from_source(source, extra_vars={"T": T, "I": tvm.script.ir})
    )


@pytest.mark.parametrize("block_scaled", [False, True])
def test_instr_descriptor_schema_validation(block_scaled):
    call = _descriptor_call(block_scaled)
    api = getattr(cuda_op, call.op.name.removeprefix("tirx.cuda."))
    with pytest.raises(TypeError, match="cannot mix attrs"):
        api(call.args[0], attrs=call.attrs, M=128)
    with pytest.raises(TypeError, match="multiple values"):
        api(call.args[0], desc=call.args[0], attrs=call.attrs)
    with pytest.raises((TypeError, ValueError)):
        api(call.args[0])
    with pytest.raises((TypeError, ValueError)):
        api(call.args[0], M=128)
    for changed in ({"M": "128"}, {"M": tvm.tirx.Var("m", "int32")}, {"typo": 0}):
        settings = (_BLOCK_SCALED if block_scaled else _DENSE) | changed
        with pytest.raises((TypeError, ValueError)):
            api(call.args[0], **settings)
    with pytest.raises((TypeError, ValueError)):
        tvm.ir.Call(call.op, [], attrs=call.attrs, ty=call.ty)
    with pytest.raises((TypeError, ValueError)):
        tvm.ir.Call(call.op, call.args, attrs={"M": 128}, ty=call.ty)
    with pytest.raises((TypeError, ValueError)):
        tvm.ir.Call(call.op, call.args, attrs=_descriptor_call(not block_scaled).attrs, ty=call.ty)


@pytest.mark.parametrize(
    "block_scaled,changed,match",
    [
        (False, {"a_dtype": "int8"}, "Invalid multiplicand"),
        (False, {"M": 16}, "Invalid matrix shape"),
        (False, {"n_cta_groups": 3}, "n_cta_groups"),
        (False, {"sat_d": True}, "Invalid kind for saturate"),
        (
            False,
            {"d_dtype": "int32", "a_dtype": "int8", "b_dtype": "int8", "K": 32, "neg_a": True},
            "Invalid kind for negate",
        ),
        (
            False,
            {"a_dtype": "float4_e2m1fn", "b_dtype": "float4_e2m1fn", "K": 32, "trans_a": True},
            "Invalid a_dtype for transpose",
        ),
        (True, {"sfb_dtype": "float16"}, "Invalid multiplicand"),
        (True, {"N": 7}, "Invalid matrix shape"),
        (True, {"n_cta_groups": 0}, "n_cta_groups"),
        (
            True,
            {"a_dtype": "float4_e2m1fn", "b_dtype": "float4_e2m1fn", "K": 64, "trans_b": True},
            "Invalid b_dtype for transpose",
        ),
    ],
)
def test_instr_descriptor_preserves_validation(block_scaled, changed, match):
    with pytest.raises(ValueError, match=match):
        call = _descriptor_call(block_scaled, **changed)
        get_codegen(call.op.name)(call.args, call.attrs)


@pytest.mark.parametrize(
    "block_scaled,changed",
    [
        (False, {}),
        (False, {"d_dtype": "float16"}),
        (False, {"a_dtype": "bfloat16", "b_dtype": "bfloat16"}),
        (False, {"trans_a": True, "trans_b": True, "neg_a": True, "neg_b": True}),
        (False, {"M": 256, "N": 256, "n_cta_groups": 2}),
        (False, {"K": 32, "is_sparse": True}),
        (
            False,
            {"d_dtype": "int32", "a_dtype": "int8", "b_dtype": "uint8", "K": 32, "sat_d": True},
        ),
        (True, {}),
        (True, {"a_dtype": "float8_e5m2", "trans_b": True, "neg_a": True, "neg_b": True}),
        (True, {"M": 256, "N": 256, "n_cta_groups": 2}),
        (True, {"K": 64, "is_sparse": True}),
        (True, {"a_dtype": "float4_e2m1fn", "b_dtype": "float4_e2m1fn", "K": 64}),
        (True, {"a_dtype": "float4_e2m1fn", "b_dtype": "float4_e2m1fn", "K": 96}),
        (
            True,
            {
                "a_dtype": "float4_e2m1fn",
                "b_dtype": "float4_e2m1fn",
                "K": 64,
                "sfa_dtype": "float8_e4m3fn",
                "sfb_dtype": "float8_e4m3fn",
            },
        ),
    ],
)
def test_instr_descriptor_attrs_host_encoding(tmp_path, block_scaled, changed):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("need a host C++ compiler for descriptor bitfield helpers")
    call = _descriptor_call(block_scaled, **changed)
    lowered, tags = get_codegen(call.op.name)(call.args, call.attrs)
    arguments = ", ".join(str(int(arg.value)) for arg in lowered.args[2:-1])
    source = (
        "#include <cstdint>\n#include <cstdio>\n"
        "#define __forceinline__ inline\n#define __host__\n#define __device__\n"
        # The bitfield layouts need no driver declarations.
        + header_generator(tags).replace("#include <cuda.h>", "")
        + lowered.args[-1].value
        + f"\nint main() {{ uint32_t desc; {lowered.args[0].value}(&desc, {arguments});"
        + ' std::printf("%u", desc); }\n'
    )
    path = tmp_path / "descriptor.cc"
    path.write_text(source)
    binary = tmp_path / "descriptor"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True)
    actual = int(subprocess.check_output([str(binary)], text=True))
    settings = (_BLOCK_SCALED if block_scaled else _DENSE) | changed
    settings["cta_group"] = settings.pop("n_cta_groups", 1)
    if block_scaled:
        settings["sf_dtype"] = settings.pop("sfa_dtype")
        settings.pop("sfb_dtype")
        expected = encode_instr_descriptor_block_scaled_uint32(**settings)
        assert actual & 0x60000030 == 0  # Both scale-factor IDs remain zero.
    else:
        expected = encode_instr_descriptor_dense_uint32(**settings)
    assert actual == expected


if __name__ == "__main__":
    tvm.testing.main()
