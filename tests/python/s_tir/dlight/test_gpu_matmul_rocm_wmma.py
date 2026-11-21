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
# pylint: disable=missing-docstring
"""dlight Matmul on RDNA (gfx11, wave32): WMMA-tensorized schedule for fp16 -> fp32 GEMMs."""

import numpy as np
import pytest

import tvm
import tvm_tirx
from tvm import s_tir, te
from tvm.s_tir import dlight
from tvm.s_tir.dlight.analysis import get_root_block
from tvm_tirx import tirx

RDNA_TARGET = {"kind": "rocm", "mcpu": "gfx1151", "thread_warp_size": 32}
GROUP = 128


def _rows(m):
    return tirx.Var("m", "int32") if m is None else m


def _gemm(m, n, k, epilogue="none"):
    """fp16 x fp16 -> fp32 GEMM (B in [N, K]); epilogue "cast_bias" adds a cast and a bias."""
    rows = _rows(m)
    a = te.placeholder((rows, k), "float16", name="A")
    b = te.placeholder((n, k), "float16", name="B")
    r = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (rows, n),
        lambda i, j: te.sum(a[i, r].astype("float32") * b[j, r].astype("float32"), axis=r),
        name="C",
    )
    if epilogue == "none":
        return tvm_tirx.IRModule({"main": te.create_function([a, b, c])})
    bias = te.placeholder((n,), "float16", name="bias")
    d = te.compute((rows, n), lambda i, j: c[i, j].astype("float16") + bias[j], name="D")
    return tvm_tirx.IRModule({"main": te.create_function([a, b, bias, d])})


def _w4_gemm(m, n, k):
    """int4 weights (8 per uint32, zero point 8, fp16 scale per 128) dequantized in a prologue."""
    rows = _rows(m)
    x = te.placeholder((rows, k), "float16", name="X")
    packed = te.placeholder((n, k // 8), "uint32", name="W")
    scale = te.placeholder((n, k // GROUP), "float16", name="S")

    def dequantize(i, j):
        nibble = (packed[i, j // 8] >> ((j % 8) * 4).astype("uint32")) & tirx.const(15, "uint32")
        return (nibble.astype("float16") - tirx.const(8, "float16")) * scale[i, j // GROUP]

    w = te.compute((n, k), dequantize, name="dequantize")
    r = te.reduce_axis((0, k), name="k")
    acc = te.compute(
        (rows, n),
        lambda i, j: te.sum(x[i, r].astype("float32") * w[j, r].astype("float32"), axis=r),
        name="matmul",
    )
    out = te.compute((rows, n), lambda i, j: acc[i, j].astype("float16"), name="cast")
    return tvm_tirx.IRModule({"main": te.create_function([x, packed, scale, out])})


def _make(kind, m, n, k):
    return _w4_gemm(m, n, k) if kind == "w4" else _gemm(m, n, k, kind)


def _schedule(mod, target):
    return dlight.ApplyDefaultSchedule(dlight.gpu.Matmul(), target=target)(mod)


def _all_blocks_in_kernels(mod):
    sch = s_tir.Schedule(mod["main"])
    for block in sch.get_child_blocks(get_root_block(sch)):
        tags = [
            str(sch.get(lp).annotations.get("thread_binding", "")) for lp in sch.get_loops(block)
        ]
        if not any(tag.startswith("blockIdx") for tag in tags):
            return False
    return True


@pytest.mark.parametrize("kind", ["none", "cast_bias", "w4"])
@pytest.mark.parametrize("m", [None, 512])
def test_rdna_matmul_selects_wmma(kind, m):
    mod = _schedule(_make(kind, m, 2048, 1536), tvm_tirx.target.Target(RDNA_TARGET))
    assert "wmma" in mod["main"].script()
    # Prologue (padding, dequantization) and epilogue (cast, bias) all run inside GPU kernels.
    assert _all_blocks_in_kernels(mod)


def test_rdna_small_static_m_uses_simt():
    mod = _schedule(_gemm(8, 2048, 1536), tvm_tirx.target.Target(RDNA_TARGET))
    assert "wmma" not in mod["main"].script()


def test_non_rdna_rocm_unchanged():
    mod = _schedule(
        _gemm(512, 2048, 1536), tvm_tirx.target.Target({"kind": "rocm", "mcpu": "gfx942"})
    )
    assert "wmma" not in mod["main"].script()


def _has_rdna_wmma():
    if not tvm_tirx.rocm(0).exist:
        return False
    from tvm_tirx.support import rocm  # pylint: disable=import-outside-toplevel

    if hasattr(rocm, "have_rdna_wmma"):
        return rocm.have_rdna_wmma()
    return str(tvm_tirx.rocm(0).compute_version).split(".", maxsplit=1)[0] in ("11", "12")


def _inputs(kind, m, n, k, rng):
    if kind == "w4":
        x = rng.uniform(-1, 1, (m, k)).astype("float16")
        q = rng.integers(0, 16, (n, k), dtype=np.uint32)
        shifts = np.arange(8, dtype=np.uint32) * 4
        packed = (q.reshape(n, k // 8, 8) << shifts).sum(-1).astype(np.uint32)
        scale = rng.uniform(0.001, 0.02, (n, k // GROUP)).astype("float16")
        w = (q.astype("float32") - 8) * np.repeat(scale.astype("float32"), GROUP, axis=1)
        return [x, packed, scale], x.astype("float32") @ w.T, "float16"
    a = rng.uniform(-1, 1, (m, k)).astype("float16")
    b = rng.uniform(-1, 1, (n, k)).astype("float16")
    ref = a.astype("float32") @ b.astype("float32").T
    if kind == "none":
        return [a, b], ref, "float32"
    bias = rng.uniform(-1, 1, (n,)).astype("float16")
    return [a, b, bias], ref + bias.astype("float32"), "float16"


@pytest.mark.gpu
@pytest.mark.skipif(not _has_rdna_wmma(), reason="needs an RDNA GPU with WMMA")
@pytest.mark.parametrize("kind", ["none", "cast_bias", "w4"])
@pytest.mark.parametrize("m, rows", [(512, 512), (None, 300)])
def test_rdna_matmul_wmma_numeric(tmp_path, kind, m, rows):
    n, k = 1024, 512
    target = tvm_tirx.target.Target({"kind": "rocm", "thread_warp_size": 32})
    lib = tvm.compile(_schedule(_make(kind, m, n, k), target), target=target)
    lib.export_library(str(tmp_path / "gemm.so"))
    f = tvm_tirx.runtime.load_module(str(tmp_path / "gemm.so"))["main"]
    ins, ref, out_dtype = _inputs(kind, rows, n, k, np.random.default_rng(0))
    dev = tvm_tirx.rocm(0)
    args = [tvm_tirx.runtime.tensor(x, dev) for x in ins]
    out = tvm_tirx.runtime.tensor(np.zeros((rows, n), out_dtype), dev)
    f(*args, out)
    dev.sync()
    np.testing.assert_allclose(out.numpy().astype("float32"), ref, rtol=1e-2, atol=1e-2)
