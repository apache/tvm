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

"""Explicit scalar composites and native packed register instructions."""

import numpy as np
import pytest

import tvm
from tvm.backend.cuda.axis import laneid, warpid
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


@pytest.mark.skipif(not env.has_cuda(), reason="CUDA required")
@pytest.mark.parametrize("operation", ["exp", "silu", "sqrt", "log2", "exp2"])
def test_ewise_composite(operation):
    @T.function(private=True)
    def scalar(x: T.float32, scale: T.float32, bias: T.float32) -> T.float32:
        if T.constexpr(operation == "exp"):
            return T.exp(x) * scale + bias
        elif T.constexpr(operation == "silu"):
            return (x / (T.float32(1) + T.exp(-x))) * scale + bias
        elif T.constexpr(operation == "sqrt"):
            return T.sqrt(x) * scale + bias
        elif T.constexpr(operation == "log2"):
            return T.log2(x) * scale + bias
        else:
            return T.exp2(x) * scale + bias

    layout = TileLayout(S[(2, 32, 7, 1) : (1 @ warpid, 1 @ laneid, 2, 1)])

    @T.function
    def kernel(A: T.Tensor((2, 32, 7, 1), "float32"), Output: T.Tensor((2, 32, 7, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        r = T.alloc_tensor(
            ty_args=[T.Tensor((2, 32, 7, 1), "float32", scope="local", layout=layout)]
        )
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A, scope="cta")
        T.cuda.tile.ewise(scalar, r, r, T.float32(2), T.float32(3), scope="cta")
        T.cuda.tile.apply(T.ptx["st.global.f32"], Output, r, scope="cta")

    f = tvm.tirx.compile(kernel, target="cuda", pipeline="tirx")
    x = np.linspace(0.1, 2, 448, dtype="float32").reshape(2, 32, 7, 1)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    operation = {
        "exp": np.exp,
        "silu": lambda x: x / (1 + np.exp(-x)),
        "sqrt": np.sqrt,
        "log2": np.log2,
        "exp2": np.exp2,
    }[operation]
    np.testing.assert_allclose(out.numpy(), operation(x) * 2 + 3, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell packed operations required")
@pytest.mark.parametrize(
    "dtype,suffix", [("float16", "f16x2"), ("bfloat16", "bf16x2"), ("float32", "f32x2")]
)
@pytest.mark.parametrize("opcode,kind", [("add", "binary"), ("mul", "binary"), ("fma", "ternary")])
def test_native_packed(dtype, suffix, opcode, kind):
    bits = tvm.DataType(dtype).bits * 2
    storage = "uint" + str(bits)
    instruction = f"{opcode}.rn.{suffix}"

    @T.function
    def kernel(
        A: T.Tensor((8, 1), storage),
        B: T.Tensor((8, 1), storage),
        Output: T.Tensor((8, 1), storage),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((8, 1), storage, layout="default")
        b = T.alloc_local((8, 1), storage, layout="default")
        T.cuda.tile.apply(T.ptx[f"ld.global.b{bits}"], a, A)
        T.cuda.tile.apply(T.ptx[f"ld.global.b{bits}"], b, B)
        if T.constexpr(kind == "ternary"):
            T.cuda.tile.apply(T.ptx[instruction], a, a, b, b)
        else:
            T.cuda.tile.apply(T.ptx[instruction], a, a, b)
        T.cuda.tile.apply(T.ptx[f"st.global.b{bits}"], Output, a)

    f = tvm.tirx.compile(kernel, target="cuda", pipeline="tirx")
    x = np.arange(16, dtype="float32").reshape(8, 2).astype(dtype)
    y = np.full((8, 2), 2, dtype=dtype)
    a = np.ascontiguousarray(x).view(storage)
    b = np.ascontiguousarray(y).view(storage)
    out = tvm.runtime.tensor(np.zeros_like(a), tvm.cuda())
    f(tvm.runtime.tensor(a, tvm.cuda()), tvm.runtime.tensor(b, tvm.cuda()), out)
    expected = (
        x.astype("float32") + 2
        if opcode == "add"
        else x.astype("float32") * 2 + (2 if opcode == "fma" else 0)
    ).astype(dtype)
    np.testing.assert_array_equal(out.numpy().view(dtype), expected)
