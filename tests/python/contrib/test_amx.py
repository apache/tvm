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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition

import tvm
from tvm import relay

from tvm import te
import tvm.testing
from tvm.topi.x86.tensor_intrin import dot_32x128x32_u8s8s32_sapphirerapids
from tvm.topi.x86.tensor_intrin import acc_32x32_int32_sapphirerapids
import numpy as np
import pytest


has_amx_runtime = pytest.mark.skipif(
    not tvm.get_global_func("runtime.amx_init", True), reason="AMX runtime not available"
)


@has_amx_runtime
@tvm.testing.requires_x86_amx
def test_amx_u8s8s32_matmul_tensorize():
    m = 1024
    k = 1024
    n = 1024

    # --------------------------Config---------------------------
    # Skip this test if "-mcpu=sapphirerapids" not supported by LLVM < 12.0
    target = "llvm -mcpu=sapphirerapids"
    dev = tvm.device(target, 0)
    if not tvm.testing.device_enabled(target):
        print("skip because %s is not enabled..." % target)
        return

    amx_init = tvm.get_global_func("runtime.amx_init")
    amx_tileconfig = tvm.get_global_func("runtime.amx_tileconfig")
    assert amx_init()
    assert amx_tileconfig(16, 64)  # config tile size to 16 rows by 64 columns.
    # --------------------------Compute--------------------------
    X = te.placeholder((m, k), name="X", dtype="uint8")
    ak = te.reduce_axis((0, k), name="k")
    packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")

    C = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        name="F",
    )

    # --------------------------Schedule--------------------------
    s = te.create_schedule(C.op)
    a_x, a_y = C.op.axis
    (a_k,) = C.op.reduce_axis

    CF = s.cache_write(C, "amx.tmm")
    a_xo, a_xi = s[C].split(a_x, factor=32)
    a_yo, a_yi = s[C].split(a_y, factor=32)
    s[C].reorder(a_xo, a_yo, a_xi, a_yi)

    s[CF].compute_at(s[C], a_yo)
    (a_k_f,) = CF.op.reduce_axis
    a_x_f, a_y_f = CF.op.axis

    a_xo_f, a_xi_f = s[CF].split(a_x_f, factor=32)
    a_yo_f, a_yi_f = s[CF].split(a_y_f, factor=32)
    a_ko_f, a_ki_f = s[CF].split(a_k_f, factor=128)
    s[CF].reorder(a_ko_f, a_xo_f, a_yo_f, a_ki_f, a_xi_f, a_yi_f)

    s[CF].tensorize(a_ki_f, dot_32x128x32_u8s8s32_sapphirerapids(LDA=k))
    s[C].tensorize(a_xi, acc_32x32_int32_sapphirerapids(LDC=n))

    lib = tvm.build(s, [X, packedW, C], target, name="intrinsic")
    asm = lib.get_source("asm")
    assert "tilezero" in asm
    assert "tileloaddt1" in asm
    assert "tdpbusd" in asm
    assert "tilestored" in asm

    # ----------------------- verify correctness --------------------------------
    # generate the plain data
    a = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
    b = np.random.uniform(1, 10, size=(n, k)).astype("int8")
    packW = np.random.uniform(1, 10, size=(n // 16, k // 4, 16, 4)).astype("int8")

    # This should occurs in pre_pack (constant folding) stage,
    # from plain data to blocked data(NC16n4c)
    for i_n in range(n):
        for i_k in range(k):
            packW[i_n // 16][i_k // 4][i_n % 16][i_k % 4] = b[i_n][i_k]

    x = tvm.nd.array(a, dev)
    w = tvm.nd.array(packW, dev)
    y = tvm.nd.array(np.zeros((m, n), dtype="int32"), dev)
    t_evaluator = lib.time_evaluator(lib.entry_name, dev, number=100)
    result = t_evaluator(x, w, y)
    print(result)
    tvm.testing.assert_allclose(y.numpy(), np.dot(a.astype("int32"), b.T.astype("int32")), rtol=0)


@has_amx_runtime
@tvm.testing.requires_x86_amx
def test_amx_check_support():
    amx_init = tvm.get_global_func("runtime.amx_init")
    amx_tileconfig = tvm.get_global_func("runtime.amx_tileconfig")
    assert amx_init()
    assert amx_tileconfig(16, 64)


if __name__ == "__main__":
    pytest.main([__file__])
