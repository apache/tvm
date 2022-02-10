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
import tvm.testing
from tvm import te
import numpy as np
from tvm.topi.x86.tensor_intrin import dot_16x1x16_uint8_int8_int32_cascadelake
from tvm.topi.x86.tensor_intrin import dot_16x1x16_uint8_int8_int32
import pytest


def dense_vnni_compute(X, packedW):
    m, k = X.shape
    n_o, _, n_i, _ = packedW.shape
    ak = te.reduce_axis((0, k), name="k")

    return te.compute(
        (m, n_o * n_i),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4),  j % 16, ak % 4].astype("int32"),
            axis=ak,
        ),
        name="F",
    )


def dense_vnni_schedule(t_fc):
    t_sch = te.create_schedule(t_fc.op)
    a_x, a_y = t_fc.op.axis
    (a_k,) = t_fc.op.reduce_axis

    a_yo, a_yi = t_sch[t_fc].split(a_y, factor=16)
    a_ko, a_ki = t_sch[t_fc].split(a_k, factor=4)
    a_xo, a_xi = t_sch[t_fc].split(a_x, factor=32)

    t_sch[t_fc].reorder(a_yo, a_xo, a_xi, a_ko, a_yi, a_ki)

    pc = dot_16x1x16_uint8_int8_int32_cascadelake()
    t_sch[t_fc].tensorize(a_yi, pc)

    return t_sch



@tvm.testing.requires_llvm
@pytest.mark.skip("skip because feature not enabled")
def test_fc_int8_acc32():
    m = 1024
    n = 1024
    k = 1024

    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")


    peak = 280
    print("Peak {} Gops/s".format(peak))
    memory_ops = m * k + n * k + 2 * m * n
    gops_per_mm = 2 * m * n * k

    # For LLVM < 8.0, it shows "'cascadelake' is not a recognized processor for this target
    # (ignoring processor)" error with the following setting. After LLVM 8.0 is enabled in the
    # test, we should use cascadelake setting.
    def verify(target="llvm -mcpu=cascadelake"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return

        t_fc = dense_vnni_compute(X, packedW)
        t_sch = dense_vnni_schedule(t_fc)

        print(tvm.lower(t_sch, [X, packedW, t_fc]))

        t_func = tvm.build(t_sch, [X, packedW, t_fc], target, name="intrinsic")

        dev = tvm.device(target, 0)
        t_evaluator = t_func.time_evaluator(t_func.entry_name, dev, number=10)
        # print(t_func.get_source("asm"))

        # generate the plain data
        a_ = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
        b_ = np.random.uniform(1, 10, size=(n, k)).astype("int8")

        packW = np.random.uniform(1, 10, size=(n // 16, (k // 4), 16, 4)).astype("int8")
        # This occurs in pre_compute stage
        for r_idx in range(n // 16):
            for ko in range(k // 4):
                for s_idx in range(16):
                    for t_idx in range(4):
                        packW[r_idx][ko][s_idx][t_idx] = b_[r_idx * 16 + s_idx][ko * 4 + t_idx]

        x = tvm.nd.array(a_, dev)
        w = tvm.nd.array(packW, dev)
        y = tvm.nd.array(np.zeros((m, n), dtype="int32"), dev)
        result = t_evaluator(x, w, y)

        gops_per_sec = gops_per_mm / result.mean / 1e9
        # verify the correctness
        tvm.testing.assert_allclose(y.numpy(), np.dot(a_, b_.T), rtol=0)
        print(
            "Tensorization: running time: {:.3f} ms, {:.2f} Gops/s, effiency: {:.2f}".format(
                result.mean * 1000, gops_per_sec, gops_per_sec / peak
            )
        )
        t_func.export_library("tensorize_acc32.o")

    verify()


if __name__ == "__main__":
    # The test requires Cascade Lake and newer Intel machines to generate the
    # correct AVX512 VNNI instruction. So, disabling the test.

    test_fc_int8_acc32()
