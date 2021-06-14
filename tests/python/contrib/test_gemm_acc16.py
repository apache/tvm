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
from tvm import te
import numpy as np
from tvm.topi.x86.tensor_intrin import dot_16x1x16_uint8_int8_int16


def benchmark_fc_int8_acc16():
    m = 128
    n = 128
    k = 128

    X = te.placeholder((m, k), name="X", dtype="uint8")
    W = te.placeholder((n, k), name="W", dtype="int8")

    peak = 512 / 16 * 2 * 2 * 2
    gops_per_mm = 2 * n * m * k
    print("Peak {} Gops/s \n".format(peak))

    def verify(target="llvm -mcpu=skylake-avx512"):
        if not tvm.runtime.enabled(target):
            print("skip because %s is not enabled..." % target)
            return

        dev = tvm.device(target, 0)
        X = te.placeholder((m, k), name="X", dtype="uint8")
        W = te.placeholder((n, k), name="W", dtype="int8")
        pc = dot_16x1x16_uint8_int8_int16()
        ak = te.reduce_axis((0, k), name="k")

        packedW = te.placeholder((n // 128, 128 * (k // 2), 2), name="packedW", dtype="int8")
        t_fc = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype("int16")
                * packedW[j // 128, (ak // 2) * 128 + j % 128, ak % 2].astype("int16"),
                axis=ak,
            ),
            name="F",
        )

        t_sch = te.create_schedule(t_fc.op)
        a_x, a_y = t_fc.op.axis
        (a_k,) = t_fc.op.reduce_axis

        a_yo, a_yi = t_sch[t_fc].split(a_y, factor=128)
        a_ko, a_ki = t_sch[t_fc].split(a_k, factor=2)

        a_xo, a_xi = t_sch[t_fc].split(a_x, factor=128)
        a_koo, a_koi = t_sch[t_fc].split(a_ko, factor=32)
        t_sch[t_fc].reorder(a_yo, a_xo, a_koo, a_xi, a_koi, a_yi, a_ki)

        t_sch[t_fc].tensorize(a_yi, pc)
        # print(tvm.lower(t_sch, [X, packedW, t_fc], simple_mode=True))
        t_func = tvm.build(t_sch, [X, packedW, t_fc], target, name="intrinsic")
        t_evaluator = t_func.time_evaluator(t_func.entry_name, dev, number=10)

        # generate the plain data
        a_ = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
        b_ = np.random.uniform(1, 10, size=(n, k)).astype("int8")

        packW = np.random.uniform(1, 10, size=(n // 128, 128 * (k // 2), 2)).astype("int8")
        # This occurs in pre_compute stage
        for r_idx in range(n // 128):
            for s_idx in range(128 * (k // 2)):
                for t_idx in range(2):
                    packW[r_idx][s_idx][t_idx] = b_[r_idx * 128 + s_idx % 128][
                        s_idx // 128 * 2 + t_idx
                    ]

        x = tvm.nd.array(a_, dev)
        w = tvm.nd.array(packW, dev)
        y = tvm.nd.array(np.zeros((m, n), dtype="int16"), dev)

        result = t_evaluator(x, w, y)
        gops_per_sec = gops_per_mm / result.mean / 1e9
        tvm.testing.assert_allclose(y.numpy(), np.dot(a_, b_.T), rtol=1e-5)
        print(
            "Tensorization: running time: {:.3f} ms, {:.2f} Gops/s, effiency: {:.2f}.".format(
                result.mean * 1000, gops_per_sec, gops_per_sec / peak
            )
        )
        # t_func.export_library("gemm_tensorize.o")

    verify()


if __name__ == "__main__":
    benchmark_fc_int8_acc16()
