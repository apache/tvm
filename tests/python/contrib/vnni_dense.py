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


@tvm.testing.requires_llvm
@pytest.mark.skip("skip because feature not enabled")
def test_fc_int8_acc32():

    for m, n, k in [(64, 800, 320), (64, 768, 512), (16, 256, 512), (128, 128, 128), (256, 512, 256), (1024, 1024, 1024)]:
        X = te.placeholder((m, k), name="X", dtype="uint8")
        W = te.placeholder((n, k), name="W", dtype="int8")

        memory_ops = m * k + n * k + 2 * m * n
        gops_per_mm = 2 * m * n * k

        # For LLVM < 8.0, it shows "'cascadelake' is not a recognized processor for this target
        # (ignoring processor)" error with the following setting. After LLVM 8.0 is enabled in the
        # test, we should use cascadelake setting.
        def verify(target="llvm -mcpu=cascadelake"):
            if not tvm.testing.device_enabled(target):
                print("skip because %s is not enabled..." % target)
                return

            dev = tvm.device(target, 0)
            pc = dot_16x1x16_uint8_int8_int32_cascadelake()
            ak = te.reduce_axis((0, k), name="k")

            # packedW = te.placeholder((n // 16, 16 * (k // 4), 4), name="packedW", dtype="int8")

            # t_fc = te.compute(
            #     (m, n),
            #     lambda i, j: te.sum(
            #         X[i, ak].astype("int32")
            #         * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4) * 16 + j % 16, ak % 4].astype("int32"),
            #         axis=ak,
            #     ),
            #     name="F",
            # )

            packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")

            t_fc = te.compute(
                (m, n),
                lambda i, j: te.sum(
                    X[i, ak].astype("int32")
                    * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4),  j % 16, ak % 4].astype("int32"),
                    axis=ak,
                ),
                name="F",
            )

            t_sch = te.create_schedule(t_fc.op)
            a_x, a_y = t_fc.op.axis
            (a_k,) = t_fc.op.reduce_axis

            a_yo, a_yi = t_sch[t_fc].split(a_y, factor=16)
            a_xo, a_xi = t_sch[t_fc].split(a_x, factor=32)
            a_ko, a_ki = t_sch[t_fc].split(a_k, factor=4)
            a_koo, a_koi = t_sch[t_fc].split(a_ko, factor=4)
            t_sch[t_fc].reorder(a_yo, a_xo, a_xi, a_koo, a_koi, a_yi, a_ki)

            t_sch[t_fc].unroll(a_koi)
            t_sch[t_fc].parallel(a_yo)
            t_sch[t_fc].tensorize(a_yi, pc)

            t_func = tvm.build(t_sch, [X, packedW, t_fc], target, name="intrinsic")
            t_evaluator = t_func.time_evaluator(t_func.entry_name, dev, number=10)

            # generate the plain data
            a_ = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
            b_ = np.random.uniform(1, 10, size=(n, k)).astype("int8")

            # packW = np.random.uniform(1, 10, size=(n // 16, 16 * (k // 4), 4)).astype("int8")
            # This occurs in pre_compute stage
            # for r_idx in range(n // 16):
            #     for s_idx in range(16 * (k // 4)):
            #         for t_idx in range(4):
            #             packW[r_idx][s_idx][t_idx] = b_[r_idx * 16 + s_idx % 16][
            #                 (s_idx // 16) * 4 + t_idx
            #             ]

            packW = np.random.uniform(1, 10, size=(n // 16, (k // 4), 16, 4)).astype("int8")

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
                "Tensorization: ({}, {}, {}), {:.2f} Gops/s".format(
                     m, n, k, gops_per_sec
                )
            )

        verify()


if __name__ == "__main__":
    # The test requires Cascade Lake and newer Intel machines to generate the
    # correct AVX512 VNNI instruction. So, disabling the test.

    test_fc_int8_acc32()
    # pass
