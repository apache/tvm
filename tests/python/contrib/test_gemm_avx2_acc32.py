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
import numpy as np
from topi.x86.tensor_intrin import dot_16x1x16_int8_int8_int32_vnni
from topi.x86.tensor_intrin import dot_1x4x16_int8_int8_int32_avx2


def test_avx2_int8_gemm_acc32():
    m = 1024
    n = 1024
    k = 1024

    X = tvm.placeholder((m, k), name='X', dtype="uint8")
    W = tvm.placeholder((n, k), name='W', dtype="int8")

    memory_ops = m * k + n * k + 2 * m * n
    gops_per_mm = 2 * m * n * k

    def verify(target="llvm -mcpu=core-avx2"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return

        ctx = tvm.context(target, 0)
        pc = dot_1x4x16_int8_int8_int32_avx2()
        ak = tvm.reduce_axis((0, k), name='k')
        packedW = tvm.placeholder(
            (n // 16, 16 * (k // 4), 4), name='packedW', dtype="int8")

        t_fc = tvm.compute((m, n), lambda i, j: tvm.sum(X[i, ak].astype(
            "int32") * packedW[j // 16, (ak // 4) * 16 + j % 16, ak % 4].astype("int32"), axis=ak), name="F")
        t_sch = tvm.create_schedule(t_fc.op)
        a_x, a_y = t_fc.op.axis
        a_k, = t_fc.op.reduce_axis

        a_yo, a_yi = t_sch[t_fc].split(a_y, factor=16)
        a_xo, a_xi = t_sch[t_fc].split(a_x, factor=32)
        a_ko, a_ki = t_sch[t_fc].split(a_k, factor=4)
        a_koo, a_koi = t_sch[t_fc].split(a_ko, factor=4)
        t_sch[t_fc].reorder(a_yo, a_xo, a_xi, a_koo, a_koi, a_yi, a_ki)

        t_sch[t_fc].unroll(a_koi)
        t_sch[t_fc].tensorize(a_yi, pc)

        t_func = tvm.build(t_sch, [X, packedW, t_fc], target, name="intrinsic")
        t_evaluator = t_func.time_evaluator(t_func.entry_name, ctx, number=10)

        # generate the plain data
        a_ = np.random.uniform(1, 10, size=(m, k)).astype("uint8")
        b_ = np.random.uniform(1, 10, size=(n, k)).astype("int8")

        packW = np.random.uniform(1, 10, size=(
            n // 16, 16 * (k // 4), 4)).astype("int8")
        # This occurs in pre_compute stage
        for r_idx in range(n // 16):
            for s_idx in range(16 * (k // 4)):
                for t_idx in range(4):
                    packW[r_idx][s_idx][t_idx] = b_[r_idx * 16 + s_idx %
                                                    16][(s_idx // 16) * 4 + t_idx]

        x = tvm.nd.array(a_, ctx)
        w = tvm.nd.array(packW, ctx)
        y = tvm.nd.array(np.zeros((m, n), dtype="int32"), ctx)
        result = t_evaluator(x, w, y)

        gops_per_sec = gops_per_mm / result.mean / 1e9
        # verify the correctness
        tvm.testing.assert_allclose(y.asnumpy(), np.dot(a_, b_.T), rtol=0)
        print('Tensorization: running time: {:.3f} ms, {:.2f} Gops/s'.format(
            result.mean * 1000, gops_per_sec))

    verify()


if __name__ == "__main__":
    test_avx2_int8_gemm_acc32()
    pass
