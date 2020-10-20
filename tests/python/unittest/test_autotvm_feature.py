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
"""Test feature extraction"""

import numpy as np

import tvm
from tvm import te
from tvm.autotvm import feature


def test_iter_feature_gemm():
    N = 128

    k = te.reduce_axis((0, N), "k")
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")
    C = te.compute(A.shape, lambda y, x: te.sum(A[y, k] * B[k, x], axis=k), name="C")

    s = te.create_schedule(C.op)

    feas = feature.get_itervar_feature(s, [A, B, C], take_log=False)

    expected = [
        {
            "_attr_": [128, 1, 128, 2097152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "A_0": [128, -1, 16384, 128, 0, 0],
            "B_0": [0, -1, 16384, 128, 0, 0],
            "C_0": [128, -1, 16384, 128, 0, 0],
            "C_1": [128, -1, 16384, 128, 0, 0],
        },
        {
            "_attr_": [128, 2, 16384, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "A_0": [0, -1, 128, 128, 0, 0],
            "B_0": [1, -1, 16384, 1, 0, 0],
            "C_0": [1, -1, 128, 128, 0, 0],
            "C_1": [1, -1, 128, 128, 0, 0],
        },
        {
            "_attr_": [128, 3, 2097152, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "A_0": [1, -1, 128, 1, 0, 0],
            "B_0": [128, -1, 128, 1, 0, 0],
            "C_1": [0, -1, 1, 128, 0, 0],
            "C_2": [0, -1, 1, 128, 0, 0],
        },
    ]

    for ans, row in zip(expected, feas):
        for pair in row:
            if pair[0] not in ans:
                continue
            assert ans[pair[0]] == pair[1:], "%s: %s vs %s" % (pair[0], ans[pair[0]], pair[1:])


def test_curve_feature_gemm():
    N = 128

    k = te.reduce_axis((0, N), "k")
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")
    C = te.compute(A.shape, lambda y, x: te.sum(A[y, k] * B[k, x], axis=k), name="C")

    s = te.create_schedule(C.op)

    feas = feature.get_buffer_curve_sample_flatten(s, [A, B, C], sample_n=30)
    # sample_n * #buffers * #curves * 2 numbers per curve
    assert len(feas) == 30 * 3 * 4 * 2


def test_feature_shape():
    """test the dimensions of flatten feature are the same"""

    N = 1024
    n_sample = 100

    def get_gemm_feature(target):
        k = te.reduce_axis((0, N), "k")
        A = te.placeholder((N, N), name="A")
        B = te.placeholder((N, N), name="B")
        C = te.compute(A.shape, lambda y, x: te.sum(A[y, k] * B[k, x], axis=k), name="C")

        s = te.create_schedule(C.op)

        y, x = s[C].op.axis
        axes = list(s[C].tile(y, x, 8, 8)) + [k]
        perm = np.random.permutation(5)
        axes = [axes[x] for x in perm]
        s[C].reorder(*axes)

        if "gpu" in target.keys:
            pick = []
            # filter out reduction axis
            for i in range(len(perm)):
                if perm[i] != 4:
                    pick.append(axes[i])
            s[C].bind(pick[0], te.thread_axis("blockIdx.x"))
            s[C].bind(pick[1], te.thread_axis("vthread"))
            s[C].bind(pick[2], te.thread_axis("threadIdx.y"))

        with target:
            feas = feature.get_itervar_feature(s, [A, B, C])
            feas = feature.flatten_itervar_feature(feas)
        return feas

    targets = [
        tvm.target.cuda(),
        tvm.target.mali(),
        tvm.target.arm_cpu(),
    ]

    for target in targets:
        dim = len(get_gemm_feature(target))
        for i in range(n_sample):
            assert dim == len(get_gemm_feature(target)), (
                "dimensions of feature do not match" " for different configurations"
            )


if __name__ == "__main__":
    test_iter_feature_gemm()
    test_curve_feature_gemm()
    test_feature_shape()
