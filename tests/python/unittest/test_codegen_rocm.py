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
import tvm
import numpy as np


def test_rocm_cross_thread_reduction():
    if not tvm.rocm(0).exist or not tvm.module.enabled("rocm"):
        print("skip because rocm is not enabled..")
        return

    # based on the reduction tutorial
    n = tvm.var("n")
    m = tvm.var("m")
    A = tvm.placeholder((n, m), name='A')
    k = tvm.reduce_axis((0, m), "k")
    B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name="B")
    s = tvm.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[B].bind(xi, tvm.thread_axis("threadIdx.y"))
    tx = tvm.thread_axis("threadIdx.x")
    s[B].bind(s[B].op.reduce_axis[0], tx)
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    s[B].set_store_predicate(tx.var.equal(0))
    frocm = tvm.build(s, [A, B], "rocm")

    nn = 128
    ctx = tvm.rocm(0)
    a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), ctx)
    frocm(a, b)
    tvm.testing.assert_allclose(
      b.asnumpy(),  np.sum(a.asnumpy(), axis=1), rtol=1e-4)


if __name__ == "__main__":
    test_rocm_cross_thread_reduction()
