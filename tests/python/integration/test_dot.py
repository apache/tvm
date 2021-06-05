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
import tvm.testing
from tvm import te
import numpy as np


@tvm.testing.requires_llvm
def test_dot():
    nn = 12
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    k = te.reduce_axis((0, n), "k")
    C = te.compute((), lambda: te.sum(A[k] * B[k], axis=k), name="C")
    s = te.create_schedule(C.op)

    def verify(target):
        f = tvm.driver.build(s, [A, B, C], target)
        # verify
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(nn,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(nn,)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), np.dot(a.numpy(), b.numpy()), rtol=1e-4)

    verify("llvm")


if __name__ == "__main__":
    test_dot()
