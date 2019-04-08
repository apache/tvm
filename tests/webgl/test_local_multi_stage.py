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

def test_local_multi_stage():
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype="int32")
    B = tvm.compute((n,), lambda i: A[i] + 1, name="B")
    C = tvm.compute((n,), lambda i: B[i] * 2, name="C")

    s = tvm.create_schedule(C.op)
    s[B].opengl()
    s[C].opengl()

    f = tvm.build(s, [A, C], "opengl", name="multi_stage")

    ctx = tvm.opengl(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx)
    f(a, c)

    tvm.testing.assert_allclose(c.asnumpy(), (a.asnumpy() + 1) * 2)

if __name__ == "__main__":
    test_local_multi_stage()
