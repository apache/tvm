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
from tvm import te

import numpy as np

def test_if_then_else_intrin_to_stmt():
    n = m = 32
    A = te.compute((n, m), lambda i, j: tvm.te.if_then_else(i > 5,
        tvm.te.if_then_else(j < 3, 0, 1),
        tvm.te.if_then_else(j > 8, 2, 3)), name='A')
    dtype = A.dtype
    s = te.create_schedule(A.op)
    mod = tvm.lower(s, [A], name="f")
    assert(type(mod["f"].body.body.body) is tvm.tir.stmt.IfThenElse)
    assert(type(mod["f"].body.body.body.then_case) is tvm.tir.stmt.IfThenElse)
    assert(type(mod["f"].body.body.body.else_case) is tvm.tir.stmt.IfThenElse)
    assert("5" in str(mod["f"].body.body.body.condition))
    assert("3" in str(mod["f"].body.body.body.then_case.condition))
    assert("8" in str(mod["f"].body.body.body.else_case.condition))

    ctx = tvm.cpu(0)
    mod = tvm.build(mod, [A], target="llvm")
    A_np = np.array([
        [0 if j < 3 else 1 for j in range(m)] if i > 5 else
        [2 if j > 8 else 3 for j in range(m)] for i in range(n)
    ], dtype=dtype)
    A_nd = tvm.nd.array(np.zeros(A_np.shape, dtype=dtype), ctx)
    mod(A_nd)
    tvm.testing.assert_allclose(A_nd.asnumpy(), A_np)

if __name__ == "__main__":
    test_if_then_else_intrin_to_stmt()
