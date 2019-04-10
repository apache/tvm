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
import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import util, emscripten

def test_local_save_load():
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype='int32')
    B = tvm.placeholder((n,), name='B', dtype='int32')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    s[C].opengl()

    f = tvm.build(s, [A, B, C], "opengl", target_host="llvm", name="myadd")

    ctx = tvm.opengl(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(high=10, size=(n)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(high=10, size=(n)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros((n), dtype=C.dtype), ctx)
    f(a, b, c)

    temp = util.tempdir()
    path_so = temp.relpath("myadd.so")
    f.export_library(path_so)
    f1 = tvm.module.load(path_so)
    f1(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

if __name__ == "__main__":
    test_local_save_load()
