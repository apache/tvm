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
import os

import tvm
from tvm.contrib import cc, util

def test_add(target_dir):
    if not tvm.runtime.enabled("cuda"):
        print("skip %s because cuda is not enabled..." % __file__)
        return
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

    s = tvm.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    fadd_cuda = tvm.build(s, [A, B, C], "cuda", target_host="llvm", name="myadd")

    fadd_cuda.save(os.path.join(target_dir, "add_gpu.o"))
    fadd_cuda.imported_modules[0].save(os.path.join(target_dir, "add_gpu.ptx"))
    cc.create_shared(os.path.join(target_dir, "add_gpu.so"),
            [os.path.join(target_dir, "add_gpu.o")])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
