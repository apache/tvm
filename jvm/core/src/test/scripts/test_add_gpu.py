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
from tvm import te
from tvm.contrib import cc, nvcc, utils


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


def test_add(target_dir):
    if not tvm.runtime.enabled("cuda"):
        print("skip %s because cuda is not enabled..." % __file__)
        return
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    fadd_cuda = tvm.build(s, [A, B, C], tvm.target.Target("cuda", host="llvm"), name="myadd")

    fadd_cuda.save(os.path.join(target_dir, "add_cuda.o"))
    fadd_cuda.imported_modules[0].save(os.path.join(target_dir, "add_cuda.ptx"))
    cc.create_shared(
        os.path.join(target_dir, "add_cuda.so"), [os.path.join(target_dir, "add_cuda.o")]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
