#!/usr/bin/env python3
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

import os.path as osp
import sys

import tvm
from tvm import te
from tvm.contrib import cc


def main(target, out_dir):
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)

    if target == "cuda":
        bx, tx = s[C].split(C.op.axis[0], factor=64)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

    fadd = tvm.build(s, [A, B, C], tvm.target.Target(target, host="llvm"), name="myadd")
    fadd.save(osp.join(out_dir, "test_add.o"))
    if target == "cuda":
        fadd.imported_modules[0].save(osp.join(out_dir, "test_add.ptx"))
    cc.create_shared(osp.join(out_dir, "test_add.so"), [osp.join(out_dir, "test_add.o")])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
