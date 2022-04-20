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
from tvm.contrib import cc, utils


def test_add(target_dir):
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], "llvm", name="myadd")

    fadd.save(os.path.join(target_dir, "add_cpu.o"))
    cc.create_shared(
        os.path.join(target_dir, "add_cpu.so"), [os.path.join(target_dir, "add_cpu.o")]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
