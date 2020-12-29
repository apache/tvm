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

"""Prepares a simple TVM library for testing."""

from os import path as osp
import sys

import tvm
from tvm import te
from tvm.contrib import cc


def main():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.te.create_schedule(C.op)
    s[C].parallel(s[C].op.axis[0])
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    obj_file = osp.join(sys.argv[1], "test.o")
    tvm.build(s, [A, B, C], "llvm").save(obj_file)
    cc.create_shared(osp.join(sys.argv[1], "test.so"), [obj_file])


if __name__ == "__main__":
    main()
