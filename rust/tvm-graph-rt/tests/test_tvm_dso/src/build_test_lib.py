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
from tvm.contrib import cc
from tvm.script import tir as T


def main():
    @T.prim_func
    def func(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(var_A, (n,), align=1)
        B = T.match_buffer(var_B, (n,), align=1)
        C = T.match_buffer(var_C, (n,), align=1)
        for i in T.parallel(n):
            with T.block("C"):
                vi = T.axis.spatial(n, i)
                C[vi] = A[vi] + B[vi]

    print(tvm.lower(func, simple_mode=True))
    obj_file = osp.join(sys.argv[1], "test.o")
    tvm.build(func, "llvm").save(obj_file)
    cc.create_shared(osp.join(sys.argv[1], "test.so"), [obj_file])


if __name__ == "__main__":
    main()
