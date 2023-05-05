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
from tvm.contrib import cc
from tvm.script import tir as T


def main(target, out_dir):
    @T.prim_func
    def func(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(var_A, (n,), align=1)
        B = T.match_buffer(var_B, (n,), align=1)
        C = T.match_buffer(var_C, (n,), align=1)
        # with T.block("root"):
        for i in range(n):
            with T.block("C"):
                v_i = T.axis.spatial(n, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = A[v_i] + B[v_i]

    if target == "cuda":
        sch = tvm.tir.Schedule(func)
        i, j = sch.split(sch.get_loops("C")[0], [None, 64])
        sch.bind(i, "blockIdx.x")
        sch.bind(j, "threadIdx.x")
        func = sch.mod["main"]

    fadd = tvm.build(func, target=tvm.target.Target(target, host="llvm"), name="myadd")
    fadd.save(osp.join(out_dir, "test_add.o"))
    if target == "cuda":
        fadd.imported_modules[0].save(osp.join(out_dir, "test_add.ptx"))
    cc.create_shared(osp.join(out_dir, "test_add.so"), [osp.join(out_dir, "test_add.o")])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
