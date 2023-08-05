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
from tvm.script import tir as T
import tvm.testing
import numpy as np
import pytest


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.testing.requires_cuda
@pytest.mark.parametrize("f_preproc", ["", "l2_cache_flush_cuda"])
def test_time_evalutor_with_preproc(f_preproc: str):
    mod = tvm.IRModule.from_expr(matmul.with_attr("global_symbol", "main"))
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("matmul")
    i, j, k = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target="cuda")
    dev = tvm.cuda(0)
    evaluator = f.time_evaluator(f.entry_name, dev, repeat=1000, number=1, f_preproc=f_preproc)

    a = tvm.nd.array(np.random.rand(128, 128).astype("float32"), device=dev)
    b = tvm.nd.array(np.random.rand(128, 128).astype("float32"), device=dev)
    c = tvm.nd.array(np.zeros((128, 128)).astype("float32"), device=dev)
    args = [a, b, c]
    print("Evaluator (f_preproc={}):\t{:.5f}ms".format(f_preproc, evaluator(*args).mean * 1000))


if __name__ == "__main__":
    test_time_evalutor_with_preproc("l2_cache_flush_cuda")
