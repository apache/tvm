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
"""codegen related to bool types"""

import tvm
import tvm.testing
from tvm import te
import numpy as np
import tvm.testing

arr_size = tvm.testing.parameter(32)


@tvm.testing.fixture
def compute(arr_size):
    A = te.placeholder((arr_size,), name="A")
    B = te.placeholder((arr_size,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) > B(*i), name="C")
    D = te.compute(C.shape, lambda *i: tvm.tir.all(C(*i), A(*i) > 1).astype("float32"), name="D")
    return [A, B, C, D]


@tvm.testing.fixture
def get_module(target, compute):
    target = tvm.target.Target(target)
    A, B, C, D = compute
    if target.kind.name == "llvm":
        return tvm.IRModule.from_expr(te.create_prim_func([A, B, D]))

    sch = tvm.tir.Schedule(te.create_prim_func([A, B, D]))
    for stage in ["C", "D"]:
        xo, xi = sch.split(sch.get_loops(stage)[0], factors=[None, 4])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "blockIdx.x")
    return sch.mod


@tvm.testing.uses_gpu
def test_cmp_load_store(target, dev, arr_size, compute, get_module):
    A, B, _, D = compute
    f = tvm.compile(get_module, target=target)

    a_np = np.random.uniform(size=arr_size).astype(A.dtype)
    b_np = np.random.uniform(size=arr_size).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    d = tvm.nd.array(np.zeros(arr_size, dtype=D.dtype), dev)
    f(a, b, d)
    np.testing.assert_equal(
        d.numpy(),
        np.logical_and(a_np > b_np, a_np > 1).astype("float32"),
    )


if __name__ == "__main__":
    tvm.testing.main()
