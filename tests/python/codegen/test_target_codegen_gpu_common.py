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
from functools import partial

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal", "vulkan -supports_int64=1", "opencl")
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_int_intrin(target, dev, dtype):
    test_funcs = [
        (tvm.tir.clz, lambda x, dtype: int(dtype[-2:]) - (len(bin(x)) - 2)),
    ]

    def run_test(tvm_intrin, np_func, dtype):
        n = 128
        A = te.placeholder((n,), name="A", dtype=dtype)
        B = te.compute(A.shape, lambda *i: tvm_intrin(A(*i)), name="B")
        func = te.create_prim_func([A, B])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("B"))
        sch.bind(x, "threadIdx.x")
        f = tvm.build(sch.mod, target=target)
        a = tvm.nd.array(np.random.randint(0, 100000, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(B.dtype), dev)
        f(a, b)
        ref = np.vectorize(partial(np_func, dtype=dtype))(a.numpy())
        tvm.testing.assert_allclose(b.numpy(), ref)

    for func in test_funcs:
        run_test(*func, dtype)


if __name__ == "__main__":
    tvm.testing.main()
