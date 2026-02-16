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
from tvm.script import tir as T, ir as I


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets(
    "cuda", "metal", {"kind": "vulkan", "supports_int64": True}, "opencl"
)
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_int_intrin(target, dev, dtype):
    test_funcs = [
        (T.clz, lambda x, dtype: int(dtype[-2:]) - (len(bin(x)) - 2)),
    ]

    for tvm_intrin, np_func in test_funcs:
        n = 128

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((n,), dtype),
                B: T.Buffer((n,), dtype),
            ):
                T.func_attr({"tir.noalias": True})
                for i0 in T.thread_binding(n, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i0 = T.axis.spatial(n, i0)
                        T.reads(A[v_i0])
                        T.writes(B[v_i0])
                        B[v_i0] = tvm_intrin(A[v_i0])

        f = tvm.compile(Module, target=target)
        a = tvm.runtime.tensor(np.random.randint(0, 100000, size=n).astype(dtype), dev)
        b = tvm.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
        f(a, b)
        ref = np.vectorize(partial(np_func, dtype=dtype))(a.numpy())
        tvm.testing.assert_allclose(b.numpy(), ref)


if __name__ == "__main__":
    tvm.testing.main()
