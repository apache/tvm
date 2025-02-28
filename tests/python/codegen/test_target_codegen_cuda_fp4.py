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
from tvm.script import tir as T
import numpy as np
import tvm.testing
from tvm.script import ir as I, relax as R, tir as T

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

native_dtype, promoted_dtype = tvm.testing.parameters(
    ("e2m1_float4x2", "float32x2"),
    ("e2m1_float4x2", "float16x2"),
)


@tvm.testing.requires_cuda_compute_version(9)
def test_e2m1_vector_conversions(native_dtype, promoted_dtype):
    vector_length = 64

    @T.prim_func
    def add(
        A: T.Buffer((vector_length,), native_dtype),
        B: T.Buffer((vector_length,), native_dtype),
        C: T.Buffer((vector_length,), native_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(vector_length):
            with T.block("C"):
                v_i = T.axis.spatial(vector_length, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(
                    native_dtype, T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i])
                )

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)
    cuda_src = fadd.imported_modules[0].get_source()
    dev = tvm.device(target, 0)

    numpytype = "float4_e2m1fn"
    if "x" in native_dtype:
        lanes = int(native_dtype.split("x")[-1])
    else:
        lanes = 1

    if "x" in promoted_dtype:
        promoted_base_dtype = promoted_dtype.split("x")[0]
    else:
        promoted_base_dtype = promoted_dtype

    np_shape = (vector_length, lanes) if lanes > 1 else (vector_length,)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    a = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    b = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
    )


if __name__ == "__main__":
    tvm.testing.main()
