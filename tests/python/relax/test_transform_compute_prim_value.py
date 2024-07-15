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
import tvm.testing
from tvm.script import ir as I, relax as R, tir as T


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.relax.transform.ComputePrimValue()


class TestPrimValueInAssertCondition(BaseCompare):
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A: R.Tensor(["N"])):
            N = T.int64()
            _ = R.assert_op(N % 16 == 0)
            return A

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A: R.Tensor(["N"])):
            N = T.int64()
            condition: R.Prim("bool") = Expected.compute_symbolic_expr(R.prim_value(N))
            _ = R.assert_op(condition)
            return A

        @T.prim_func(private=True)
        def compute_symbolic_expr(N: T.int64) -> T.bool:
            T.func_attr({"tir.is_host_func": True})
            T.ret(N % 16 == 0)


class TestPrimValueInBranchCondition(BaseCompare):
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A: R.Tensor(["N"])):
            N = T.int64()
            if R.prim_value(N % 16 == 0):
                out = R.call_packed("fast_vectorized_impl", A, sinfo_args=[A.struct_info])
            else:
                out = R.call_packed("slow_non_vectorized_impl", A, sinfo_args=[A.struct_info])
            return out

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A: R.Tensor(["N"])):
            N = T.int64()
            condition: R.Prim("bool") = Expected.compute_symbolic_expr(R.prim_value(N))
            if condition:
                out = R.call_packed("fast_vectorized_impl", A, sinfo_args=[A.struct_info])
            else:
                out = R.call_packed("slow_non_vectorized_impl", A, sinfo_args=[A.struct_info])
            return out

        @T.prim_func(private=True)
        def compute_symbolic_expr(N: T.int64) -> T.bool:
            T.func_attr({"tir.is_host_func": True})
            T.ret(N % 16 == 0)


class TestPrimValueInPureFunction(BaseCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(_N: R.Prim(value="N"), _M: R.Prim(value="M")) -> R.Prim(value="N*M"):
            N = T.int64()
            M = T.int64()
            out = R.prim_value(N * M)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(_N: R.Prim(value="N"), _M: R.Prim(value="M")) -> R.Prim(value="N*M"):
            N = T.int64()
            M = T.int64()
            out = Expected.compute_symbolic_expr(R.prim_value(N), R.prim_value(M))
            return out

        @T.prim_func(private=True)
        def compute_symbolic_expr(N: T.int64, M: T.int64) -> T.int64:
            T.func_attr({"tir.is_host_func": True})
            T.ret(N * M)


if __name__ == "__main__":
    tvm.testing.main()
