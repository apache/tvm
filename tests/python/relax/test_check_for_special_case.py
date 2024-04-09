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

import pytest

import tvm
import tvm.testing
from tvm.script import relax as R, tir as T

specialize_by_tir_var = tvm.testing.parameter(
    by_dict={
        "specialize-by-string": False,
        "specialize-by-tir-var": True,
    }
)


def test_check_for_static_value(specialize_by_tir_var):
    """Specialized implementations may be produced for static shapes

    The specialized variables may be given either as strings, or as
    TIR variables.
    """

    @R.function(private=True)
    def before(
        A: R.Tensor(("M", "K"), "float16"), B: R.Tensor(("K", "N"), "float16")
    ) -> R.Tensor(("M", "N"), "float16"):
        return R.matmul(A, B)

    @R.function(private=True)
    def expected(
        A: R.Tensor(("M", "K"), "float16"), B: R.Tensor(("K", "N"), "float16")
    ) -> R.Tensor(("M", "N"), "float16"):
        M = T.int64()
        K = T.int64()
        N = T.int64()

        @R.function(private=True)
        def general_case(
            A: R.Tensor(("M", "K"), "float16"), B: R.Tensor(("K", "N"), "float16")
        ) -> R.Tensor(("M", "N"), "float16"):
            return R.matmul(A, B)

        @R.function(private=True)
        def special_case(
            A: R.Tensor((128, 64), "float16"), B: R.Tensor((64, 32), "float16")
        ) -> R.Tensor((128, 32), "float16"):
            return R.matmul(A, B)

        if R.prim_value(M == 128 and K == 64 and N == 32):
            lambda_result = special_case(A, B)
            out: R.Tensor((M, N), "float16") = lambda_result
        else:
            lambda_result = general_case(A, B)
            out: R.Tensor((M, N), "float16") = lambda_result
        return out

    if specialize_by_tir_var:
        M, K = before.params[0].struct_info.shape
        _, N = before.params[1].struct_info.shape
        symbolic_var_map = {M: 128, K: 64, N: 32}
    else:
        symbolic_var_map = {"M": 128, "K": 64, "N": 32}

    after = before.check_for_special_case(symbolic_var_map)
    tvm.ir.assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm.testing.main()
