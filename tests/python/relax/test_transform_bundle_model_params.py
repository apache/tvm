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

from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
import tvm.topi.testing


def test_basic():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            params: R.Tuple(R.Tensor([16], "float32"), R.Tensor([16], "float32")),
        ) -> R.Tensor([16], "float32"):
            expr = a
            b = params[0]
            expr = R.add(expr, b)
            c = params[1]
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_no_model_params():
    """If all parameters are inputs, model params should be an empty tuple

    This ensures that a caller does not need to check whether the
    model has compile-time inputs, and can instead provide the output
    of a lifted parameter transformation in all cases, even if that
    transformation returns an empty tuple.
    """

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 3})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
            params: R.Tuple(),
        ) -> R.Tensor([16], "float32"):
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
