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
    transform = tvm.relax.transform.ExpandTupleArguments()


class TestSimple(BaseCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Before.func((A, B))

        @R.function(private=True)
        def func(args: R.Tuple([R.Tensor, R.Tensor])) -> R.Tensor:
            return args[0]

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Expected.func(A, B)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return A


class TestNested(BaseCompare):
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return Before.func(((A, B), (C, D)))

        @R.function(private=True)
        def func(
            args: R.Tuple(
                [
                    R.Tuple([R.Tensor, R.Tensor]),
                    R.Tuple([R.Tensor, R.Tensor]),
                ]
            )
        ) -> R.Tensor:
            return args[0][1]

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return Expected.func(A, B, C, D)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return B


if __name__ == "__main__":
    tvm.testing.main()
