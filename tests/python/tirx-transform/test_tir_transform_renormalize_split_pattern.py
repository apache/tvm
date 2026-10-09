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
from tvm.script import tirx as T


def test_renormalize_split_pattern():
    @tvm.script.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
            for i in T.serial(24):
                for j in T.serial(32):
                    A[i * 32 + j] = T.if_then_else(
                        128 <= i * 32 + j and i * 32 + j < 640,
                        (i * 32 + j) % 128 // 32,
                        0,
                    )
                B[T.ramp(i * 4, 1, 4)] = T.ramp(i * 128, 1, 4) % 256 // 8

    @tvm.script.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
            for i in T.serial(24):
                for j in T.serial(32):
                    A[i * 32 + j] = T.if_then_else(
                        1 <= (i + j // 32) // 4 and (i + j // 32) // 20 < 1,
                        (i + j // 32) % 4,
                        0,
                    )
                B[T.ramp(i * 4, 1, 4)] = T.ramp(i * 128, 1, 4) // 8 % 32

    @tvm.script.ir_module
    class AfterSimplified:
        @T.function
        def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
            for i in T.serial(24):
                for j in T.serial(32):
                    A[i * 32 + j] = T.if_then_else(4 <= i and i < 20, i % 4, 0)
                B[T.ramp(i * 4, 1, 4)] = T.broadcast(i % 2 * 16, 4)

    after = tvm.tirx.transform.RenormalizeSplitPattern()(Before)
    tvm.ir.assert_structural_equal(after, After)
    after = tvm.tirx.transform.StmtSimplify()(after)
    tvm.ir.assert_structural_equal(after, AfterSimplified)


if __name__ == "__main__":
    tvm.testing.main()
