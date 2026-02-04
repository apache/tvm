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
from tvm.script import tir as T, ir as I


def test_remove_assume():
    """Remove any instance of T.assume"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(1, "int32")):
            T.evaluate(T.assume(A[0] == 5))
            A[0] = 10

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(1, "int32")):
            A[0] = 10

    After = tvm.tir.transform.RemoveAssume()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_remove_assume_loop():
    """Loops containing only T.assume should be removed"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                T.evaluate(T.assume(A[i] == 0))

            for i in T.serial(16):
                A[i] = 10

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 10

    After = tvm.tir.transform.RemoveAssume()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
