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
"""Test eliminate common subexpr pass"""
import tvm
import tvm.testing
from tvm.relax.transform import EliminateCommonSubexpr
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(EliminateCommonSubexpr()(input), expected)


def test_simple():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                lv1 = R.add(x, y)
                gv = R.multiply(lv0, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                gv = R.multiply(lv0, lv0)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_skip_callback():
    pass


def test_tuple_get_time():
    pass


def test_tuple_arg():
    pass


if __name__ == "__main__":
    tvm.testing.main()
