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
from tvm import relax
from tvm import tir
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R, ir as I


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_relax_function():
    @I.ir_module
    class Before:
        @R.function(private=True)
        def f():
            return R.const(1, "int32")

        @R.function
        def f1():
            R.func_attr({"global_symbol": "f"})
            cls = Before
            gv: R.Tensor((), dtype="int32") = cls.f()
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def f():
            cls = Expected
            gv: R.Tensor((), dtype="int32") = cls.f1()
            return gv

        @R.function(private=True)
        def f1():
            return R.const(1, "int32")

    After = relax.transform.NormalizeGlobalVar()(Before)

    assert not relax.analysis.well_formed(Before)
    assert relax.analysis.well_formed(After)
    assert_structural_equal(After, Expected)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_tir_function():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def f(x: T.Buffer((1,), "int32")):
            x[0] = T.int32(0)

        @R.function
        def f1():
            R.func_attr({"global_symbol": "f"})
            cls = Before
            gv: R.Tensor((), dtype="int32") = R.call_tir(cls.f, (), R.Tensor((1,), dtype="int32"))
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def f1(x: T.Buffer((1,), "int32")):
            x[0] = 0

        @R.function
        def f() -> R.Tensor((1,), dtype="int32"):
            cls = Expected
            gv = R.call_tir(cls.f1, R.tuple(), out_sinfo=R.Tensor((1,), dtype="int32"))
            return gv

    After = relax.transform.NormalizeGlobalVar()(Before)

    assert not relax.analysis.well_formed(Before)
    assert relax.analysis.well_formed(After)
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
