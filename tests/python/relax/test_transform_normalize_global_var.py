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
import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R, ir as I


def test_normalize_relax_function():
    # parser will check well-formedness so we can't use it to construct this example
    bb = relax.BlockBuilder()
    f = relax.Function(
        [],
        relax.SeqExpr([], relax.Constant(tvm.nd.array(np.int32(1)), R.Tensor((), "int32"))),
        R.Tensor((), "int32"),
    )
    f_gv = bb.add_func(bb.normalize(f).without_attr("global_symbol"), "f")
    with bb.function("f1", []):
        gv = bb.emit(f_gv(), "gv")
        bb.emit_func_output(gv)
    Before = bb.get()
    Before.update_func(Before.get_global_var("f1"), Before["f1"].with_attr("global_symbol", "f"))

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
    # parser will check well-formedness so we can't use it to construct this example
    bb = relax.BlockBuilder()

    @T.prim_func(private=True)
    def f(x: T.Buffer((1,), "int32")):
        x[0] = T.int32(0)

    f_gv = bb.add_func(f, "f")
    with bb.function("f1", []):
        gv = bb.emit(R.call_tir(f_gv, (), R.Tensor((1,), dtype="int32")))
        bb.emit_func_output(gv)
    Before = bb.get()
    Before.update_func(Before.get_global_var("f1"), Before["f1"].with_attr("global_symbol", "f"))

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
