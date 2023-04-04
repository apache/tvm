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
from tvm.script import tir as T, ir as I
import tvm.testing


def test_annotate_entry_func_single_primfunc():
    @tvm.script.ir_module
    class MockModule:
        @T.prim_func
        def func1(A: T.Buffer((16,), "float32")):
            for i in T.serial(16):
                if i == 5:
                    if i == 5:
                        A[i] = 0.0

    mod = MockModule
    assert mod
    assert mod["func1"].attrs is None
    after = tvm.tir.transform.AnnotateEntryFunc()(mod)
    assert (
        after["func1"].attrs
        and "tir.is_entry_func" in after["func1"].attrs
        and after["func1"].attrs["tir.is_entry_func"]
    )


# Test module
@tvm.script.ir_module
class MockModule:
    @T.prim_func
    def func1(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.prim_func
    def func2(A: T.Buffer((32,), "float32")):
        for i in T.serial(32):
            if i == 15:
                if i == 15:
                    A[i] = 0.0


@pytest.mark.xfail
def test_annotate_entry_func_multiple_primfunc():
    mod = MockModule
    assert mod
    assert mod["func1"].attrs is None
    assert mod["func2"].attrs is None
    # This should fail
    after = tvm.tir.transform.AnnotateEntryFunc()(mod)


def test_bind_target():
    mod = MockModule
    assert mod

    target = tvm.target.Target("cuda")
    assert mod["func1"].attrs is None
    assert mod["func2"].attrs is None
    after = tvm.tir.transform.BindTarget(target)(mod)

    assert after["func1"].attrs and "target" in after["func1"].attrs
    assert after["func1"].attrs["target"] == target
    assert after["func2"].attrs and "target" in after["func2"].attrs
    assert after["func2"].attrs["target"] == target


def test_filter_primfunc():
    mod = MockModule
    assert mod
    # Annotate each function for testing
    mod["func1"] = mod["func1"].with_attr("temp", "test1")
    mod["func2"] = mod["func2"].with_attr("temp", "test2")

    # Test condition that does not filter out anything
    def checker_filter_out_none(func: tvm.tir.PrimFunc):
        return (func.attrs is not None) and ("temp" in func.attrs)

    after = tvm.tir.transform.Filter(checker_filter_out_none)(mod)
    assert len(after.functions) == 2
    # Filtered functions should satisfy the given condition.
    assert checker_filter_out_none(after["func1"])
    assert checker_filter_out_none(after["func2"])

    # Test condition that selectively filters out primfuncs
    def checker_filter_out_one(func: tvm.tir.PrimFunc):
        return (func.attrs is not None) and ("temp" in func.attrs) and func.attrs["temp"] == "test1"

    after = tvm.tir.transform.Filter(checker_filter_out_one)(mod)
    assert len(after.functions) == 1
    # Filtered functions should satisfy the given condition.
    assert checker_filter_out_one(after["func1"])

    # Test condition that filters out everything
    def checker_filter_out_both(func: tvm.tir.PrimFunc):
        return (func.attrs is not None) and ("invalid_attr" in func.attrs)

    after = tvm.tir.transform.Filter(checker_filter_out_both)(mod)
    assert len(after.functions) == 0


class TestFilterRemovesGlobalVarMap(tvm.testing.CompareBeforeAfter):
    """Filtering out a function should be identical to never adding it

    This test is to guard against hidden state in the IRModule that
    remains after filtering.  Previously, this was observed in the
    `IRModuleNode::global_var_map_`, which retained entries of
    filtered-out functions.
    """

    transform = tvm.tir.transform.Filter(lambda prim_func: False)

    def before(self):
        @I.ir_module
        class module:
            @T.prim_func
            def func():
                T.evaluate(0)

        return module

    def expected(self):
        @I.ir_module
        class module:
            pass

        return module


if __name__ == "__main__":
    tvm.testing.main()
