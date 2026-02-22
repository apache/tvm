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
from tvm.script import ir as I
from tvm.script import tir as T


def test_annotate_entry_func_single_primfunc():
    @tvm.script.ir_module
    class MockModule:
        @T.prim_func(private=True)
        def func1(A: T.Buffer((16,), "float32")):
            for i in T.serial(16):
                if i == 5:
                    if i == 5:
                        A[i] = 0.0

    mod = MockModule
    assert mod
    assert not mod["func1"].attrs
    after = tvm.tir.transform.AnnotateEntryFunc()(mod)
    assert (
        after["func1"].attrs
        and "tir.is_entry_func" in after["func1"].attrs
        and after["func1"].attrs["tir.is_entry_func"]
    )


# Test module
@tvm.script.ir_module
class MockModule:
    @T.prim_func(private=True)
    def func1(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.prim_func(private=True)
    def func2(A: T.Buffer((32,), "float32")):
        for i in T.serial(32):
            if i == 15:
                if i == 15:
                    A[i] = 0.0


@pytest.mark.xfail
def test_annotate_entry_func_multiple_primfunc():
    mod = MockModule
    assert mod
    assert not mod["func1"].attrs
    assert not mod["func2"].attrs
    # This should fail
    after = tvm.tir.transform.AnnotateEntryFunc()(mod)


def test_bind_target():
    mod = MockModule
    assert mod

    target = tvm.target.Target("cuda")
    assert not mod["func1"].attrs
    assert not mod["func2"].attrs
    after = tvm.tir.transform.BindTarget(target)(mod)

    assert "target" in after["func1"].attrs
    assert after["func1"].attrs["target"] == target
    assert "target" in after["func2"].attrs
    assert after["func2"].attrs["target"] == target


def test_bind_target_adds_attribute():
    """BindTarget adds the "target" attribute"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    After = tvm.tir.transform.BindTarget(tvm.target.Target("cuda"))(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_host_to_exposed_function():
    """BindTarget adds the host target to externally-exposed functions"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main"})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            T.evaluate(0)

    After = tvm.tir.transform.BindTarget(tvm.target.Target("cuda", host="llvm"))(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_host_to_internal_function():
    """Internal functions have a target annotation, but without the host

    The host portion of the target annotation provides host
    parameters, and is used to expose a function externally as part of
    `MakePackedAPI` and `MakeUnpackedAPI`.  For internal functions, no
    external exposure is required, so the host attribute should not be
    used.
    """

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def main():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def main():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    After = tvm.tir.transform.BindTarget(tvm.target.Target("cuda", host="llvm"))(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_ignores_existing():
    """BindTarget should not replace existing annotations"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            T.func_attr({"target": T.target("nvptx")})
            T.evaluate(0)

    Expected = Before

    After = tvm.tir.transform.BindTarget(tvm.target.Target("cuda"))(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_updates_host():
    """BindTarget should update host for existing annotations"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            T.func_attr({"global_symbol": "func", "target": T.target("nvptx")})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            T.func_attr(
                {
                    "global_symbol": "func",
                    "target": T.target("nvptx", host={"kind": "llvm", "opt-level": 0}),
                }
            )
            T.evaluate(0)

    After = tvm.tir.transform.BindTarget(
        tvm.target.Target("cuda", host={"kind": "llvm", "opt-level": 0})
    )(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_multiple_functions():
    """BindTarget may apply to multiple functions in a module"""

    @I.ir_module
    class Before:
        @T.prim_func
        def func1():
            T.evaluate(0)

        @T.prim_func
        def func2():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func
        def func1():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

        @T.prim_func
        def func2():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    After = tvm.tir.transform.BindTarget(tvm.target.Target("cuda"))(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_device_host_call_same_func():
    """BindTarget should bind the device target to the function if it is called from device"""

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            return a + b

        @T.prim_func
        def main(
            A: T.Buffer((128, 128), "int32"),
            B: T.Buffer((128, 128), "int32"),
            C: T.Buffer((128, 128), "int32"),
        ):
            T.func_attr({"global_symbol": "main"})
            length: T.int32 = Before.add(64, 64)  # Call from host
            for bx in T.thread_binding(length, "blockIdx.x"):
                for tx in T.thread_binding(length, "threadIdx.x"):
                    C[bx, tx] = Before.add(A[bx, tx], B[bx, tx])  # Call from device

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target("cuda")})
            return a + b

        @T.prim_func(private=True)
        def add_host(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target({"kind": "llvm", "opt-level": 0})})
            return a + b

        @T.prim_func
        def main(
            A: T.Buffer((128, 128), "int32"),
            B: T.Buffer((128, 128), "int32"),
            C: T.Buffer((128, 128), "int32"),
        ):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "target": T.target("cuda", host={"kind": "llvm", "opt-level": 0}),
                }
            )
            length: T.int32 = Expected.add_host(64, 64)  # Call from host
            for bx in T.thread_binding(length, "blockIdx.x"):
                for tx in T.thread_binding(length, "threadIdx.x"):
                    C[bx, tx] = Expected.add(A[bx, tx], B[bx, tx])  # Call from device

    After = tvm.tir.transform.BindTarget(
        tvm.target.Target("cuda", host={"kind": "llvm", "opt-level": 0})
    )(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_filter_primfunc():
    mod = MockModule
    assert mod
    # Annotate each function for testing
    mod["func1"] = mod["func1"].with_attr("temp", "test1")
    mod["func2"] = mod["func2"].with_attr("temp", "test2")

    # Test condition that does not filter out anything
    def checker_filter_out_none(func: tvm.tir.PrimFunc):
        return "temp" in func.attrs

    after = tvm.tir.transform.Filter(checker_filter_out_none)(mod)
    assert len(after.functions) == 2
    # Filtered functions should satisfy the given condition.
    assert checker_filter_out_none(after["func1"])
    assert checker_filter_out_none(after["func2"])

    # Test condition that selectively filters out primfuncs
    def checker_filter_out_one(func: tvm.tir.PrimFunc):
        return ("temp" in func.attrs) and func.attrs["temp"] == "test1"

    after = tvm.tir.transform.Filter(checker_filter_out_one)(mod)
    assert len(after.functions) == 1
    # Filtered functions should satisfy the given condition.
    assert checker_filter_out_one(after["func1"])

    # Test condition that filters out everything
    def checker_filter_out_both(func: tvm.tir.PrimFunc):
        return "invalid_attr" in func.attrs

    after = tvm.tir.transform.Filter(checker_filter_out_both)(mod)
    assert len(after.functions) == 0


def test_filter_removes_global_var_map():
    """Filtering out a function should be identical to never adding it

    This test is to guard against hidden state in the IRModule that
    remains after filtering.  Previously, this was observed in the
    `IRModuleNode::global_var_map_`, which retained entries of
    filtered-out functions.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def func():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        pass

    After = tvm.tir.transform.Filter(lambda prim_func: False)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
