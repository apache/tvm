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

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_annotate_entry_func_single_function():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            T.func_attr({"tirx.is_entry_func": True})
            T.evaluate(0)

    assert "tirx.is_entry_func" not in (Before["main"].attrs or {})
    After = tvm_tirx.tirx.transform.AnnotateEntryFunc()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_annotate_entry_func_multiple_function():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def func1():
            T.evaluate(0)

        @T.function(private=True)
        def func2():
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.AnnotateEntryFunc()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)


def test_bind_target_adds_attribute():
    """BindTarget adds the "target" attribute"""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda"))(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_host_to_exposed_function():
    """BindTarget adds the host target to externally-exposed functions"""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"global_symbol": "main"})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda", host="llvm"))(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


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
        @T.function(private=True)
        def main():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda", host="llvm"))(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_ignores_existing():
    """BindTarget should not replace existing annotations"""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": T.target("nvptx")})
            T.evaluate(0)

    Expected = Before

    After = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda"))(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_updates_host():
    """BindTarget should update host for existing annotations"""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"global_symbol": "func", "target": T.target("nvptx")})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr(
                {
                    "global_symbol": "func",
                    "target": T.target("nvptx", host={"kind": "llvm", "opt-level": 0}),
                }
            )
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.BindTarget(
        tvm_tirx.target.Target("cuda", host={"kind": "llvm", "opt-level": 0})
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_multiple_functions():
    """BindTarget may apply to multiple functions in a module"""

    @I.ir_module
    class Before:
        @T.function
        def func1():
            T.evaluate(0)

        @T.function
        def func2():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def func1():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

        @T.function
        def func2():
            T.func_attr({"target": T.target("cuda")})
            T.evaluate(0)

    assert all("target" not in (func.attrs or {}) for func in Before.functions.values())
    After = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda"))(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_device_host_call_same_func():
    """BindTarget should bind the device target to the function if it is called from device"""

    @I.ir_module
    class Before:
        @T.function(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            return a + b

        @T.function
        def main(
            A: T.Tensor((128, 128), "int32"),
            B: T.Tensor((128, 128), "int32"),
            C: T.Tensor((128, 128), "int32"),
        ):
            T.func_attr({"global_symbol": "main"})
            length: T.let[T.int32] = Before.add(64, 64)  # Call from host
            with T.launch_thread("blockIdx.x", length) as bx:
                with T.launch_thread("threadIdx.x", length) as tx:
                    C[bx, tx] = Before.add(A[bx, tx], B[bx, tx])  # Call from device

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target("cuda")})
            return a + b

        @T.function(private=True)
        def add_host(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target({"kind": "llvm", "opt-level": 0})})
            return a + b

        @T.function
        def main(
            A: T.Tensor((128, 128), "int32"),
            B: T.Tensor((128, 128), "int32"),
            C: T.Tensor((128, 128), "int32"),
        ):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "target": T.target("cuda", host={"kind": "llvm", "opt-level": 0}),
                }
            )
            length: T.let[T.int32] = Expected.add_host(64, 64)  # Call from host
            with T.launch_thread("blockIdx.x", length) as bx:
                with T.launch_thread("threadIdx.x", length) as tx:
                    C[bx, tx] = Expected.add(A[bx, tx], B[bx, tx])  # Call from device

    After = tvm_tirx.tirx.transform.BindTarget(
        tvm_tirx.target.Target("cuda", host={"kind": "llvm", "opt-level": 0})
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_bind_target_with_tirx_device_entry():
    """BindTarget classifies calls after T.device_entry as device-side."""

    @I.ir_module
    class Before:
        @T.function(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            return a + b

        @T.function
        def main(A: T.Tensor((1,), "int32")):
            T.func_attr({"global_symbol": "main"})
            host_value: T.let[T.int32] = Before.add(1, 2)
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(1,)))
            tx = T.cuda.thread_idx("x")
            A[tx] = Before.add(host_value, 3)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target({"arch": "sm_100a", "kind": "cuda"})})
            return a + b

        @T.function(private=True)
        def add_host(a: T.int32, b: T.int32) -> T.int32:
            T.func_attr({"target": T.target({"kind": "llvm", "opt-level": 0})})
            return a + b

        @T.function
        def main(A: T.Tensor((1,), "int32")):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "target": T.target(
                        {
                            "arch": "sm_100a",
                            "host": {"kind": "llvm", "opt-level": 0},
                            "kind": "cuda",
                        }
                    ),
                }
            )
            host_value: T.let[T.int32] = Expected.add_host(1, 2)
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(1,)))
            tx = T.cuda.thread_idx("x")
            A[tx] = Expected.add(host_value, 3)

    target = tvm_tirx.target.Target(
        {"kind": "cuda", "arch": "sm_100a"}, host={"kind": "llvm", "opt-level": 0}
    )
    After = tvm_tirx.tirx.transform.BindTarget(target)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_filter_function():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def func1():
            T.func_attr({"temp": "test1"})
            T.evaluate(0)

        @T.function(private=True)
        def func2():
            T.func_attr({"temp": "test2"})
            T.evaluate(0)

    @I.ir_module
    class One:
        @T.function(private=True)
        def func1():
            T.func_attr({"temp": "test1"})
            T.evaluate(0)

    @I.ir_module
    class Empty:
        pass

    for predicate, Expected in [
        (lambda func: "temp" in func.attrs, Before),
        (lambda func: func.attrs["temp"] == "test1", One),
        (lambda func: "invalid_attr" in func.attrs, Empty),
    ]:
        After = tvm_tirx.tirx.transform.Filter(predicate)(Before)
        tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_filter_removes_global_var_map():
    """Filtering out a function should be identical to never adding it

    This test is to guard against hidden state in the IRModule that
    remains after filtering.  Previously, this was observed in the
    `IRModuleNode::global_var_map_`, which retained entries of
    filtered-out functions.
    """

    @I.ir_module
    class Before:
        @T.function
        def func():
            T.evaluate(0)

    @I.ir_module
    class Expected:
        pass

    After = tvm_tirx.tirx.transform.Filter(lambda function: False)(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
