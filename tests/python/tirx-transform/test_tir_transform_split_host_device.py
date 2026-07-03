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
from tvm.script import ir as I
from tvm.script import tirx as T


def test_public_api_surface():
    assert hasattr(tvm.tirx.transform, "SplitHostDevice")
    assert not hasattr(tvm.tirx.transform, "AnnotateDeviceRegions")
    assert not hasattr(tvm.tirx.transform, "LowerDeviceKernelLaunch")


def test_ssa_across_entire_module():
    """The host and device functions should not share TIR vars

    Any arguments that are passed from the host to the device should
    be in terms of independent TIR variables.
    """

    @I.ir_module
    class before:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host="llvm")})
            for i in range(16):
                T.attr(0, "device_scope", 0)
                for j in range(16):
                    T.evaluate(i)

    after = tvm.tirx.transform.SplitHostDevice()(before)
    loop_var = after["main"].body.loop_var
    param_var = after["main_kernel"].params[0]

    assert not loop_var.same_as(param_var)


def test_split_host_device():
    """SplitHostDevice divides a function at the "target" attribute"""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.attr(T.target("cuda"), "target", 0)
            T.evaluate(n)

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.call_packed("main_kernel", n)

        @T.prim_func(s_tir=True)
        def main_kernel(n: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "main_kernel",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            T.evaluate(n)

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_split_host_device_on_cpu():
    """A kernel running on the CPU may return an error code"""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.attr(T.target("llvm"), "target", 0)
            T.evaluate(n)

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            kernel_error_code: T.let[T.int32] = T.call_extern("int32", "main_kernel", n)
            assert kernel_error_code == 0, "Error executing compute kernel"

        @T.prim_func(s_tir=True)
        def main_kernel(n: T.int32) -> T.int32:
            T.func_attr(
                {
                    "target": T.target("llvm"),
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            T.evaluate(n)
            T.ret(0)

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_split_host_device_without_func_host_attribute():
    """Like test_split_host_device, but no host specified in the host's target

    The `T.attr` specifying the device still requires splitting out
    the kernel.
    """

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("llvm")})
            T.attr(T.target("cuda"), "target", 0)
            T.evaluate(n)

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("llvm")})
            T.call_packed("main_kernel", n)

        @T.prim_func(s_tir=True)
        def main_kernel(n: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "main_kernel",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            T.evaluate(n)

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_split_host_device_without_device_region():
    """Like test_split_host_device, but no device regions to extract

    Because MakePackedAPI/MakeUnpackedAPI still require both the
    device and host, SplitHostDevice does not modify the "target"
    attribute.
    """

    @T.prim_func(s_tir=True)
    def Before():
        T.func_attr({"target": T.target("ext_dev", host="llvm")})
        T.evaluate(0)

    Expected = Before

    After = tvm.tirx.transform.SplitHostDevice()(tvm.IRModule.from_expr(Before))
    tvm.ir.assert_structural_equal(After["Before"], Expected)


def test_split_host_device_name_collision():
    """Like test_split_host_device, but with the default name already taken

    The default name is generated as `func.name + "_kernel"`.  If this
    name is already taken by another function in the IRModule, then
    SplitHostDevice should select a different name.
    """

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.attr(T.target("cuda"), "target", 0)
            T.evaluate(n)

        @T.prim_func(s_tir=True)
        def main_kernel():
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr({"target": T.target("cuda", host={"kind": "llvm", "opt-level": 0})})
            T.call_packed("main_kernel_1", n)

        @T.prim_func(s_tir=True)
        def main_kernel_1(n: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "main_kernel_1",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            T.evaluate(n)

        @T.prim_func(s_tir=True)
        def main_kernel():
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(0)

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_dynamic_launch_thread():
    """Dynamic T.launch_thread may depend on host-side variable

    A dynamic parameter for `T.launch_thread` may have an extent that
    is computed using variables outside of the `T.target` section.

    This is a regression test to catch a previous failure mode, in
    which SplitHostDevice generated output with undefined variables,
    if the only use of a variable occurred in the extent of a
    `T.launch_thread` statement.

    While the launch-lowering stage will hoist the
    computation of the extent from the device kernel to the host
    function, the IRModule must be well-defined at all stages of
    lowering.  Even if a variable is only used as part of a thread
    extent, `SplitHostDevice` should treat it as a kernel parameter, to
    provide a definition of the variable within the TIR device kernel.
    """

    @I.ir_module
    class before:
        @T.prim_func(s_tir=True)
        def default_function(var_A: T.handle, var_B: T.handle, seq_len: T.int32):
            T.func_attr({"target": T.target("cuda")})

            A = T.match_buffer(var_A, [seq_len], "int32")
            B = T.match_buffer(var_B, [seq_len], "int32")

            num_blocks: T.let[T.int32] = (seq_len + 127) // 128
            with T.attr(T.target("cuda"), "target", 0):
                blockIdx_x = T.launch_thread("blockIdx.x", num_blocks)
                threadIdx_x = T.launch_thread("threadIdx.x", 128)
                if blockIdx_x * 128 + threadIdx_x < seq_len:
                    B[blockIdx_x * 128 + threadIdx_x] = A[blockIdx_x * 128 + threadIdx_x]

    @I.ir_module
    class expected:
        @T.prim_func(s_tir=True)
        def default_function(var_A: T.handle, var_B: T.handle, seq_len: T.int32):
            T.func_attr({"target": T.target("cuda")})
            A = T.match_buffer(var_A, (seq_len,), "int32")
            B = T.match_buffer(var_B, (seq_len,), "int32")
            num_blocks: T.let[T.int32] = (seq_len + 127) // 128
            expected.default_function_kernel(A.data, B.data, num_blocks, seq_len)

        @T.prim_func(private=True, s_tir=True)
        def default_function_kernel(
            A_data: T.handle("int32"),
            B_data: T.handle("int32"),
            num_blocks: T.int32,
            seq_len: T.int32,
        ):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "tirx.is_global_func": True,
                    "tirx.noalias": True,
                }
            )
            A = T.decl_buffer(seq_len, "int32", data=A_data)
            B = T.decl_buffer(seq_len, "int32", data=B_data)
            blockIdx_x = T.launch_thread("blockIdx.x", num_blocks)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            if blockIdx_x * 128 + threadIdx_x < seq_len:
                B[blockIdx_x * 128 + threadIdx_x] = A[blockIdx_x * 128 + threadIdx_x]

    after = tvm.tirx.transform.SplitHostDevice()(before)

    tvm.tirx.analysis.verify_well_formed(after)
    tvm.ir.assert_structural_equal(expected, after)


def test_symbolic_var_parameter():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(var_A: T.handle, var_B: T.handle):
            T.func_attr({"target": T.target("cuda")})
            m = T.int64()
            A = T.match_buffer(var_A, (m,))
            B = T.match_buffer(var_B, (m,))
            T.attr(T.target("cuda"), "target", 0)
            blockIdx_x = T.launch_thread("blockIdx.x", m)
            B_1 = T.decl_buffer((m,), data=B.data)
            A_1 = T.decl_buffer((m,), data=A.data)
            B_1[blockIdx_x] = A_1[blockIdx_x]

    after = tvm.tirx.transform.SplitHostDevice()(Module)
    assert len(after["main_kernel"].params) == 3
    assert isinstance(after["main_kernel"].params[2], tvm.tirx.Var)


def test_thread_extent_region_extracted_as_device_kernel():
    """A bare thread_extent is annotated and extracted as a device kernel."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            i = T.launch_thread("threadIdx.x", 16)
            A[i] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.call_packed("main_kernel", A.data, 16)

        @T.prim_func(s_tir=True)
        def main_kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": ["threadIdx.x"],
                    "global_symbol": "main_kernel",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            i = T.launch_thread("threadIdx.x", 16)
            A[i] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_device_scope_region_extracted_as_device_kernel():
    """A bare device_scope is annotated and extracted as a device kernel."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.attr(0, "device_scope", 0)
            A[0] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            T.call_packed("main_kernel", A.data)

        @T.prim_func(s_tir=True)
        def main_kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "main_kernel",
                    "tirx.noalias": True,
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            T.attr(0, "device_scope", 0)
            A[0] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_lower_device_kernel_launch():
    """Kernel calls are lowered using the public SplitHostDevice pass."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            Before.kernel(A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("cuda")})
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.call_packed("kernel", A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "kernel",
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_externally_visible_kernel_launch():
    """Kernel launch lowering preserves a pre-defined global_symbol."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            Before.kernel(A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("cuda"), "global_symbol": "kernel_by_another_name"})
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.call_packed("kernel_by_another_name", A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": [],
                    "global_symbol": "kernel_by_another_name",
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_collect_launch_parameter():
    """Thread launch extents are appended to the host launch call."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32")):
            T.func_attr({"target": T.target("llvm")})
            Before.kernel(A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "global_symbol": "kernel",
                }
            )
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            i = T.launch_thread("threadIdx.x", 16)
            A[i] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.call_packed("kernel", A.data, 16)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": ["threadIdx.x"],
                    "global_symbol": "kernel",
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            i = T.launch_thread("threadIdx.x", 16)
            A[i] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_same_device_different_target():
    """Same-device calls with different codegen are lowered to extern calls."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            Before.kernel(A.data)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("c")})
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            A[0] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.call_extern("kernel", A.data, dtype="void")

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32")):
            T.func_attr(
                {
                    "target": T.target("c"),
                    "global_symbol": "kernel",
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            A[0] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bind_before_thread_extent():
    """Bind-defined thread extents are inlined into launch arguments."""

    @I.ir_module
    class Before:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32"), n: T.int32):
            T.func_attr({"target": T.target("llvm")})
            Before.kernel(A.data, n)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32"), n: T.int32):
            T.func_attr({"target": T.target("cuda"), "global_symbol": "kernel"})
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            v: T.let[T.int32] = n + 1
            i = T.launch_thread("threadIdx.x", v)
            A[i] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(16, "float32"), n: T.int32):
            T.func_attr({"target": T.target("llvm")})
            T.call_packed("kernel", A.data, n, n + 1)

        @T.prim_func(s_tir=True)
        def kernel(A_data: T.handle("float32"), n: T.int32):
            T.func_attr(
                {
                    "target": T.target("cuda"),
                    "calling_conv": 2,
                    "tirx.kernel_launch_params": ["threadIdx.x"],
                    "global_symbol": "kernel",
                    "tirx.is_global_func": True,
                }
            )
            A = T.decl_buffer(16, dtype="float32", data=A_data)
            v: T.let[T.int32] = n + 1
            i = T.launch_thread("threadIdx.x", v)
            A[i] = 0.0

    After = tvm.tirx.transform.SplitHostDevice()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
