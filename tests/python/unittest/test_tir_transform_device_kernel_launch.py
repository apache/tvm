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


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.LowerDeviceKernelLaunch()


class TestLowerDeviceKernelLaunch(BaseCompare):
    """Kernel launch parameters are added at the call site

    The "tir.kernel_launch_params" determines which parameters belong
    to the runtime, and which below to the device-side PrimFunc.
    Parameters that are required prior to launching a kernel (e.g. the
    number of Cuda threads to use) are stored in the
    `"tir.kernel_launch_params"` attribute, and are used by the
    runtime prior in order to launch the generated kernel.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                mod.kernel(A.data)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("cuda")})
                A = T.decl_buffer(1, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                T.call_packed("kernel", A.data)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "calling_conv": 2,
                        "tir.kernel_launch_params": [],
                        "global_symbol": "kernel",
                        "tir.is_global_func": True,
                    }
                )
                A = T.decl_buffer(1, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod


class TestExternallyVisibleKernelLaunch(BaseCompare):
    """Like TestLowerDeviceKernelLaunch, with pre-defined global_symbol

    Because the host and kernel will be handled by different code
    generators, the device-side kernel must be externally exposed for
    use by the host-side wrapper, even if the host-side wrapper does
    not directly expose the kernel.  Therefore, a "global_symbol"
    attribute must be added for the kernel if not already present.

    If the kernel already has a specific name, that name should be
    preserved.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                mod.kernel(A.data)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("cuda"), "global_symbol": "kernel_by_another_name"})
                A = T.decl_buffer(1, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                T.call_packed("kernel_by_another_name", A.data)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "calling_conv": 2,
                        "tir.kernel_launch_params": [],
                        "global_symbol": "kernel_by_another_name",
                        "tir.is_global_func": True,
                    }
                )
                A = T.decl_buffer(1, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod


class TestCollectLaunchParameter(BaseCompare):
    """Kernel launch parameters are added at the call site

    The "tir.kernel_launch_params" determines which parameters belong
    to the runtime, and which below to the device-side PrimFunc.
    Parameters that are required prior to launching a kernel (e.g. the
    number of Cuda threads to use) are stored in the
    `"tir.kernel_launch_params"` attribute, and are used by the
    runtime prior in order to launch the generated kernel.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(16, "float32")):
                T.func_attr({"target": T.target("llvm")})
                mod.kernel(A.data)

            @T.prim_func
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

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(16, "float32")):
                T.func_attr({"target": T.target("llvm")})
                T.call_packed("kernel", A.data, 16)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr(
                    {
                        "target": T.target("cuda"),
                        "calling_conv": 2,
                        "tir.kernel_launch_params": ["threadIdx.x"],
                        "global_symbol": "kernel",
                        "tir.is_global_func": True,
                    }
                )
                A = T.decl_buffer(16, dtype="float32", data=A_data)
                i = T.launch_thread("threadIdx.x", 16)
                A[i] = 0.0

        return mod


class TestSameDeviceDifferentTarget(BaseCompare):
    """Handle subroutine calls to same device, different codegen

    The device kernel launch is only required when the caller and
    callee are on different devices.  However, if the caller and
    callee use different codegen, then the call cannot be handled as
    an internal call by a single codegen.  Instead, it should be
    lowered to a `T.call_extern`.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                mod.kernel(A.data)

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr({"target": T.target("c")})
                A = T.decl_buffer(16, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(1, "float32")):
                T.func_attr({"target": T.target("llvm")})
                T.call_extern("kernel", A.data, dtype="void")

            @T.prim_func
            def kernel(A_data: T.handle("float32")):
                T.func_attr(
                    {
                        "target": T.target("c"),
                        "global_symbol": "kernel",
                        "tir.is_global_func": True,
                    }
                )
                A = T.decl_buffer(16, dtype="float32", data=A_data)
                A[0] = 0.0

        return mod


if __name__ == "__main__":
    tvm.testing.main()
