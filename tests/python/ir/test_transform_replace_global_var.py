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

import tvm.testing
from tvm.script import ir as I, relax as R, tir as T


def _get_before_module():
    @I.ir_module
    class Module:
        @R.function
        def relax_main(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Module.relax_subroutine(A)
            C = R.call_tir(Module.tir_main, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Module.tir_main(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Module.tir_subroutine(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    return Module


def test_no_op_if_no_replacements():
    """If no replacements are performed, the IRModule is unmodified"""

    before = _get_before_module()
    expected = before

    after = before.replace_global_vars({})

    tvm.ir.assert_structural_equal(expected, after)
    assert before.same_as(after)


def test_replace_relax_main():
    """An externally-exposed Relax function may be replaced

    In this example, the "relax_main" function is renamed.  This
    requires changing both the GlobalVar used to refer to the
    function, and the "global_symbol" attribute of the
    externally-exposed function.

    """

    before = _get_before_module()
    after = before.replace_global_vars({"relax_main": "relax_main_with_new_name"})

    @I.ir_module
    class Expected:
        @R.function
        def relax_main_with_new_name(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Expected.relax_subroutine(A)
            C = R.call_tir(Expected.tir_main, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Expected.tir_main(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Expected.tir_subroutine(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    tvm.ir.assert_structural_equal(Expected, after)


def test_replace_relax_subroutine():
    """An internal Relax function may be replaced

    In this example, the "relax_subroutine" function is renamed.  This
    requires changing both the GlobalVar used to refer to the
    function, and the GlobalVar used to call the subroutine within
    "relax_main".  The "global_symbol" attribute does not need to be
    updated, because internal functions do not have this attribute.

    """

    before = _get_before_module()
    after = before.replace_global_vars({"relax_subroutine": "relax_subroutine_with_new_name"})

    @I.ir_module
    class Expected:
        @R.function
        def relax_main(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Expected.relax_subroutine_with_new_name(A)
            C = R.call_tir(Expected.tir_main, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Expected.tir_main(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine_with_new_name(
            A: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Expected.tir_subroutine(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    tvm.ir.assert_structural_equal(Expected, after)


def test_replace_tir_main():
    """An externally-exposed TIR function may be replaced

    In this example, the "tir_main" function is renamed.  This
    requires changing both the GlobalVar used to refer to the
    function, the "global_symbol" attribute of the externally-exposed
    function.  In addition, calls to the TIR function should be
    updated to use the new GlobalVar.

    """

    before = _get_before_module()
    after = before.replace_global_vars({"tir_main": "tir_main_with_new_name"})

    @I.ir_module
    class Expected:
        @R.function
        def relax_main(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Expected.relax_subroutine(A)
            C = R.call_tir(Expected.tir_main_with_new_name, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Expected.tir_main_with_new_name(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main_with_new_name(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Expected.tir_subroutine(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    tvm.ir.assert_structural_equal(Expected, after)


def test_replace_tir_subroutine():
    """An internally-exposed TIR function may be replaced

    In this example, the "tir_subroutine" function is renamed.  This
    requires changing both the GlobalVar used to refer to the
    function, and the GlobalVar used to refer to it.  Internal
    functions do not have the "global_symbol" attribute, so it does
    not need to be updated.

    """

    before = _get_before_module()
    after = before.replace_global_vars({"tir_subroutine": "tir_subroutine_with_new_name"})

    @I.ir_module
    class Expected:
        @R.function
        def relax_main(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Expected.relax_subroutine(A)
            C = R.call_tir(Expected.tir_main, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Expected.tir_main(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Expected.tir_subroutine_with_new_name(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine_with_new_name(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    tvm.ir.assert_structural_equal(Expected, after)


def test_simultaneous_replacements():
    """Multiple replacements may be performed simultaneously"""

    before = _get_before_module()
    after = before.replace_global_vars(
        {
            "relax_main": "relax_main_with_new_name",
            "relax_subroutine": "relax_subroutine_with_new_name",
            "tir_main": "tir_main_with_new_name",
            "tir_subroutine": "tir_subroutine_with_new_name",
        }
    )

    @I.ir_module
    class Expected:
        @R.function
        def relax_main_with_new_name(A: R.Tensor([16], "float32")) -> R.Tensor([16], "float32"):
            R.func_attr({"relax.force_pure": True})

            B = Expected.relax_subroutine_with_new_name(A)
            C = R.call_tir(Expected.tir_main_with_new_name, B, out_sinfo=R.Tensor([16], "float32"))

            D = R.builtin.alloc_tensor(R.shape([16]), "float32", runtime_device_index=0)
            Expected.tir_main_with_new_name(C, D)

            return D

        @R.function(private=True)
        def relax_subroutine_with_new_name(
            A: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            B = R.add(A, R.prim_value(T.float32(1.0)))
            return B

        @T.prim_func
        def tir_main_with_new_name(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
            Expected.tir_subroutine_with_new_name(A.data, B.data)

        @T.prim_func(private=True)
        def tir_subroutine_with_new_name(A_data: T.ptr("float32"), B_data: T.ptr("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

    tvm.ir.assert_structural_equal(Expected, after)


if __name__ == "__main__":
    tvm.testing.main()
