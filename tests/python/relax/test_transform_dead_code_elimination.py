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
from tvm.relax.transform import DeadCodeElimination
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(DeadCodeElimination()(input), expected)


def test_simple():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv2, axes=[0, 3, 1, 2]
                )
                gv22: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(gv21, bias)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_2block():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv2, axes=[0, 3, 1, 2]
                )
                gv22: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(gv21, bias)
                R.output(gv2)
            gv3 = R.astype(gv2, dtype="float16")
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv2)
            gv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.astype(gv2, dtype="float16")
            return gv3

    verify(Input, Expected)


def check_if_func_exists(mod, func_name):
    gvs = [gv.name_hint for gv in mod.get_global_vars()]
    return func_name in gvs


def test_unused_relax_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_add(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.call_tir(InputModule.tir_add, (x, w), R.Tensor((16, 16), dtype="float32"))
            return gv0

    mod = InputModule
    assert mod
    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


provide_entry_func_name = tvm.testing.parameter(True, False)


def test_unused_relax_func_custom_entry_func(provide_entry_func_name):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func(private=True)
        def tir_add(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def foo(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.call_tir(InputModule.tir_add, (x, w), R.Tensor((16, 16), dtype="float32"))
            return gv0

    mod = InputModule
    assert mod

    if provide_entry_func_name:
        entry_functions = ["foo"]
    else:
        entry_functions = None

    # Test entry function other than "main".
    new_mod = DeadCodeElimination(entry_functions=entry_functions)(mod)
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_tracking_through_externally_exposed_func(provide_entry_func_name):
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func(private=True)
        def tir_add(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def foo(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.call_tir(InputModule.tir_add, (x, w), R.Tensor((16, 16), dtype="float32"))
            return gv0

        @R.function
        def main(x: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
            return x

    mod = InputModule
    assert mod

    # Test tracking of usage through externally-exposed function
    new_mod = DeadCodeElimination(entry_functions=["main"])(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "foo")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_relax_func_symbolic_shape():
    # Test with relax function w/ symbolic shape.
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_add(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func(x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")):
            m, k = T.int64(), T.int64()
            gv0 = R.call_tir(InputModule.tir_add, (x, w), R.Tensor((m + 1, k), dtype="float32"))
            return gv0

    mod = InputModule
    assert mod

    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "tir_add")
    assert not check_if_func_exists(new_mod, "unused_func")


def test_unused_prim_func():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function
        def relax_add(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = InputModule.relax_add(x, w)
            return gv0

    mod = InputModule
    assert mod
    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    assert check_if_func_exists(new_mod, "relax_add")
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external linkage.
    assert check_if_func_exists(new_mod, "unused_func")


def test_multiple_unused_funcs():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def unused_func1(
            x: T.Buffer((16, 16), "float32"),
            y: T.Buffer((16, 16), "float32"),
            z: T.Buffer((16, 16), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_unused"})
            for i, j in T.grid(16, 16):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    z[vi, vj] = x[vi, vj] + y[vi, vj]

        @R.function(private=True)
        def unused_func2(x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")):
            gv0 = R.add(x, w)
            return gv0

        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"), w: R.Tensor((16, 16), "float32")
        ) -> R.Tensor((16, 16), "float32"):
            gv0 = R.add(x, w)
            return gv0

    mod = InputModule
    assert mod

    new_mod = DeadCodeElimination()(mod)
    assert check_if_func_exists(new_mod, "main")
    # RemoveUnusedFunction pass won't remove the function with global symbol for the external linkage.
    assert check_if_func_exists(new_mod, "unused_func1")
    assert not check_if_func_exists(new_mod, "unused_func2")


def test_unused_dfb():
    # test if an unused dataflow block can be removed.
    @tvm.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv0, lv1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC"
                )
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(lv2)
            gv3 = R.astype(lv2, dtype="float16")
            # dead block
            with R.dataflow():
                lv4: R.Tensor((2, 4, 26, 26), dtype="float16") = R.permute_dims(
                    gv3, axes=[0, 3, 1, 2]
                )
                R.output(lv4)
            return gv3

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv0, lv1, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC"
                )
                R.output(lv2)
            gv3 = R.astype(lv2, dtype="float16")
            return gv3

    verify(Input, Expected)


def test_unused_dfb2():
    # test if an unused dataflow block can be removed.
    @tvm.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            # dead block
            with R.dataflow():
                lv0: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x, axes=[0, 2, 3, 1]
                )
                R.output(lv0)

            gv_x = R.astype(x, dtype="float16")
            gv_w = R.astype(x, dtype="float16")

            with R.dataflow():
                lv1: R.Tensor((2, 28, 28, 3), dtype="float16") = R.permute_dims(
                    gv_x, axes=[0, 2, 3, 1]
                )
                lv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.permute_dims(
                    gv_w, axes=[0, 2, 3, 1]
                )
                lv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.nn.conv2d(
                    lv1, lv2, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC"
                )
                # dead instruction -> usee lv1 also dead.
                lv4: R.Tensor((2, 3, 28, 28), dtype="float32") = R.permute_dims(
                    lv0, axes=[0, 3, 1, 2]
                )
                R.output(lv3)
            return lv3

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        ):
            gv_x = R.astype(x, dtype="float16")
            gv_w = R.astype(x, dtype="float16")

            with R.dataflow():
                lv1: R.Tensor((2, 28, 28, 3), dtype="float16") = R.permute_dims(
                    gv_x, axes=[0, 2, 3, 1]
                )
                lv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.permute_dims(
                    gv_w, axes=[0, 2, 3, 1]
                )
                lv3: R.Tensor((2, 26, 26, 4), dtype="float16") = R.nn.conv2d(
                    lv1, lv2, data_layout="NHWC", kernel_layout="OHWI", out_layout="NHWC"
                )
                R.output(lv3)
            return lv3

    verify(Input, Expected)


def test_extern_func():
    """DeadCodeElimination should retain the ExternFunc in the IRModule."""

    builder = tvm.relax.BlockBuilder()
    builder.add_func(tvm.relax.extern("extern_func"), "extern_func")
    before = builder.get()

    verify(before, before)


if __name__ == "__main__":
    tvm.testing.main()
