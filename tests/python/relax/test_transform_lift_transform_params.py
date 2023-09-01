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
from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
import numpy as np
import tvm.topi.testing


def test_basic():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ) -> None:
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), "float32"),
            w1: R.Tensor((3, 16, 3, 3), "float32"),
            w2: R.Tensor((16, 16, 3, 3), "float32"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                w1_transformed = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW, w1, R.Tensor((16, 3, 3, 3), "float32")
                )
                conv1 = R.nn.conv2d(
                    x, w1_transformed, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), dtype="float32"),
            param0: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param1: R.Tensor((16, 3, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                param1 = param1
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    param1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                param0 = param0
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    param0,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(conv2)
            return conv2

        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
                lv2 = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    (lv1,),
                    out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
                )
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 3, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_tuple():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"), w1: R.Tensor((16, 16, 3, 3), "float32")
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                l0 = (w1,)
                l1 = (l0,)
                l2 = l1[0]
                l3 = l2[0]
                conv1 = R.nn.conv2d(x, l3, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                conv2 = R.nn.conv2d(
                    conv1, w1, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), dtype="float32"),
            param0: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param1: R.Tensor((16, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = param1
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((16, 16, 3, 3), dtype="float32") = param0
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(conv2)
            return conv2

        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32"))
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[0]
                lv1: R.Tensor((16, 16, 3, 3), dtype="float32") = params[0]
                l0: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32")) = (lv1,)
                l1: R.Tuple(R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32"))) = (l0,)
                l2: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32")) = l1[0]
                lv2: R.Tensor((16, 16, 3, 3), dtype="float32") = l2[0]
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_condition():
    """Test case that the conditional statement can't be lifted"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"),
            w1: R.Tensor((16, 16, 3, 3), "float32"),
            w2: R.Tensor((16, 16, 3, 3), "float32"),
            cond: R.Tensor((), "bool"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            if cond:
                w = w1
            else:
                w = w2
            with R.dataflow():
                conv1 = R.nn.conv2d(x, w, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                R.output(conv1)
            return conv1

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((), dtype="bool"),
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"),
            R.Tensor((16, 16, 3, 3), dtype="float32"),
            R.Tensor((), dtype="bool"),
        ):
            with R.dataflow():
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[0]
                lv1: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                lv2: R.Tensor((), dtype="bool") = params[2]
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((), dtype="bool"),
                ) = (lv, lv1, lv2)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, 16, 224, 224), "float32"),
            param0: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param1: R.Tensor((16, 16, 3, 3), dtype="float32"),
            param2: R.Tensor((), dtype="bool"),
        ) -> R.Tensor((1, 16, 224, 224), "float32"):
            R.func_attr({"num_input": 1})
            gv: R.Tensor((), dtype="bool") = param2
            if gv:
                gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = param0
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            else:
                gv2: R.Tensor((16, 16, 3, 3), dtype="float32") = param1
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = gv2
            with R.dataflow():
                conv1 = R.nn.conv2d(x, w, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                R.output(conv1)
            return conv1

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_multiple_functions():
    @tvm.script.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

        @R.function
        def func2(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((256, 128), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

        @R.function
        def func3(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func1(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = param0
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, lv, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 256), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 256), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

        @R.function
        def func2(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 128), dtype="float32"),
        ) -> R.Tensor((256, 128), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv1: R.Tensor((256, 128), dtype="float32") = param0
                y: R.Tensor((256, 128), dtype="float32") = R.matmul(x, lv1, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func2_transform_params(
            params: R.Tuple(R.Tensor((128, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((128, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 128), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

        @R.function
        def func3(
            x: R.Tensor((256, 256), dtype="float32"), w1: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                w1_t: R.Tensor((256, 256), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_t, out_dtype="void")
                R.output(y)
            return y

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_stop_lifting():
    @tvm.script.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                w1_t1 = R.builtin.stop_lift_params(w1_t)
                w1_add = R.add(w1_t1, R.const(1, "float32"))
                y = R.matmul(x, w1_add)
                R.output(y)
            return y

    @I.ir_module
    class Expected:
        @R.function
        def func1(
            x: R.Tensor((256, 256), dtype="float32"),
            param0: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = param0
                w1_add: R.Tensor((256, 256), dtype="float32") = R.add(lv, R.const(1, "float32"))
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_add, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = params[0]
                lv1: R.Tensor((256, 256), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                gv: R.Tuple(R.Tensor((256, 256), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_symbolic_var_1():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(shape: R.Shape(["n"])):
            R.func_attr({"num_input": 1})
            n = T.int64()
            with R.dataflow():
                zeros = R.zeros((n, n), "float32")
            return shape

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(params: R.Tuple) -> R.Tuple:
            with R.dataflow():
                gv: R.Tuple = R.tuple()
                R.output(gv)
            return gv

        @R.function
        def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
            R.func_attr({"num_input": 1})
            n = T.int64()
            with R.dataflow():
                zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
                R.output()
            return shape

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


def test_symbolic_var_2():
    @I.ir_module
    class Before:
        @T.prim_func
        def zeros(var_T_full: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            T_full = T.match_buffer(var_T_full, (n, n))
            for ax0, ax1 in T.grid(n, n):
                with T.block("T_full"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads()
                    T.writes(T_full[v_ax0, v_ax1])
                    T_full[v_ax0, v_ax1] = T.float32(0)

        @R.function
        def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
            R.func_attr({"num_input": 1})
            n = T.int64()
            cls = Before
            with R.dataflow():
                zeros = R.call_tir(
                    cls.zeros, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float32")
                )
                R.output()
            return shape

    @I.ir_module
    class Expected:
        @T.prim_func
        def zeros(var_T_full: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            T_full = T.match_buffer(var_T_full, (n, n))
            # with T.block("root"):
            for ax0, ax1 in T.grid(n, n):
                with T.block("T_full"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads()
                    T.writes(T_full[v_ax0, v_ax1])
                    T_full[v_ax0, v_ax1] = T.float32(0)

        @R.function
        def main_transform_params(params: R.Tuple) -> R.Tuple:
            with R.dataflow():
                gv: R.Tuple = R.tuple()
                R.output(gv)
            return gv

        @R.function
        def main(shape: R.Shape(["n"])) -> R.Shape(["n"]):
            R.func_attr({"num_input": 1})
            n = T.int64()
            cls = Expected
            with R.dataflow():
                zeros = R.call_tir(
                    cls.zeros, R.tuple(), out_sinfo=R.Tensor((n, n), dtype="float32")
                )
                R.output()
            return shape

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
