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
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
import numpy as np
import tvm.topi.testing


@pytest.mark.parametrize("consume_params", [True, False])
def test_basic(consume_params):
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
            w2: R.Tensor((16, 16, 3, 3), dtype="float32"),
            w1_transformed: R.Tensor((16, 3, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    w1_transformed,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    w2,
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
            R.func_attr({"num_input": 0})
            cls = Expected
            with R.dataflow():
                lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
                lv2 = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    (lv1,),
                    out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
                )
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 3, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class ExpectedConsumeParams:
        @R.function
        def main(
            x: R.Tensor((1, 3, 224, 224), dtype="float32"),
            w2: R.Tensor((16, 16, 3, 3), dtype="float32"),
            w1_transformed: R.Tensor((16, 3, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    x,
                    w1_transformed,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                conv2: R.Tensor((1, 16, 224, 224), dtype="float32") = R.nn.conv2d(
                    conv1,
                    w2,
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
            R.func_attr({"num_input": 0})
            cls = ExpectedConsumeParams
            with R.dataflow():
                lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
                _1: R.Tuple = R.call_pure_packed(
                    "vm.builtin.tuple_reset_item",
                    params,
                    R.prim_value(T.int32(0)),
                    sinfo_args=(R.Tuple,),
                )
                lv2 = R.call_tir(
                    cls.transform_layout_IOHW_to_OIHW,
                    (lv1,),
                    out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
                )
                lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
                _2: R.Tuple = R.call_pure_packed(
                    "vm.builtin.tuple_reset_item",
                    params,
                    R.prim_value(T.int32(1)),
                    sinfo_args=(R.Tuple,),
                )
                gv: R.Tuple(
                    R.Tensor((16, 16, 3, 3), dtype="float32"),
                    R.Tensor((16, 3, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

    mod = Before
    expected = Expected if not consume_params else ExpectedConsumeParams
    with tvm.transform.PassContext(
        config={"relax.lift_transform_params.consume_params": consume_params}
    ):
        after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, expected)

    names_after = [param.name_hint for param in after["main"].params]
    names_expected = [param.name_hint for param in expected["main"].params]
    assert names_after == names_expected


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

        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor((16, 16, 3, 3), dtype="float32"))
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                lv = params[0]
                lv0 = (lv,)
                lv1 = (lv0,)
                lv2 = params[0]
                lv3 = params[0]
                gv = (lv2, lv3)
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
            R.func_attr({"num_input": 0})
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
            if param2:
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = param0
            else:
                w: R.Tensor((16, 16, 3, 3), dtype="float32") = param1
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
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, param0, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            R.func_attr({"num_input": 0})
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
                y: R.Tensor((256, 128), dtype="float32") = R.matmul(x, param0, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func2_transform_params(
            params: R.Tuple(R.Tensor((128, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 128), dtype="float32")):
            R.func_attr({"num_input": 0})
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
                w1_add: R.Tensor((256, 256), dtype="float32") = R.add(param0, R.const(1, "float32"))
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_add, out_dtype="void")
                R.output(y)
            return y

        @R.function
        def func1_transform_params(
            params: R.Tuple(R.Tensor((256, 256), dtype="float32"))
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32")):
            R.func_attr({"num_input": 0})
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
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tuple = R.tuple()
                R.output()
            # All instance of the empty tuple are normalized to be
            # in-line.
            return R.tuple()

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
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tuple = R.tuple()
                R.output()
            return R.tuple()

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


def test_symbolic_var_from_shape():
    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([16, 16], "int32"),
            B: R.Tensor([16, 16], "int32"),
            shape: R.Shape(["slice_index"]),
        ) -> R.Tensor([16], "int32"):
            R.func_attr({"num_input": 1})
            slice_index = T.int64()
            cls = Before
            with R.dataflow():
                B_slice = R.call_tir(
                    cls.slice,
                    [B],
                    tir_vars=R.ShapeExpr([slice_index]),
                    out_sinfo=R.Tensor([16], dtype="int32"),
                )
                A_slice = R.call_tir(
                    cls.slice,
                    [A],
                    tir_vars=R.ShapeExpr([slice_index]),
                    out_sinfo=R.Tensor([16], dtype="int32"),
                )
                A_scale = R.multiply(A_slice, B_slice)
                R.output(A_scale)
            return A_scale

        @T.prim_func(private=True)
        def slice(
            Input_2d: T.Buffer(shape=[16, 16], dtype="int32"),
            Output_Slice: T.Buffer(shape=[16], dtype="int32"),
            slice_index: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for j in range(16):
                with T.block("T_full"):
                    vj = T.axis.remap("S", [j])
                    Output_Slice[vj] = Input_2d[slice_index, vj]

    @I.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([16, 16], "int32"),
            shape: R.Shape(["slice_index"]),
            B_slice: R.Tensor([16], "int32"),
        ) -> R.Tensor([16], "int32"):
            R.func_attr({"num_input": 1})
            slice_index = T.int64()
            cls = Expected
            with R.dataflow():
                A_slice = R.call_tir(
                    cls.slice,
                    [A],
                    tir_vars=R.ShapeExpr([slice_index]),
                    out_sinfo=R.Tensor([16], dtype="int32"),
                )
                A_scale = R.multiply(A_slice, B_slice)
                R.output(A_scale)
            return A_scale

        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor([16, 16], "int32"), R.Shape(["slice_index"]))
        ):
            R.func_attr({"num_input": 0})
            slice_index = T.int64()
            cls = Expected
            with R.dataflow():
                B = params[0]
                # extra_symbolic_vars = params[1]
                B_slice = R.call_tir(
                    cls.slice,
                    [B],
                    tir_vars=R.ShapeExpr([slice_index]),
                    out_sinfo=R.Tensor([16], dtype="int32"),
                )
                output = (R.ShapeExpr([slice_index]), B_slice)
                R.output(output)
            return output

        @T.prim_func(private=True)
        def slice(
            Input_2d: T.Buffer(shape=[16, 16], dtype="int32"),
            Output_Slice: T.Buffer(shape=[16], dtype="int32"),
            slice_index: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for j in range(16):
                with T.block("T_full"):
                    vj = T.axis.remap("S", [j])
                    Output_Slice[vj] = Input_2d[slice_index, vj]

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(Expected, after)


def test_symbolic_var_in_param_shape():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 16, 224, "n"), "float32"),
            w1: R.Tensor((16, "m", 3, 3), "float32"),
            w2: R.Tensor((16, "m", 3, 3), "float32"),
        ) -> R.Tensor((1, 16, 224, "n"), "float32"):
            m = T.int64()
            n = T.int64()
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros = R.zeros((n, n), "float32")
                w1 = R.add(w1, R.const(1, "float32"))
                conv1 = R.nn.conv2d(x, w1, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW")
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=(1, 1), data_layout="NCHW", kernel_layout="OIHW"
                )
                R.output(conv2)
            return conv2

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, "m", 3, 3), dtype="float32"),
                R.Tensor((16, "m", 3, 3), dtype="float32"),
            )
        ) -> R.Tuple(
            R.Tensor((16, "m", 3, 3), dtype="float32"), R.Tensor((16, "m", 3, 3), dtype="float32")
        ):
            R.func_attr({"num_input": 0})
            m = T.int64()
            with R.dataflow():
                lv1: R.Tensor((16, m, 3, 3), dtype="float32") = params[0]
                lv2: R.Tensor((16, m, 3, 3), dtype="float32") = R.add(lv1, R.const(1, "float32"))
                lv: R.Tensor((16, m, 3, 3), dtype="float32") = params[1]
                gv: R.Tuple(
                    R.Tensor((16, m, 3, 3), dtype="float32"),
                    R.Tensor((16, m, 3, 3), dtype="float32"),
                ) = (lv, lv2)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, 16, 224, "n"), dtype="float32"),
            transformed_param_0: R.Tensor((16, "m", 3, 3), dtype="float32"),
            transformed_param_1: R.Tensor((16, "m", 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 16, 224, "n"), dtype="float32"):
            n = T.int64()
            m = T.int64()
            R.func_attr({"num_input": 1})
            with R.dataflow():
                zeros: R.Tensor((n, n), dtype="float32") = R.zeros(R.shape([n, n]), dtype="float32")
                conv1: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(
                    x,
                    transformed_param_1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                conv2: R.Tensor((1, 16, 224, n), dtype="float32") = R.nn.conv2d(
                    conv1,
                    transformed_param_0,
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

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


# not supported yet
@pytest.mark.xfail
def test_symbolic_var_defined_in_params_but_used_in_weights():
    """A symbolic variable's occurrence in the weights may not define it

    In order to be a source of definition, a symbolic variable in the
    parameters must occur as a distinct parameter, as a tensor shape
    `R.Tensor(["var"])`, an explicit `R.Shape(["var"])`, or as a
    `R.Prim(value="var")`.  A variable that is part of a larger
    expression, such as `R.Tensor(["m * n"])`, are variable usages,
    not variable definitions.
    """

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["m", "n"], "float32"),
            weight: R.Tensor(["m * n"], "float32"),
        ) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            R.func_attr({"num_input": 1})
            with R.dataflow():
                weight = R.add(weight, R.const(1, "float32"))
                weight = R.reshape(weight, [m, n])
                output = R.multiply(x, weight)
                R.output(output)
            return output

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor(("k",), dtype="float32"))
        ) -> R.Tuple(R.Tensor(dtype="float32", ndim=1)):
            R.func_attr({"num_input": 0})
            k = T.int64()
            with R.dataflow():
                lv: R.Tensor((k,), dtype="float32") = params[0]
                gv: R.Tuple(R.Tensor((k,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor(("m", "n"), dtype="float32"),
            transformed_param_0: R.Tensor(dtype="float32", ndim=1),
        ) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.int64()
            n = T.int64()
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor(dtype="float32", ndim=1) = transformed_param_0
                weight: R.Tensor(dtype="float32", ndim=1) = R.add(lv, R.const(1, "float32"))
                weight_1: R.Tensor((m, n), dtype="float32") = R.reshape(weight, R.shape([m, n]))
                output: R.Tensor((m, n), dtype="float32") = R.multiply(x, weight_1)
                R.output(output)
            return output

    After = relax.transform.LiftTransformParams()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_only_lift_when_variable_uses_constants():
    """A variable that has no inputs should not be lifted

    For example, `R.zeros`, or the result of allocation function
    calls.
    """

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([16], "int32"),
            B: R.Tensor([16], "int32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                offset = R.ones([16], "int32")
                A_offset = R.add(A, offset)
                B_offset = R.add(B, offset)
                output = R.multiply(A_offset, B_offset)
                R.output(output)
            return output

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([16], "int32"),
            B_offset: R.Tensor([16], "int32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                offset = R.ones([16], "int32")
                A_offset = R.add(A, offset)
                output = R.multiply(A_offset, B_offset)
                R.output(output)
            return output

        @R.function
        def main_transform_params(params: R.Tuple([R.Tensor([16], "int32")])):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                offset = R.ones([16], "int32")
                B = params[0]
                B_offset = R.add(B, offset)
                output = (B_offset,)
                R.output(output)
            return output

    mod = Before
    after = relax.transform.LiftTransformParams()(mod)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
