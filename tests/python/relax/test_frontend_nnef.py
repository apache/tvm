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

import numpy as np

import _nnef
import nnef

import tvm
import tvm.testing
from tvm import relax
import tvm.relax.frontend.nnef

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
import tvm.topi as topi

from ..nightly.frontend.nnef import cases_string


def get_case_graph(name):
    if "-" in name:
        name = name.replace("-", "_")
    return nnef.parse_string(getattr(cases_string, name))


def verify_model_struct(model_name, binding, expected):
    graph = get_case_graph(model_name)
    for operation in graph.operations:
        if operation.name == "variable":
            tensor_name = operation.outputs["output"]

            shape = operation.attribs["shape"]

            assert (
                operation.dtype == "scalar"
            ), f"variable of type {operation.dtype} is not supported, please update verify_model"

            data = np.ones(shape).astype("float32")

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(
                tensor.name, tensor.dtype, shape, data, tensor.quantization
            )

    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)

    mod = relax.frontend.nnef.from_nnef(graph)
    tvm.ir.assert_structural_equal(mod, expected)


def get_unary_mod(method, dt="float32", o_dtype=None):
    global dtype
    dtype = dt
    if not o_dtype:
        o_dtype = dtype

    def _appl_shape(sh, osh=None):
        global shape, o_shape
        if not osh:
            osh = sh
        shape, o_shape = sh, osh

        @tvm.script.ir.ir_module
        class expected:
            @R.function
            def main(in1: R.Tensor(shape, dtype)) -> R.Tensor(o_shape, o_dtype):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor(o_shape, dtype=o_dtype) = method(in1)
                    R.output(lv)
                return lv

        return expected

    return _appl_shape


def get_binary_mod(method, dt="float32", o_dtype=None):
    global dtype
    dtype = dt
    if not o_dtype:
        o_dtype = dtype

    def _appl_shape(sh1, sh2=None, osh=None):
        global shape1, shape2, o_shape
        if not sh2:
            sh2 = sh1
        if not osh:
            osh = sh1
        shape1, shape2, o_shape = sh1, sh2, osh

        @tvm.script.ir.ir_module
        class expected:
            @R.function
            def main(
                lhs: R.Tensor(shape1, dtype), rhs: R.Tensor(shape2, dtype)
            ) -> R.Tensor(o_shape, o_dtype):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor(o_shape, dtype=o_dtype) = method(lhs, rhs)
                    R.output(lv)
                return lv

        return expected

    return _appl_shape


# graph tests
def test_copy():
    @I.ir_module
    class expected_2d:
        @R.function
        def main(input: R.Tensor((4, 16), dtype="float32")) -> R.Tensor((4, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.identity, input)
                gv: R.Tensor((4, 16), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("copy_2d", {}, expected_2d)

    @I.ir_module
    class expected_4d:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.identity, input)
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("copy_4d", {}, expected_4d)


def test_neg():
    expected = get_unary_mod(R.negative)

    shape = (4, 16)
    verify_model_struct("neg_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("neg_4d", {}, expected(shape))


def test_rcp():
    def method(in1):
        return R.divide(R.const(1, "float32"), in1)

    expected = get_unary_mod(method)
    shape = (4, 16)
    verify_model_struct("rcp_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("rcp_4d", {}, expected(shape))


def test_exp():
    expected = get_unary_mod(R.exp)
    shape1 = (4, 16)
    verify_model_struct("exp_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("exp_4d", {}, expected(shape1))


def test_log():
    expected = get_unary_mod(R.log)

    shape = (4, 16)
    verify_model_struct("log_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("log_4d", {}, expected(shape))


def test_sin():
    expected = get_unary_mod(R.sin)

    shape = (4, 16)
    verify_model_struct("sin_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("sin_4d", {}, expected(shape))


def test_cos():
    expected = get_unary_mod(R.cos)

    shape = (4, 16)
    verify_model_struct("cos_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("cos_4d", {}, expected(shape))


def test_tan():
    expected = get_unary_mod(R.tan)

    # 2D
    shape = (4, 16)
    verify_model_struct("tan_2d", {}, expected(shape))

    # 4D
    shape = (4, 16, 32, 32)
    verify_model_struct("tan_4d", {}, expected(shape))


def test_sinh():
    expected = get_unary_mod(R.sinh)

    shape = (4, 16)
    verify_model_struct("sinh_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("sinh_4d", {}, expected(shape))


def test_cosh():
    expected = get_unary_mod(R.cosh)

    shape = (4, 16)
    verify_model_struct("cosh_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("cosh_4d", {}, expected(shape))


def test_tanh():
    expected = get_unary_mod(R.tanh)
    shape = (4, 16)
    verify_model_struct("tanh_2d_standalone", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("tanh_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((4, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.tanh(lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("tanh", binding, expected)


def test_asin():
    expected = get_unary_mod(R.asin)

    shape = (4, 16)
    verify_model_struct("asin_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("asin_4d", {}, expected(shape))


def test_acos():
    expected = get_unary_mod(R.acos)

    shape = (4, 16)
    verify_model_struct("acos_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("acos_4d", {}, expected(shape))


def test_atan():
    expected = get_unary_mod(R.atan)

    shape = (4, 16)
    verify_model_struct("atan_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("atan_4d", {}, expected(shape))


def test_asinh():
    expected = get_unary_mod(R.asinh)

    shape = (4, 16)
    verify_model_struct("asinh_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("asinh_4d", {}, expected(shape))


def test_acosh():
    expected = get_unary_mod(R.acosh)

    shape = (4, 16)
    verify_model_struct("acosh_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("acosh_4d", {}, expected(shape))


def test_atanh():
    expected = get_unary_mod(R.atanh)

    shape = (4, 16)
    verify_model_struct("atanh_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("atanh_4d", {}, expected(shape))


def test_abs():
    expected = get_unary_mod(R.abs)
    shape = (4, 16)
    verify_model_struct("abs_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("abs_4d", {}, expected(shape))


def test_sign():
    expected = get_unary_mod(R.sign)
    shape1 = (4, 16)
    verify_model_struct("sign_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("sign_4d", {}, expected(shape1))


def test_not():
    expected = get_unary_mod(R.logical_not, dt="bool")
    shape1 = (4, 16)
    verify_model_struct("not_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("not_4d", {}, expected(shape1))


def test_floor():
    expected = get_unary_mod(R.floor)

    shape = (4, 16)
    verify_model_struct("floor_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("floor_4d", {}, expected(shape))


def test_ceil():
    expected = get_unary_mod(R.ceil)

    shape = (4, 16)
    verify_model_struct("ceil_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("ceil_4d", {}, expected(shape))


def test_round():
    expected = get_unary_mod(R.round)

    shape = (4, 16)
    verify_model_struct("round_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("round_4d", {}, expected(shape))


def test_add():
    expected = get_binary_mod(R.add)
    shape1 = (4, 16)
    verify_model_struct("add_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("add_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("add_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.add(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("add_4d_constant", {}, expected(shape1))


def test_sub():
    expected = get_binary_mod(R.subtract)
    shape1 = (4, 16)
    verify_model_struct("sub_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("sub_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("sub_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.subtract(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("sub_4d_constant", {}, expected(shape1))


def test_mul():
    expected = get_binary_mod(R.multiply)
    shape1 = (4, 16)
    verify_model_struct("mul_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("mul_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("mul_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.multiply(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("mul_4d_constant", {}, expected(shape1))


def test_div():
    expected = get_binary_mod(R.divide)
    shape1 = (4, 16)
    verify_model_struct("div_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("div_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("div_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.divide(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("div_4d_constant", {}, expected(shape1))


def test_pow():
    expected = get_binary_mod(R.power)
    shape1 = (4, 16)
    verify_model_struct("pow_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("pow_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("pow_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.power(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("pow_4d_constant", {}, expected(shape1))


def test_lt():
    expected = get_binary_mod(R.less, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("lt_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("lt_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("lt_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.less(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("lt_4d_constant", {}, expected(shape1))


def test_gt():
    expected = get_binary_mod(R.greater, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("gt_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("gt_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("gt_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.greater(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("gt_4d_constant", {}, expected(shape1))


def test_le():
    expected = get_binary_mod(R.less_equal, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("le_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("le_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("le_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.less_equal(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("le_4d_constant", {}, expected(shape1))


def test_ge():
    expected = get_binary_mod(R.greater_equal, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("ge_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("ge_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("ge_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.greater_equal(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("ge_4d_constant", {}, expected(shape1))


def test_eq():
    expected = get_binary_mod(R.equal, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("eq_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("eq_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("eq_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.equal(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("eq_4d_constant", {}, expected(shape1))


def test_ne():
    expected = get_binary_mod(R.not_equal, o_dtype="bool")
    shape1 = (4, 16)
    verify_model_struct("ne_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("ne_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("ne_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.not_equal(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method, o_dtype="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("ne_4d_constant", {}, expected(shape1))


def test_and():
    expected = get_binary_mod(R.logical_and, dt="bool")
    shape1 = (4, 16)
    verify_model_struct("and_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("and_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("and_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.logical_and(in1, R.const(False, "bool"))

    expected = get_unary_mod(method, dt="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("and_4d_constant", {}, expected(shape1))


def test_or():
    expected = get_binary_mod(R.logical_or, dt="bool")
    shape1 = (4, 16)
    verify_model_struct("or_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("or_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("or_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.logical_or(in1, R.const(False, "bool"))

    expected = get_unary_mod(method, dt="bool")
    shape1 = (4, 16, 32, 32)
    verify_model_struct("or_4d_constant", {}, expected(shape1))


def test_select():
    @I.ir_module
    class expected:
        @R.function
        def main(
            cond: R.Tensor((4, 16, 32, 32), dtype="bool"),
            input1: R.Tensor((4, 16, 32, 32), dtype="float32"),
            input2: R.Tensor((4, 16, 32, 32), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.where(cond, input1, input2)
                R.output(gv)
            return gv

    verify_model_struct("select_4d", {}, expected)

    def get_custom_mod(tf):
        @I.ir_module
        class expected:
            @R.function
            def main(
                input1: R.Tensor((4, 16, 32, 32), dtype="float32"),
                input2: R.Tensor((4, 16, 32, 32), dtype="float32"),
            ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.where(
                        R.const(tf, "bool"), input1, input2
                    )
                    R.output(gv)
                return gv

        return expected

    verify_model_struct("select_4d_true", {}, get_custom_mod(True))
    verify_model_struct("select_4d_false", {}, get_custom_mod(False))


def test_sqr():
    def method(in1):
        return R.power(in1, R.const(2, "float32"))

    expected = get_unary_mod(method)
    shape = (4, 16)
    verify_model_struct("sqr_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("sqr_4d", {}, expected(shape))


def test_sqrt():
    expected = get_unary_mod(R.sqrt)
    shape = (4, 16)
    verify_model_struct("sqrt_2d", {}, expected(shape))
    shape = (4, 16, 32, 32)
    verify_model_struct("sqrt_4d", {}, expected(shape))


def test_rsqr():
    def method(in1):
        return R.power(in1, R.const(-2, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16)
    verify_model_struct("rsqr_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("rsqr_4d", {}, expected(shape1))


def test_rsqrt():
    expected = get_unary_mod(R.rsqrt)
    shape1 = (4, 16)
    verify_model_struct("rsqrt_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("rsqrt_4d", {}, expected(shape1))


def test_log2():
    @I.ir_module
    class expected:
        @R.function
        def main(input: R.Tensor((4, 16), dtype="float32")) -> R.Tensor((4, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.log2, input)
                gv: R.Tensor((4, 16), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("log2_2d", {}, expected)

    @I.ir_module
    class expected4:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.log2, input)
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("log2_4d", {}, expected4)


def test_min():
    expected = get_binary_mod(R.minimum)
    shape1 = (4, 16)
    verify_model_struct("min_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("min_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("min_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.minimum(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("min_4d_constant", {}, expected(shape1))


def test_max():
    expected = get_binary_mod(R.maximum)
    shape1 = (4, 16)
    verify_model_struct("max_2d", {}, expected(shape1))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("max_4d", {}, expected(shape1))

    shape1, shape2 = (4, 16, 32, 32), (1, 16, 1, 1)
    verify_model_struct("max_4d_broadcast", {}, expected(shape1, shape2))

    def method(in1):
        return R.maximum(in1, R.const(0.5, "float32"))

    expected = get_unary_mod(method)
    shape1 = (4, 16, 32, 32)
    verify_model_struct("max_4d_constant", {}, expected(shape1))


def test_clamp():
    # custom module needed
    def get_custom(dtype="float32"):
        def _appl_shape(shape):
            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input1: R.Tensor(shape, dtype=dtype),
                    input2: R.Tensor(shape, dtype=dtype),
                    input3: R.Tensor(shape, dtype=dtype),
                ) -> R.Tensor(shape, dtype=dtype):
                    R.func_attr({"num_input": 3})
                    with R.dataflow():
                        lv: R.Tensor(shape, dtype=dtype) = R.minimum(input1, input3)
                        gv: R.Tensor(shape, dtype=dtype) = R.maximum(lv, input2)
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom()
    shape = (4, 16)
    verify_model_struct("clamp_2d", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("clamp_4d", {}, expected(shape))

    # constant limit
    # limits need to be extracted from graph
    graph = get_case_graph("clamp_4d_constant")
    print(graph.operations[1])
    op = graph.operations[1]
    lowlim = op.inputs["a"]
    highlim = op.inputs["b"]
    shape = (4, 16, 32, 32)

    @tvm.script.ir.ir_module
    class expected:
        @R.function
        def main(input1: R.Tensor(shape, dtype="float32")) -> R.Tensor(shape, dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor(shape, dtype="float32") = R.clip(
                    input1, R.prim_value(T.float32(lowlim)), R.prim_value(T.float32(highlim))
                )
                R.output(lv)
            return lv

    verify_model_struct("clamp_4d_constant", {}, expected)


def test_conv():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 8, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 8, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 16), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 16, 1, 1])
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    binding = {
        "filter": np.ones((16, 8, 3, 3), dtype="float32"),
        "bias": np.ones(
            (
                1,
                16,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("conv3x3", binding, expected)

    @I.ir_module
    class expected_stride:
        @R.function
        def main(
            input: R.Tensor((4, 8, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 8, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 16), dtype="float32"),
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.conv2d(
                    input,
                    filter,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 16, 1, 1])
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    verify_model_struct("conv3x3_stride2x2", binding, expected_stride)

    @I.ir_module
    class expected_group:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 1, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 16), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 16, 1, 1])
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    binding = {
        "filter": np.ones((16, 1, 3, 3), dtype="float32"),
        "bias": np.ones(
            (
                1,
                16,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("conv3x3_groups0", binding, expected_group)


def test_deconv():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 8, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 8), dtype="float32"),
        ) -> R.Tensor((4, 8, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 8, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 8, 1, 1])
                )
                gv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    binding = {
        "filter": np.ones((16, 8, 3, 3), dtype="float32"),
        "bias": np.ones(
            (
                1,
                8,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("deconv3x3", binding, expected)

    @I.ir_module
    class expected_stride:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 8, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 8), dtype="float32"),
        ) -> R.Tensor((4, 8, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 64, 64), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    filter,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 8, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 8, 1, 1])
                )
                gv: R.Tensor((4, 8, 64, 64), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    verify_model_struct("deconv3x3_stride2x2", binding, expected_stride)

    @I.ir_module
    class expected_group:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            filter: R.Tensor((16, 1, 3, 3), dtype="float32"),
            bias: R.Tensor((1, 16), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 16, 1, 1])
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv, lv1)
                R.output(gv)
            return gv

    binding = {
        "filter": np.ones((16, 1, 3, 3), dtype="float32"),
        "bias": np.ones(
            (
                1,
                16,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("deconv3x3_groups0", binding, expected_group)


def test_box():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.ones(
                    R.shape([16, 1, 3, 3]), dtype="float32"
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("box3x3", {}, expected)

    @I.ir_module
    class expected_stride:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.ones(
                    R.shape([16, 1, 3, 3]), dtype="float32"
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("box3x3_stride1x1", {}, expected_stride)


def test_debox():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.ones(
                    R.shape([16, 1, 3, 3]), dtype="float32"
                )
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("debox3x3", {}, expected)

    @I.ir_module
    class expected_stride:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.ones(
                    R.shape([16, 1, 3, 3]), dtype="float32"
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("debox3x3_stride1x1", {}, expected_stride)


def test_nearest_downsample():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 1, 1), dtype="float32") = R.ones(
                    R.shape([16, 1, 1, 1]), dtype="float32"
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("nearest_downsample", {}, expected)


def test_area_downsample():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 2, 2), dtype="float32") = R.full(
                    R.shape([16, 1, 2, 2]), R.const(0.25, "float32"), dtype="float32"
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    verify_model_struct("area_downsample", {}, expected)


def test_nearest_upsample():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(
                    topi.image.resize2d,
                    input,
                    [0, 0, 0, 0],
                    [64, 64],
                    method="nearest_neighbor",
                    rounding_method="round",
                )
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("nearest_upsample", {}, expected)


def test_bilinear_upsample():
    @I.ir_module
    class expected_s_c:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((1, 1, 4, 4), dtype="float32"),
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 4, 4), dtype="float32") = R.tile(w1, repeats=[16, 1, 1, 1])
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    binding = {
        "w1": np.array(
            [
                [
                    [
                        [0.0625, 0.1875, 0.1875, 0.0625],
                        [0.1875, 0.5625, 0.5625, 0.1875],
                        [0.1875, 0.5625, 0.5625, 0.1875],
                        [0.0625, 0.1875, 0.1875, 0.0625],
                    ]
                ]
            ],
            dtype="float32",
        )
    }
    verify_model_struct("bilinear_upsample_symmetric_constant", binding, expected_s_c)

    @I.ir_module
    class expected_s_r:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(
                    topi.image.resize2d,
                    input,
                    [0, 0, 0, 0],
                    [64, 64],
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                )
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("bilinear_upsample_symmetric_replicate", {}, expected_s_r)

    @I.ir_module
    class expected_a_c:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(
                    topi.image.resize2d,
                    input,
                    [0, 0, 0, 0],
                    [64, 64],
                    method="linear",
                    coordinate_transformation_mode="align_corners",
                )
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("bilinear_upsample_aligned_constant", {}, expected_a_c)

    @I.ir_module
    class expected_a_r:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(
                    topi.image.resize2d,
                    input,
                    [0, 0, 0, 0],
                    [64, 64],
                    method="linear",
                    coordinate_transformation_mode="align_corners",
                )
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("bilinear_upsample_aligned_replicate", {}, expected_a_r)

    @I.ir_module
    class expected_as_c:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((1, 1, 4, 4), dtype="float32"),
        ) -> R.Tensor((4, 16, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.tile(w1, repeats=[16, 1, 1, 1])
                gv: R.Tensor((4, 16, 64, 64), dtype="float32") = R.nn.conv2d_transpose(
                    input,
                    lv,
                    strides=[2, 2],
                    padding=[1, 1, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    binding = {
        "w1": np.array([[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]], dtype="float32")
    }
    verify_model_struct("bilinear_upsample_asymmetric_constant", binding, expected_as_c)

    # Skip because Replicate - Edge mode is currently not supported in Relax
    # verify_model_struct("bilinear_upsample_asymmetric_replicate", {}, None)


def test_sum_reduce():
    def method(in1):
        return R.sum(in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 1, 32, 32)
    verify_model_struct("sum_reduce_channel", {}, expected(shape, o_shape))

    def method(in1):
        return R.sum(in1, axis=[2, 3], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 1, 1)
    verify_model_struct("sum_reduce_spatial", {}, expected(shape, o_shape))


def test_max_reduce():
    def method(in1):
        return R.max(in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 1, 32, 32)
    verify_model_struct("max_reduce_channel", {}, expected(shape, o_shape))

    def method(in1):
        return R.max(in1, axis=[2, 3], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 1, 1)
    verify_model_struct("max_reduce_spatial", {}, expected(shape, o_shape))


def test_min_reduce():
    def method(in1):
        return R.min(in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 1, 32, 32)
    verify_model_struct("min_reduce_channel", {}, expected(shape, o_shape))

    def method(in1):
        return R.min(in1, axis=[2, 3], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 1, 1)
    verify_model_struct("min_reduce_spatial", {}, expected(shape, o_shape))


def test_argmax_reduce():
    @I.ir_module
    class expected_ch:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 1, 32, 32), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.argmax, input, [1], keepdims=True)
                gv: R.Tensor((4, 1, 32, 32), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model_struct("argmax_reduce_channel", {}, expected_ch)

    @I.ir_module
    class expected_sp:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 1, 1), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.argmax, input, [2, 3], keepdims=True)
                gv: R.Tensor((4, 16, 1, 1), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model_struct("argmax_reduce_spatial", {}, expected_sp)


def test_argmin_reduce():
    @I.ir_module
    class expected_ch:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 1, 32, 32), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.argmin, input, [1], keepdims=True)
                gv: R.Tensor((4, 1, 32, 32), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model_struct("argmin_reduce_channel", {}, expected_ch)

    @I.ir_module
    class expected_sp:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 1, 1), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.argmin, input, [2, 3], keepdims=True)
                gv: R.Tensor((4, 16, 1, 1), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model_struct("argmin_reduce_spatial", {}, expected_sp)


def test_all_reduce():
    @I.ir_module
    class expected_ch:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="bool")
        ) -> R.Tensor((4, 1, 32, 32), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.all, input, [1], True)
                gv: R.Tensor((4, 1, 32, 32), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model_struct("all_reduce_channel", {}, expected_ch)

    @I.ir_module
    class expected_sp:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="bool")
        ) -> R.Tensor((4, 16, 1, 1), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.all, input, [2, 3], True)
                gv: R.Tensor((4, 16, 1, 1), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model_struct("all_reduce_spatial", {}, expected_sp)


def test_any_reduce():
    @I.ir_module
    class expected_ch:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="bool")
        ) -> R.Tensor((4, 1, 32, 32), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.any, input, [1], True)
                gv: R.Tensor((4, 1, 32, 32), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model_struct("any_reduce_channel", {}, expected_ch)

    @I.ir_module
    class expected_sp:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="bool")
        ) -> R.Tensor((4, 16, 1, 1), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.any, input, [2, 3], True)
                gv: R.Tensor((4, 16, 1, 1), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model_struct("any_reduce_spatial", {}, expected_sp)


def test_mean_reduce():
    def method(in1):
        return R.mean(in1, axis=[2, 3], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 1, 1)
    verify_model_struct("mean_reduce_spatial", {}, expected(shape, o_shape))

    def method(in1):
        return R.mean(in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 1, 32, 32)
    verify_model_struct("mean_reduce_channel", {}, expected(shape, o_shape))


def test_reshape():
    def met(in1):
        return R.reshape(in1, R.shape(list(o_shape)))

    shape = (2, 3, 3, 3, 2)
    o_shape = (2, 3, 9, 2)
    expected = get_unary_mod(met)
    verify_model_struct("reshape_partial", {}, expected(shape, o_shape))

    shape = (4, 16, 1, 1)
    o_shape = (4, 16)
    verify_model_struct("reshape_squeeze", {}, expected(shape, o_shape))

    shape = (4, 16, 32, 32)
    o_shape = (4, 16384)
    verify_model_struct("reshape_flatten", {}, expected(shape, o_shape))


def test_squeeze():
    def method(in1):
        return R.squeeze(in1, axis=[2, 3])

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 1, 1), (4, 16)
    verify_model_struct("squeeze_spatial", {}, expected(shape, o_shape))


def test_unsqueeze():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16), dtype="float32")
        ) -> R.Tensor((4, 16, 1, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 1), dtype="float32") = R.expand_dims(input, axis=[2])
                gv: R.Tensor((4, 16, 1, 1), dtype="float32") = R.expand_dims(lv, axis=[3])
                R.output(gv)
            return gv

    verify_model_struct("unsqueeze", {}, expected)


def test_transpose():
    def method(in1):
        return R.permute_dims(in1, axes=[0, 3, 1, 2])

    expected = get_unary_mod(method)
    shape, o_shape = (4, 32, 32, 16), (4, 16, 32, 32)
    verify_model_struct("transpose_nhwc_to_nchw", {}, expected(shape, o_shape))

    def method(in1):
        return R.permute_dims(in1, axes=[0, 2, 3, 1])

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 32, 32, 16)
    verify_model_struct("transpose_nchw_to_nhwc", {}, expected(shape, o_shape))


def test_split():
    @tvm.script.ir.ir_module
    class expected_ch:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((4, 8, 32, 32), dtype="float32"), R.Tensor((4, 8, 32, 32), dtype="float32")
        ):
            R.func_attr({"num_input": 1})
            with (R.dataflow()):
                lv: R.Tuple(
                    R.Tensor((4, 8, 32, 32), dtype="float32"),
                    R.Tensor((4, 8, 32, 32), dtype="float32"),
                ) = R.split(input, indices_or_sections=[8], axis=1)
                lv1: R.Tensor((4, 8, 32, 32), dtype="float32") = lv[0]
                lv2: R.Tensor((4, 8, 32, 32), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((4, 8, 32, 32), dtype="float32"),
                    R.Tensor((4, 8, 32, 32), dtype="float32"),
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    verify_model_struct("split_channel", {}, expected_ch)

    @tvm.script.ir.ir_module
    class expected_ub:
        @R.function
        def main(
            input: R.Tensor((4, 32, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((4, 12, 3), dtype="float32"),
            R.Tensor((4, 4, 3), dtype="float32"),
            R.Tensor((4, 16, 3), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((4, 12, 3), dtype="float32"),
                    R.Tensor((4, 4, 3), dtype="float32"),
                    R.Tensor((4, 16, 3), dtype="float32"),
                ) = R.split(input, indices_or_sections=[12, 16], axis=1)
                lv1: R.Tensor((4, 12, 3), dtype="float32") = lv[0]
                lv2: R.Tensor((4, 4, 3), dtype="float32") = lv[1]
                lv3: R.Tensor((4, 16, 3), dtype="float32") = lv[2]
                gv: R.Tuple(
                    R.Tensor((4, 12, 3), dtype="float32"),
                    R.Tensor((4, 4, 3), dtype="float32"),
                    R.Tensor((4, 16, 3), dtype="float32"),
                ) = (lv1, lv2, lv3)
                R.output(gv)
            return gv

    verify_model_struct("split_unbalanced", {}, expected_ub)


def test_concat():
    def method(in1, in2):
        return R.concat((in1, in2), axis=1)

    expected = get_binary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 32, 32, 32)
    verify_model_struct("concat_channel", {}, expected(shape, osh=o_shape))


def test_stack():
    @tvm.script.ir.ir_module
    class expected:
        @R.function
        def main(
            input1: R.Tensor((4, 16, 32, 32), dtype="float32"),
            input2: R.Tensor((4, 16, 32, 32), dtype="float32"),
        ) -> R.Tensor((4, 2, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((4, 1, 16, 32, 32), dtype="float32") = R.expand_dims(input1, axis=[1])
                lv1: R.Tensor((4, 1, 16, 32, 32), dtype="float32") = R.expand_dims(input2, axis=[1])
                gv: R.Tensor((4, 2, 16, 32, 32), dtype="float32") = R.concat((lv, lv1), axis=1)
                R.output(gv)
            return gv

    verify_model_struct("stack", {}, expected)


def test_unstack():
    @tvm.script.ir.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 3, 16), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((4, 16), dtype="float32"),
            R.Tensor((4, 16), dtype="float32"),
            R.Tensor((4, 16), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with (R.dataflow()):
                lv: R.Tuple(
                    R.Tensor((4, 1, 16), dtype="float32"),
                    R.Tensor((4, 1, 16), dtype="float32"),
                    R.Tensor((4, 1, 16), dtype="float32"),
                ) = R.split(input, indices_or_sections=[1, 2], axis=1)
                lv1: R.Tensor((4, 1, 16), dtype="float32") = lv[0]
                lv2: R.Tensor((4, 16), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((4, 1, 16), dtype="float32") = lv[1]
                lv4: R.Tensor((4, 16), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv5: R.Tensor((4, 1, 16), dtype="float32") = lv[2]
                lv6: R.Tensor((4, 16), dtype="float32") = R.squeeze(lv5, axis=[1])
                gv: R.Tuple(
                    R.Tensor((4, 16), dtype="float32"),
                    R.Tensor((4, 16), dtype="float32"),
                    R.Tensor((4, 16), dtype="float32"),
                ) = (lv2, lv4, lv6)
                R.output(gv)
            return gv

    verify_model_struct("unstack", {}, expected)


def test_slice():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 30, 28), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 16, 30, 28), dtype="float32") = R.strided_slice(
                    input,
                    (R.prim_value(2), R.prim_value(3)),
                    (R.prim_value(1), R.prim_value(2)),
                    (R.prim_value(-1), R.prim_value(-2)),
                    (R.prim_value(1), R.prim_value(1)),
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    verify_model_struct("slice", {}, expected)

    @I.ir_module
    class expected_stride:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 4, 12, 29), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 4, 12, 29), dtype="float32") = R.strided_slice(
                    input,
                    (R.prim_value(1), R.prim_value(2), R.prim_value(3)),
                    (R.prim_value(5), R.prim_value(16), R.prim_value(2)),
                    (R.prim_value(1), R.prim_value(4), R.prim_value(-1)),
                    (R.prim_value(-1), R.prim_value(-1), R.prim_value(1)),
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    verify_model_struct("slice_strides", {}, expected_stride)


def test_pad():
    @I.ir_module
    class expected01:
        @R.function
        def main(
            input: R.Tensor((1, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((1, 16, 33, 33), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.nn.mirror_pad, input, [0, 0, 0, 0], [0, 0, 1, 1], "REFLECT")
                gv: R.Tensor((1, 16, 33, 33), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("pad_0-1_reflect", {}, expected01)

    @I.ir_module
    class expected10:
        @R.function
        def main(
            input: R.Tensor((1, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((1, 16, 33, 33), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.nn.mirror_pad, input, [0, 0, 1, 1], [0, 0, 0, 0], "REFLECT")
                gv: R.Tensor((1, 16, 33, 33), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("pad_1-0_reflect", {}, expected10)

    @I.ir_module
    class expected11:
        @R.function
        def main(
            input: R.Tensor((1, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((1, 16, 34, 34), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.nn.mirror_pad, input, [0, 0, 1, 1], [0, 0, 1, 1], "REFLECT")
                gv: R.Tensor((1, 16, 34, 34), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("pad_1-1_reflect", {}, expected11)

    def method_wr(pw):
        def method(in1):
            return R.nn.pad(in1, pad_value=R.const(0, "float32"), pad_width=pw, pad_mode="constant")

        return method

    expected = get_unary_mod(method_wr([0, 0, 0, 0, 0, 1, 0, 1]))
    shape, o_shape = (1, 16, 32, 32), (1, 16, 33, 33)
    verify_model_struct("pad_0-1_constant", {}, expected(shape, o_shape))

    expected = get_unary_mod(method_wr([0, 0, 0, 0, 1, 0, 1, 0]))
    verify_model_struct("pad_1-0_constant", {}, expected(shape, o_shape))

    expected = get_unary_mod(method_wr([0, 0, 0, 0, 1, 1, 1, 1]))
    o_shape = (1, 16, 34, 34)
    verify_model_struct("pad_1-1_constant", {}, expected(shape, o_shape))

    # Replicate - Edge mode is currently not supported in TVM relax
    # verify_model_struct("pad_0-1_replicate", {}, None)
    # verify_model_struct("pad_1-0_replicate", {}, None)
    # verify_model_struct("pad_1-1_replicate", {}, None)


def test_tile():
    def method(in1):
        return R.tile(
            in1,
            repeats=[
                1,
                1,
                3,
                3,
            ],
        )

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 96, 96)
    verify_model_struct("tile_spatial", {}, expected(shape, o_shape))

    def method(in1):
        return R.tile(in1, repeats=[1, 16])

    expected = get_unary_mod(method)
    shape, o_shape = (16, 1), (16, 16)
    verify_model_struct("tile_channel", {}, expected(shape, o_shape))

    def method(in1):
        return R.tile(in1, repeats=[16, 1])

    expected = get_unary_mod(method)
    shape, o_shape = (1, 16), (16, 16)
    verify_model_struct("tile_batch", {}, expected(shape, o_shape))


def test_matmul():
    def method(in1, in2):
        return R.matmul(in1, in2, out_dtype="void")

    shape1, shape2 = (4, 16), (16, 4)
    expected = get_binary_mod(method)
    verify_model_struct("matmul_2d", {}, expected(shape1, shape2, (4, 4)))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("matmul_4d", {}, expected(shape1))

    def get_custom_mod(dtype="float32"):
        def _appl_shape(sh1, sh2, osh, axes):
            global shape1, shape2, o_shape
            shape1, shape2, o_shape = sh1, sh2, osh
            sh1_t = tuple([shape1[i] for i in axes])

            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    lhs: R.Tensor(shape1, dtype), rhs: R.Tensor(shape2, dtype)
                ) -> R.Tensor(o_shape, "float32"):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv: R.Tensor(sh1_t, dtype=dtype) = R.permute_dims(lhs, axes=axes)
                        gv: R.Tensor(o_shape, dtype="float32") = R.matmul(lv, rhs, out_dtype="void")
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    shape1 = (4, 16)
    expected = get_custom_mod()
    verify_model_struct("matmul_2d_transpose", {}, expected(shape1, shape1, (16, 16), [1, 0]))

    shape1 = (4, 16, 32, 32)
    verify_model_struct("matmul_4d_transpose", {}, expected(shape1, shape1, shape1, [0, 1, 3, 2]))


def test_sigmoid():
    expected = get_unary_mod(R.sigmoid)
    shape = (4, 16)
    verify_model_struct("sigmoid_2d_standalone", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("sigmoid_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((4, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.sigmoid(lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("sigmoid", binding, expected)


def test_relu():
    expected = get_unary_mod(R.nn.relu)
    shape = (4, 16)
    verify_model_struct("relu_2d_standalone", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("relu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((4, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.relu(lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("relu", binding, expected)


def test_prelu():
    def get_custom_mod(dtype="float32"):
        def _appl_shape(sh1, sh2, osh, axes):
            global shape1, shape2, o_shape
            shape1, shape2, o_shape = sh1, sh2, osh
            expanded = [1] * len(sh1)
            expanded[1] = sh2[0]

            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input1: R.Tensor(shape1, dtype="float32"),
                    input2: R.Tensor(shape2, dtype="float32"),
                ) -> R.Tensor(o_shape, dtype="float32"):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv: R.Tensor(shape1, dtype="bool") = R.less(input1, R.const(0, "float32"))
                        lv1: R.Tensor(expanded, dtype="float32") = R.expand_dims(input2, axis=axes)
                        lv2: R.Tensor(o_shape, dtype="float32") = R.multiply(lv1, input1)
                        gv: R.Tensor(o_shape, dtype="float32") = R.where(lv, lv2, input1)
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom_mod()
    shape1, shape2 = (16, 16), (16,)
    verify_model_struct("prelu_2d_standalone", {}, expected(shape1, shape2, shape1, [0]))

    shape1, shape2 = (16, 16, 32, 32), (16,)
    verify_model_struct("prelu_4d_standalone", {}, expected(shape1, shape2, shape1, [0, 2, 3]))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input1: R.Tensor((16, 16, 32, 32), dtype="float32"),
            input2: R.Tensor((16,), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((16, 16, 32, 32), dtype="bool") = R.less(lv, R.const(0, "float32"))
                lv2: R.Tensor((1, 16, 1, 1), dtype="float32") = R.expand_dims(
                    input2, axis=[0, 2, 3]
                )
                lv3: R.Tensor((16, 16, 32, 32), dtype="float32") = R.multiply(lv2, lv)
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.where(lv1, lv3, lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("prelu", binding, expected)


def test_leaky_relu():
    alpha = 0.5

    def method(in1):
        return R.nn.leakyrelu(in1, alpha)

    expected = get_unary_mod(method)
    shape = (16, 16)

    verify_model_struct("leaky_relu_2d_standalone", {}, expected(shape))

    shape = (16, 16, 32, 32)
    verify_model_struct("leaky_relu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((16, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.leakyrelu(lv, alpha=0.5)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("leaky_relu", binding, expected)


def test_elu():
    def get_custom_mod():
        def _appl_shape(shape):
            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input: R.Tensor(shape, dtype="float32")
                ) -> R.Tensor(shape, dtype="float32"):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        lv: R.Tensor(shape, dtype="float32") = R.exp(input)
                        lv1: R.Tensor(shape, dtype="bool") = R.less(input, R.const(0, "float32"))
                        lv2: R.Tensor(shape, dtype="float32") = R.subtract(
                            lv, R.const(1, "float32")
                        )
                        lv3: R.Tensor(shape, dtype="float32") = R.multiply(
                            R.const(1, "float32"), lv2
                        )
                        gv: R.Tensor(shape, dtype="float32") = R.where(lv1, lv3, input)
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom_mod()
    shape = (16, 16)

    verify_model_struct("elu_2d_standalone", {}, expected(shape))

    shape = (16, 16, 32, 32)
    verify_model_struct("elu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((16, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((16, 16, 32, 32), dtype="float32") = R.exp(lv)
                lv2: R.Tensor((16, 16, 32, 32), dtype="bool") = R.less(lv, R.const(0, "float32"))
                lv3: R.Tensor((16, 16, 32, 32), dtype="float32") = R.subtract(
                    lv1, R.const(1, "float32")
                )
                lv4: R.Tensor((16, 16, 32, 32), dtype="float32") = R.multiply(
                    R.const(1, "float32"), lv3
                )
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.where(lv2, lv4, lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("elu", binding, expected)


def test_selu():
    def get_custom_mod():
        def _appl_shape(shape):
            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input: R.Tensor(shape, dtype="float32")
                ) -> R.Tensor(shape, dtype="float32"):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        lv: R.Tensor(shape, dtype="float32") = R.exp(input)
                        lv1: R.Tensor(shape, dtype="bool") = R.less(input, R.const(0, "float32"))
                        lv2: R.Tensor(shape, dtype="float32") = R.subtract(
                            lv, R.const(1, "float32")
                        )
                        lv3: R.Tensor(shape, dtype="float32") = R.multiply(
                            R.const(1.6732631921768188, "float32"), lv2
                        )
                        lv4: R.Tensor(shape, dtype="float32") = R.where(lv1, lv3, input)
                        gv: R.Tensor(shape, dtype="float32") = R.multiply(
                            R.const(1.0507010221481323, "float32"), lv4
                        )
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom_mod()
    shape = (16, 16)
    verify_model_struct("selu_2d_standalone", {}, expected(shape))

    shape = (16, 16, 32, 32)
    verify_model_struct("selu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((16, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((16, 16, 32, 32), dtype="float32") = R.exp(lv)
                lv2: R.Tensor((16, 16, 32, 32), dtype="bool") = R.less(lv, R.const(0, "float32"))
                lv3: R.Tensor((16, 16, 32, 32), dtype="float32") = R.subtract(
                    lv1, R.const(1, "float32")
                )
                lv4: R.Tensor((16, 16, 32, 32), dtype="float32") = R.multiply(
                    R.const(1.6732631921768188, "float32"), lv3
                )
                lv5: R.Tensor((16, 16, 32, 32), dtype="float32") = R.where(lv2, lv4, lv)
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.multiply(
                    R.const(1.0507010221481323, "float32"), lv5
                )
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("selu", binding, expected)


def test_gelu():
    expected = get_unary_mod(R.nn.gelu)
    shape = (16, 16)
    verify_model_struct("gelu_2d_standalone", {}, expected(shape))

    shape = (16, 16, 32, 32)
    verify_model_struct("gelu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((16, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.gelu(lv)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("gelu", binding, expected)


def test_silu():
    def get_custom_mod():
        def _appl_shape(shape):
            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input: R.Tensor(shape, dtype="float32")
                ) -> R.Tensor(shape, dtype="float32"):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        lv: R.Tensor(shape, dtype="float32") = R.sigmoid(input)
                        gv: R.Tensor(shape, dtype="float32") = R.multiply(input, lv)
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom_mod()
    shape = (16, 16)

    verify_model_struct("silu_2d_standalone", {}, expected(shape))

    shape = (16, 16, 32, 32)
    verify_model_struct("silu_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((16, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((16, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((16, 16, 32, 32), dtype="float32") = R.sigmoid(lv)
                gv: R.Tensor((16, 16, 32, 32), dtype="float32") = R.multiply(lv, lv1)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("silu", binding, expected)


def test_softmax():
    def method(in1):
        return R.nn.softmax(in1, axis=1)

    expected = get_unary_mod(method)
    shape = (4, 16)
    verify_model_struct("softmax_2d_standalone", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("softmax_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.softmax(lv, axis=1)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("softmax", binding, expected)


def test_softplus():
    def get_custom_mod():
        def _appl_shape(shape):
            @tvm.script.ir.ir_module
            class expected:
                @R.function
                def main(
                    input: R.Tensor(shape, dtype="float32")
                ) -> R.Tensor(shape, dtype="float32"):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        lv: R.Tensor(shape, dtype="float32") = R.exp(input)
                        lv1: R.Tensor(shape, dtype="float32") = R.add(lv, R.const(1, "float32"))
                        gv: R.Tensor(shape, dtype="float32") = R.log(lv1)
                        R.output(gv)
                    return gv

            return expected

        return _appl_shape

    expected = get_custom_mod()
    shape = (4, 16)

    verify_model_struct("softplus_2d_standalone", {}, expected(shape))

    shape = (4, 16, 32, 32)
    verify_model_struct("softplus_4d_standalone", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            w1: R.Tensor((16, 1, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((4, 16, 32, 32), dtype="float32") = R.exp(lv)
                lv2: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv1, R.const(1, "float32"))
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.log(lv2)
                R.output(gv)
            return gv

    binding = {"w1": np.ones([16, 1, 1, 1], dtype="float32")}
    verify_model_struct("softplus", binding, expected)


def test_linear():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16), dtype="float32"),
            weights: R.Tensor((32, 16), dtype="float32"),
            bias: R.Tensor((1, 32), dtype="float32"),
        ) -> R.Tensor((4, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 32), dtype="float32") = R.permute_dims(weights, axes=[1, 0])
                lv1: R.Tensor((4, 32), dtype="float32") = R.matmul(input, lv, out_dtype="void")
                lv2: R.Tensor((1, 32), dtype="float32") = R.reshape(bias, R.shape([1, 32]))
                gv: R.Tensor((4, 32), dtype="float32") = R.add(lv1, lv2)
                R.output(gv)
            return gv

    binding = {
        "weights": np.ones([32, 16], dtype="float32"),
        "bias": np.ones([1, 32], dtype="float32"),
    }
    verify_model_struct("linear", binding, expected)

    @I.ir_module
    class expected_nb:
        @R.function
        def main(
            input: R.Tensor((4, 16), dtype="float32"), weights: R.Tensor((32, 16), dtype="float32")
        ) -> R.Tensor((4, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 32), dtype="float32") = R.permute_dims(weights, axes=[1, 0])
                gv: R.Tensor((4, 32), dtype="float32") = R.matmul(input, lv, out_dtype="void")
                R.output(gv)
            return gv

    binding = {"weights": np.ones([32, 16], dtype="float32")}
    verify_model_struct("linear_nobias", binding, expected_nb)


def test_separable_conv():
    @tvm.script.ir.ir_module
    class expected_wb:
        @R.function
        def main(
            input1: R.Tensor((4, 8, 32, 32), dtype="float32"),
            plane_filter: R.Tensor((8, 1, 3, 3), dtype="float32"),
            point_filter: R.Tensor((16, 8, 1, 1), dtype="float32"),
            bias: R.Tensor(
                (
                    1,
                    16,
                ),
                dtype="float32",
            ),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.nn.conv2d(
                    input1,
                    plane_filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=8,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    lv,
                    point_filter,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv2: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 16, 1, 1])
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv1, lv2)
                R.output(gv)
            return gv

    binding = {
        "plane_filter": np.ones((8, 1, 3, 3), dtype="float32"),
        "point_filter": np.ones((16, 8, 1, 1), dtype="float32"),
        "bias": np.ones(
            (
                1,
                16,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("separable_conv3x3", binding, expected_wb)

    @tvm.script.ir.ir_module
    class expected:
        @R.function
        def main(
            input1: R.Tensor((4, 8, 32, 32), dtype="float32"),
            plane_filter: R.Tensor((8, 1, 3, 3), dtype="float32"),
            point_filter: R.Tensor((16, 8, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 16, 16), dtype="float32") = R.nn.conv2d(
                    input1,
                    plane_filter,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    dilation=[1, 1],
                    groups=8,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.conv2d(
                    lv,
                    point_filter,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    binding = {
        "plane_filter": np.ones((8, 1, 3, 3), dtype="float32"),
        "point_filter": np.ones((16, 8, 1, 1), dtype="float32"),
    }
    verify_model_struct("separable_conv3x3_with_attrs", binding, expected)


def test_separable_deconv():
    @tvm.script.ir.ir_module
    class expected_wb:
        @R.function
        def main(
            input1: R.Tensor((4, 16, 32, 32), dtype="float32"),
            plane_filter: R.Tensor((8, 1, 3, 3), dtype="float32"),
            point_filter: R.Tensor((16, 8, 1, 1), dtype="float32"),
            bias: R.Tensor(
                (
                    1,
                    8,
                ),
                dtype="float32",
            ),
        ) -> R.Tensor((4, 8, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    input1,
                    point_filter,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv1: R.Tensor((4, 8, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    lv,
                    plane_filter,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=8,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv2: R.Tensor((1, 16, 1, 1), dtype="float32") = R.reshape(
                    bias, R.shape([1, 8, 1, 1])
                )
                gv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.add(lv1, lv2)
                R.output(gv)
            return gv

    binding = {
        "plane_filter": np.ones((8, 1, 3, 3), dtype="float32"),
        "point_filter": np.ones((16, 8, 1, 1), dtype="float32"),
        "bias": np.ones(
            (
                1,
                8,
            ),
            dtype="float32",
        ),
    }
    verify_model_struct("separable_deconv3x3", binding, expected_wb)

    @tvm.script.ir.ir_module
    class expected:
        @R.function
        def main(
            input1: R.Tensor((4, 16, 32, 32), dtype="float32"),
            plane_filter: R.Tensor((8, 1, 3, 3), dtype="float32"),
            point_filter: R.Tensor((16, 8, 1, 1), dtype="float32"),
        ) -> R.Tensor((4, 8, 64, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 8, 32, 32), dtype="float32") = R.nn.conv2d_transpose(
                    input1,
                    point_filter,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 8, 64, 64), dtype="float32") = R.nn.conv2d_transpose(
                    lv,
                    plane_filter,
                    strides=[2, 2],
                    padding=[0, 0, 1, 1],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    groups=8,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                R.output(gv)
            return gv

    binding = {
        "plane_filter": np.ones((8, 1, 3, 3), dtype="float32"),
        "point_filter": np.ones((16, 8, 1, 1), dtype="float32"),
    }
    verify_model_struct("separable_deconv3x3_with_attrs", binding, expected)


def test_max_pool():
    def method(in1):
        return R.nn.max_pool2d(
            in1,
            pool_size=[3, 3],
            strides=[2, 2],
            dilation=[1, 1],
            padding=[0, 0, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            layout="NCHW",
            out_layout="NCHW",
        )

    expected = get_unary_mod(method)
    shape, o_shape = (4, 16, 32, 32), (4, 16, 16, 16)

    verify_model_struct("max_pool3x3", {}, expected(shape, o_shape))

    def method(in1):
        return R.nn.max_pool2d(
            in1,
            pool_size=[3, 3],
            strides=[1, 1],
            dilation=[1, 1],
            padding=[1, 1, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            layout="NCHW",
            out_layout="NCHW",
        )

    expected = get_unary_mod(method)
    shape = (4, 16, 32, 32)
    verify_model_struct("max_pool3x3_stride1x1", {}, expected(shape))

    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 33, 33), dtype="float32") = R.nn.pad(
                    input,
                    pad_value=R.const(0, "float32"),
                    pad_width=[0, 0, 0, 0, 0, 1, 0, 1],
                    pad_mode="constant",
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.max_pool2d(
                    lv,
                    pool_size=[3, 3],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                R.output(gv)
            return gv

    verify_model_struct("max_pool3x3_constant-border", {}, expected)


def test_avg_pool():
    shape, o_shape = (4, 16, 32, 32), (4, 16, 16, 16)

    def method(in1):
        return R.nn.avg_pool2d(
            in1,
            pool_size=[3, 3],
            strides=[2, 2],
            dilation=[1, 1],
            padding=[0, 0, 1, 1],
            ceil_mode=False,
            count_include_pad=True,
            layout="NCHW",
            out_layout="NCHW",
        )

    expected = get_unary_mod(method)
    verify_model_struct("avg_pool3x3", {}, expected(shape, o_shape))

    def method(in1):
        return R.nn.avg_pool2d(
            in1,
            pool_size=[3, 3],
            strides=[1, 1],
            dilation=[1, 1],
            padding=[1, 1, 1, 1],
            ceil_mode=False,
            count_include_pad=True,
            layout="NCHW",
            out_layout="NCHW",
        )

    expected = get_unary_mod(method)
    verify_model_struct("avg_pool3x3_stride1x1", {}, expected(shape))

    def method(in1):
        return R.nn.avg_pool2d(
            in1,
            pool_size=[3, 3],
            strides=[2, 2],
            dilation=[1, 1],
            padding=[0, 0, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            layout="NCHW",
            out_layout="NCHW",
        )

    expected = get_unary_mod(method)
    verify_model_struct("avg_pool3x3_ignore-border", {}, expected(shape, o_shape))


def test_rms_pool():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.power(
                    input, R.const(2, "float32")
                )
                lv1: R.Tensor((4, 16, 16, 16), dtype="float32") = R.nn.avg_pool2d(
                    lv,
                    pool_size=[3, 3],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 1, 1],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tensor((4, 16, 16, 16), dtype="float32") = R.sqrt(lv1)
                R.output(gv)
            return gv

    verify_model_struct("rms_pool3x3", {}, expected)


def test_local_response_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.emit_te(topi.nn.lrn, input, 5, 1, 1e-5, 0.75, 1.0)
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model_struct("local_response_normalization", {}, expected)


def test_local_mean_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.full(
                    R.shape([16, 1, 3, 3]), R.const(0.1111111119389534, "float32"), dtype="float32"
                )
                lv1: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.subtract(input, lv1)
                R.output(gv)
            return gv

    verify_model_struct("local_mean_normalization", {}, expected)


def test_local_variance_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.power(
                    input, R.const(2, "float32")
                )
                lv1: R.Tensor((16, 1, 3, 3), dtype="float32") = R.full(
                    R.shape([16, 1, 3, 3]), R.const(0.1111111119389534, "float32"), dtype="float32"
                )
                lv2: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv3: R.Tensor((4, 16, 32, 32), dtype="float32") = R.sqrt(lv2)
                lv4: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv3, R.const(1, "float32"))
                lv5: R.Tensor((4, 16, 32, 32), dtype="float32") = R.maximum(
                    lv4, R.const(9.9999997473787516e-06, "float32")
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.divide(input, lv5)
                R.output(gv)
            return gv

    verify_model_struct("local_variance_normalization", {}, expected)


def test_local_contrast_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16, 1, 3, 3), dtype="float32") = R.full(
                    R.shape([16, 1, 3, 3]), R.const(0.1111111119389534, "float32"), dtype="float32"
                )
                lv1: R.Tensor((4, 16, 32, 32), dtype="float32") = R.nn.conv2d(
                    input,
                    lv,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=16,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="void",
                )
                lv2: R.Tensor((4, 16, 32, 32), dtype="float32") = R.subtract(input, lv1)
                lv3: R.Tensor((4, 16, 32, 32), dtype="float32") = R.power(
                    lv2, R.const(2, "float32")
                )
                lv4: R.Tensor((16, 4, 32, 32), dtype="float32") = R.permute_dims(
                    lv3, axes=[1, 0, 2, 3]
                )
                lv5: R.Tensor((16, 4, 32, 32), dtype="float32") = R.nn.avg_pool2d(
                    lv4,
                    pool_size=[3, 3],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[1, 1, 1, 1],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv6: R.Tensor((4, 16, 32, 32), dtype="float32") = R.permute_dims(
                    lv5, axes=[1, 0, 2, 3]
                )
                lv7: R.Tensor((4, 16, 32, 32), dtype="float32") = R.sqrt(lv6)
                lv8: R.Tensor((4, 16, 32, 32), dtype="float32") = R.add(lv7, R.const(1, "float32"))
                lv9: R.Tensor((4, 16, 32, 32), dtype="float32") = R.maximum(
                    lv8, R.const(9.9999997473787516e-06, "float32")
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.divide(lv2, lv9)
                R.output(gv)
            return gv

    verify_model_struct("local_contrast_normalization", {}, expected)


def test_l1_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.abs(input)
                lv1: R.Tensor((4, 1, 32, 32), dtype="float32") = R.sum(lv, axis=[1], keepdims=True)
                lv2: R.Tensor((4, 1, 32, 32), dtype="float32") = R.add(lv1, R.const(1, "float32"))
                lv3: R.Tensor((4, 1, 32, 32), dtype="float32") = R.maximum(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.divide(input, lv3)
                R.output(gv)
            return gv

    verify_model_struct("l1_normalization", {}, expected)


def test_l2_normalization():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32")
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.power(
                    input, R.const(2, "float32")
                )
                lv1: R.Tensor((4, 1, 32, 32), dtype="float32") = R.sum(lv, axis=[1], keepdims=True)
                lv2: R.Tensor((4, 1, 32, 32), dtype="float32") = R.sqrt(lv1)
                lv3: R.Tensor((4, 1, 32, 32), dtype="float32") = R.add(lv2, R.const(0, "float32"))
                lv4: R.Tensor((4, 1, 32, 32), dtype="float32") = R.maximum(
                    lv3, R.const(0.0010000000474974513, "float32")
                )
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = R.divide(input, lv4)
                R.output(gv)
            return gv

    verify_model_struct("l2_normalization", {}, expected)


def test_batch_norm():
    @I.ir_module
    class expected:
        @R.function
        def main(
            input: R.Tensor((4, 16, 32, 32), dtype="float32"),
            mean: R.Tensor((1, 16), dtype="float32"),
            variance: R.Tensor((1, 16), dtype="float32"),
            offset: R.Tensor((1, 16), dtype="float32"),
            scale: R.Tensor((1, 16), dtype="float32"),
        ) -> R.Tensor((4, 16, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((16,), dtype="float32") = R.squeeze(scale, axis=[0])
                lv1: R.Tensor((16,), dtype="float32") = R.squeeze(offset, axis=[0])
                lv2: R.Tensor((16,), dtype="float32") = R.squeeze(mean, axis=[0])
                lv3: R.Tensor((16,), dtype="float32") = R.squeeze(variance, axis=[0])
                lv4 = R.emit_te(topi.nn.batch_norm, input, lv, lv1, lv2, lv3, 1, 1e-3)
                gv: R.Tensor((4, 16, 32, 32), dtype="float32") = lv4[0]
                R.output(gv)
            return gv

    binding = {
        "mean": np.ones((1, 16), dtype="float32"),
        "variance": np.ones((1, 16), dtype="float32"),
        "offset": np.ones((1, 16), dtype="float32"),
        "scale": np.ones((1, 16), dtype="float32"),
    }
    verify_model_struct("batch_norm", binding, expected)
