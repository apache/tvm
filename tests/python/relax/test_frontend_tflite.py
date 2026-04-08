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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TFLite to Relax converter tests"""

import os

import numpy as np
import pytest
import tensorflow as tf
import tflite.Model
from tensorflow.keras import applications as keras_app

import tvm
from tvm import relax
from tvm.relax.frontend.tflite import from_tflite
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R


def _get_mod_from_cfunc(cfunc):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cfunc])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model_buf = converter.convert()
    if hasattr(tflite.Model, "Model"):
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    else:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")
    return mod


def verify(TestClass, expected=None):
    if isinstance(TestClass, type):
        cf = TestClass().func.get_concrete_function()
    else:
        cf = TestClass
    mod = _get_mod_from_cfunc(cf)

    if expected:
        tvm.ir.assert_structural_equal(mod, expected)

    # Run E2E test only on nightly
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    # Inputs
    tf_inputs = []
    tvm_inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        data = np.random.uniform(0, 1, size=shape).astype(arg.struct_info.dtype)
        tvm_inputs.append(data)
        tf_inputs.append(tf.constant(data))

    # TF Run
    tf_output = cf(*tf_inputs)

    # TVM Run
    tgt = tvm.target.Target("llvm")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *tvm_inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")

    if isinstance(tf_output, tuple):
        for tf_out, tvm_out in zip(tf_output, tvm_output):
            np.testing.assert_allclose(tf_out.numpy(), tvm_out.numpy(), rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(tf_output.numpy(), tvm_output.numpy(), rtol=1e-5, atol=1e-5)


def test_add_one_2d():
    class AddOne2D(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x + 1

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.add(x, R.const(1.0, "float32"))
                R.output(gv)
            return gv

    verify(AddOne2D, Expected)


def test_add_n():
    class AddN(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
            ]
        )
        def func(self, x, y, z):
            return tf.add_n([x, y, z])

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
            y: R.Tensor((2, 2), dtype="float32"),
            z: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.add(x, y)
                gv: R.Tensor((2, 2), dtype="float32") = R.add(lv, z)
                R.output(gv)
            return gv

    verify(AddN, Expected)


def test_split():
    class Split(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b, c = tf.split(x, 3, axis=1)
            return tf.raw_ops.Pack(values=[a, b, c], axis=1)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 3, 10), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                ) = R.split(x, indices_or_sections=3, axis=1)
                lv1: R.Tensor((1, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 1, 10), dtype="float32") = R.expand_dims(lv1, axis=[1])
                lv3: R.Tensor((1, 10), dtype="float32") = lv[1]
                lv4: R.Tensor((1, 1, 10), dtype="float32") = R.expand_dims(lv3, axis=[1])
                lv5: R.Tensor((1, 10), dtype="float32") = lv[2]
                lv6: R.Tensor((1, 1, 10), dtype="float32") = R.expand_dims(lv5, axis=[1])
                gv: R.Tensor((1, 3, 10), dtype="float32") = R.concat((lv2, lv4, lv6), axis=1)
                R.output(gv)
            return gv

    verify(Split, Expected)


def test_pack():
    class Pack(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.raw_ops.Pack(values=[x, y], axis=0)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tensor((2, 2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.expand_dims(x, axis=[0])
                lv1: R.Tensor((1, 2, 3), dtype="float32") = R.expand_dims(y, axis=[0])
                gv: R.Tensor((2, 2, 3), dtype="float32") = R.concat((lv, lv1), axis=0)
                R.output(gv)
            return gv

    verify(Pack, Expected)


def test_cast():
    class Cast(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.cast(x, tf.int32)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="int32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="int32") = R.astype(x, dtype="int32")
                R.output(gv)
            return gv

    verify(Cast, Expected)


def test_expand_dims():
    class ExpandDims(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.expand_dims(x, axis=2)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30, 1), dtype="float32") = R.reshape(x, R.shape([1, 30, 1]))
                R.output(gv)
            return gv

    verify(ExpandDims, Expected)


def test_transpose():
    class Transpose(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            x = tf.expand_dims(x, axis=2)
            return tf.transpose(x, perm=[0, 2, 1])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 1, 30), dtype="float32") = R.reshape(x, R.shape([1, 1, 30]))
                R.output(gv)
            return gv

    verify(Transpose, Expected)


def test_reshape():
    class Reshape(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.reshape(x, (1, 2, 15))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 2, 15), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 2, 15), dtype="float32") = R.reshape(x, R.shape([1, 2, 15]))
                R.output(gv)
            return gv

    verify(Reshape, Expected)


def test_concat_v2():
    class ConcatV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b, c = tf.split(x, 3, axis=1)
            axis = tf.add(tf.constant(1, dtype="int32"), tf.constant(0, dtype="int32"))
            return tf.raw_ops.ConcatV2(values=[a, b, c], axis=axis)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                    R.Tensor((1, 10), dtype="float32"),
                ) = R.split(x, indices_or_sections=3, axis=1)
                lv1: R.Tensor((1, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 10), dtype="float32") = lv[1]
                lv3: R.Tensor((1, 10), dtype="float32") = lv[2]
                gv: R.Tensor((1, 30), dtype="float32") = R.concat((lv1, lv2, lv3), axis=1)
                R.output(gv)
            return gv

    verify(ConcatV2, Expected)


def test_multi_output():
    class MultiOutput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            y = 2 * x
            return x, y

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.multiply(x, R.const(2.0, "float32"))
                gv: R.Tuple(
                    R.Tensor((2, 2), dtype="float32"), R.Tensor((2, 2), dtype="float32")
                ) = (x, lv)
                R.output(gv)
            return gv

    verify(MultiOutput, Expected)


def test_elu():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.elu(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.exp(x)
                lv1: R.Tensor((1, 30), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((1, 30), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((1, 30), dtype="float32") = R.multiply(R.const(-1.0, "float32"), lv2)
                lv4: R.Tensor((1, 30), dtype="float32") = R.nn.relu(x)
                gv: R.Tensor((1, 30), dtype="float32") = R.add(lv3, lv4)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_gelu():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.gelu(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.multiply(
                    x, R.const(0.70710676908493042, "float32")
                )
                lv1: R.Tensor((1, 30), dtype="float32") = R.erf(lv)
                lv2: R.Tensor((1, 30), dtype="float32") = R.multiply(lv1, R.const(0.5, "float32"))
                lv3: R.Tensor((1, 30), dtype="float32") = R.add(R.const(0.5, "float32"), lv2)
                gv: R.Tensor((1, 30), dtype="float32") = R.multiply(x, lv3)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_swish():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.swish(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.sigmoid(x)
                gv: R.Tensor((1, 30), dtype="float32") = R.multiply(x, lv)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_fill():
    class TfInput(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 30), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            fill_out = tf.fill((1, 30), y)
            return x + fill_out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 30), dtype="float32"), y: R.Tensor((), dtype="float32")
        ) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.add(x, y)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.add, R.add),
        (tf.subtract, R.subtract),
        (tf.multiply, R.multiply),
        (tf.divide, R.divide),
        (tf.math.floormod, R.floor_mod),
        (tf.math.floordiv, R.floor_divide),
    ],
)
def test_binary(tf_op, relax_op):
    class Binary(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 2), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf_op(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="float32"), y: R.Tensor((2, 2), dtype="float32")
        ) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = relax_op(x, y)
                R.output(gv)
            return gv

    verify(Binary, Expected)


def test_pow():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.pow(x, 4)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.power(x, R.const(4.0, "float32"))
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def test_square():
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.square(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.nn.relu, R.nn.relu),
        (tf.nn.relu6, R.nn.relu6),
        (tf.math.floor, R.floor),
        (tf.math.ceil, R.ceil),
        (tf.math.tanh, R.tanh),
        (tf.math.sigmoid, R.sigmoid),
        (tf.math.abs, R.abs),
        (tf.math.cos, R.cos),
        (tf.math.sin, R.sin),
        (tf.math.exp, R.exp),
        (tf.math.log, R.log),
        (tf.math.negative, R.negative),
        (tf.round, R.round),
        (tf.math.rsqrt, R.rsqrt),
        (tf.nn.softmax, R.nn.softmax),
        (tf.math.sqrt, R.sqrt),
    ],
)
def test_element_wise(tf_op, relax_op):
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = relax_op(x)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.less, R.less),
        (tf.math.less_equal, R.less_equal),
        (tf.math.greater, R.greater),
        (tf.math.greater_equal, R.greater_equal),
        (tf.math.equal, R.equal),
        (tf.math.not_equal, R.not_equal),
    ],
)
def test_split_compare(tf_op, relax_op):
    class Compare(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf_op(a, b, name=None)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 15), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 15), dtype="float32"),
                    R.Tensor((1, 15), dtype="float32"),
                ) = R.split(x, indices_or_sections=2, axis=1)
                lv1: R.Tensor((1, 15), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 15), dtype="float32") = lv[1]
                gv: R.Tensor((1, 15), dtype="bool") = relax_op(lv1, lv2)
                R.output(gv)
            return gv

    verify(Compare, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.logical_not, R.logical_not),
    ],
)
def test_logical_unary(tf_op, relax_op):
    class Logical(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
            ]
        )
        def func(self, x):
            return tf_op(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 2), dtype="bool"),
        ) -> R.Tensor((2, 2), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="bool") = relax_op(x)
                R.output(gv)
            return gv

    verify(Logical, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.math.logical_or, R.logical_or),
        (tf.math.logical_and, R.logical_and),
    ],
)
def test_logical(tf_op, relax_op):
    class Logical(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 2), dtype=tf.bool),
            ]
        )
        def func(self, x, y):
            return tf_op(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2), dtype="bool"), y: R.Tensor((2, 2), dtype="bool")) -> R.Tensor(
            (2, 2), dtype="bool"
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="bool") = relax_op(x, y)
                R.output(gv)
            return gv

    verify(Logical, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.add, R.add),
        (tf.subtract, R.subtract),
        (tf.multiply, R.multiply),
        (tf.divide, R.divide),
        (tf.math.floormod, R.floor_mod),
        (tf.math.maximum, R.maximum),
        (tf.math.minimum, R.minimum),
    ],
)
def test_split_binary(tf_op, relax_op):
    class Binary(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf_op(a, b, name=None)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 15), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 15), dtype="float32"),
                    R.Tensor((1, 15), dtype="float32"),
                ) = R.split(x, indices_or_sections=2, axis=1)
                lv1: R.Tensor((1, 15), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 15), dtype="float32") = lv[1]
                gv: R.Tensor((1, 15), dtype="float32") = relax_op(lv1, lv2)
                R.output(gv)
            return gv

    verify(Binary, Expected)


@pytest.mark.parametrize(
    "tf_op, relax_op, axis, out_shape",
    [
        (tf.math.argmax, R.argmax, 0, (30,)),
        (tf.math.argmin, R.argmin, 1, (5,)),
    ],
)
def test_reduce(tf_op, relax_op, axis, out_shape):
    class TfInput(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(5, 30), dtype=tf.float32)])
        def func(self, x):
            return tf_op(x, axis=axis)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 30), dtype="float32")) -> R.Tensor(out_shape, dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor(out_shape, dtype="int64") = relax_op(x, axis=axis, keepdims=False)
                R.output(gv)
            return gv

    verify(TfInput, Expected)


def _make_conv2d_module(data_shape, kernel_shape, data_format, strides, padding):
    class Conv2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                tf.TensorSpec(shape=kernel_shape, dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.conv2d(
                input=data,
                filters=kernel,
                data_format=data_format,
                strides=strides,
                padding=padding,
            )

    return Conv2DModule


def test_conv2d_same():
    Conv2DModule = _make_conv2d_module(
        (1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
            kernel: R.Tensor((3, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 3, 3, 32), dtype="float32") = R.permute_dims(
                    kernel, axes=[3, 0, 1, 2]
                )
                lv1: R.Tensor((3, 3, 32, 32), dtype="float32") = R.permute_dims(
                    lv, axes=[1, 2, 3, 0]
                )
                lv2: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.conv2d(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                    out_dtype="void",
                )
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.add(
                    lv2, R.const(np.zeros((32,), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(Conv2DModule, Expected)


def test_conv2d_valid():
    Conv2DModule = _make_conv2d_module(
        (1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "VALID"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
            kernel: R.Tensor((3, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 126, 126, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 3, 3, 32), dtype="float32") = R.permute_dims(
                    kernel, axes=[3, 0, 1, 2]
                )
                lv1: R.Tensor((3, 3, 32, 32), dtype="float32") = R.permute_dims(
                    lv, axes=[1, 2, 3, 0]
                )
                lv2: R.Tensor((1, 126, 126, 32), dtype="float32") = R.nn.conv2d(
                    data,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                    out_layout="NHWC",
                    out_dtype="void",
                )
                gv: R.Tensor((1, 126, 126, 32), dtype="float32") = R.add(
                    lv2, R.const(np.zeros((32,), dtype="float32"))
                )
                R.output(gv)
            return gv

    verify(Conv2DModule, Expected)


def _make_pool2d_module(pool, data_shape, ksize, data_format, strides, padding):
    class Pool2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
            ]
        )
        def func(self, data):
            return pool(
                input=data,
                ksize=ksize,
                data_format=data_format,
                strides=strides,
                padding=padding,
            )

    return Pool2DModule


def test_avg_pool2d_same():
    Pool2DModule = _make_pool2d_module(
        tf.nn.avg_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.avg_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 1, 1],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


def test_avg_pool2d_valid():
    Pool2DModule = _make_pool2d_module(
        tf.nn.avg_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "VALID"
    )
    verify(Pool2DModule)


def test_max_pool2d_same():
    Pool2DModule = _make_pool2d_module(
        tf.nn.max_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "SAME"
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 128, 128, 32), dtype="float32"),
        ) -> R.Tensor((1, 128, 128, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 128, 128, 32), dtype="float32") = R.nn.max_pool2d(
                    data,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 1, 1],
                    ceil_mode=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                R.output(gv)
            return gv

    verify(Pool2DModule, Expected)


def test_max_pool2d_valid():
    Pool2DModule = _make_pool2d_module(
        tf.nn.max_pool2d, (1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "VALID"
    )
    verify(Pool2DModule)


@pytest.mark.parametrize(
    "net, shape",
    [
        # Limiting the tests for CI
        (keras_app.Xception, (1, 299, 299, 3)),
        # (keras_app.VGG16, (1, 224, 224, 3)),
        # (keras_app.VGG19, (1, 224, 224, 3)),
        (keras_app.ResNet50, (1, 224, 224, 3)),
        # (keras_app.ResNet50V2, (1, 224, 224, 3)),
        # (keras_app.ResNet101, (1, 224, 224, 3)),
        # (keras_app.ResNet101V2, (1, 224, 224, 3)),
        # (keras_app.ResNet152, (1, 224, 224, 3)),
        # (keras_app.ResNet152V2, (1, 224, 224, 3)),
        (keras_app.InceptionResNetV2, (1, 299, 299, 3)),
        # (keras_app.MobileNet, (1, 224, 224, 3)),
        (keras_app.MobileNetV2, (1, 224, 224, 3)),
        (keras_app.DenseNet121, (1, 224, 224, 3)),
        # (keras_app.DenseNet169, (1, 224, 224, 3)),
        # (keras_app.DenseNet201, (1, 224, 224, 3)),
        (keras_app.NASNetMobile, (1, 224, 224, 3)),
        # (keras_app.NASNetLarge, (1, 331, 331, 3)),
        (keras_app.EfficientNetB0, (1, 224, 224, 3)),
        # (keras_app.EfficientNetB1, (1, 240, 240, 3)),
        # (keras_app.EfficientNetB2, (1, 260, 260, 3)),
        # (keras_app.EfficientNetB3, (1, 300, 300, 3)),
        # (keras_app.EfficientNetB4, (1, 380, 380, 3)),
        # (keras_app.EfficientNetB5, (1, 456, 456, 3)),
        # (keras_app.EfficientNetB6, (1, 528, 528, 3)),
        # (keras_app.EfficientNetB7, (1, 600, 600, 3)),
        (keras_app.EfficientNetV2B0, (1, 224, 224, 3)),
        # (keras_app.EfficientNetV2B1, (1, 240, 240, 3)),
        # (keras_app.EfficientNetV2B2, (1, 260, 260, 3)),
        # (keras_app.EfficientNetV2B3, (1, 300, 300, 3)),
        # (keras_app.EfficientNetV2S, (1, 384, 384, 3)),
        # (keras_app.EfficientNetV2M, (1, 480, 480, 3)),
        # (keras_app.EfficientNetV2L, (1, 480, 480, 3)),
        # (keras_app.ConvNeXtTiny, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtSmall, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtBase, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtLarge, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtXLarge, (1, 224, 224, 3)),
    ],
)
def test_networks(net, shape):
    # Run network tests only in nightly builds
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    class NetworkModule(tf.Module):
        def __init__(self):
            self.model = net(weights=None, include_top=True)

        @tf.function
        def func(self, data):
            return self.model(data, training=False)

    model = NetworkModule()
    concrete_func = model.func.get_concrete_function(tf.TensorSpec(shape=shape, dtype=tf.float32))

    verify(concrete_func)


def test_batch_matmul():
    class BatchMatMul(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 4, 5), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.matmul(x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
            y: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tensor((2, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(x, y, out_dtype="void")
                gv: R.Tensor((2, 3, 5), dtype="float32") = R.reshape(lv, R.shape([2, 3, 5]))
                R.output(gv)
            return gv

    verify(BatchMatMul, Expected)


def test_batch_matmul_adj():
    class BatchMatMulAdj(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 4, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 5, 4), dtype=tf.float32),
            ]
        )
        def func(self, x, y):
            return tf.matmul(x, y, transpose_a=True, transpose_b=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 4, 3), dtype="float32"),
            y: R.Tensor((2, 5, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3, 4), dtype="float32") = R.permute_dims(x, axes=[0, 2, 1])
                lv1: R.Tensor((2, 4, 5), dtype="float32") = R.permute_dims(y, axes=[0, 2, 1])
                lv2: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(lv, lv1, out_dtype="void")
                gv: R.Tensor((2, 3, 5), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 5]))
                R.output(gv)
            return gv

    verify(BatchMatMulAdj, Expected)


def _verify_nms_v5(mod, tf_func, boxes_np, scores_np):
    """E2E verify for NMS: only run on nightly, compare valid outputs only."""
    if "CI_ENV_NIGHTLY" not in os.environ:
        return

    tf_indices, tf_scores, tf_valid = tf_func(
        tf.constant(boxes_np), tf.constant(scores_np)
    )
    n_valid = int(tf_valid.numpy())

    tgt = tvm.target.Target("llvm")
    ex = tvm.compile(mod, tgt)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", boxes_np, scores_np)
    vm.invoke_stateful("main")
    tvm_indices, tvm_scores, tvm_valid = vm.get_outputs("main")

    assert int(tvm_valid.numpy()) == n_valid
    np.testing.assert_array_equal(
        tf_indices.numpy()[:n_valid],
        tvm_indices.numpy()[:n_valid],
    )
    np.testing.assert_allclose(
        tf_scores.numpy()[:n_valid],
        tvm_scores.numpy()[:n_valid],
        rtol=1e-5,
        atol=1e-5,
    )


def _build_nms_v5_mod(num_boxes, max_output_size, iou_threshold, score_threshold):
    """Convert a NonMaxSuppressionV5 TFLite model to a Relax module.

    Scalar params must be Python literals (not tf.constant) so TFLite can
    statically infer output shapes during conversion.
    """

    class NMSv5Module(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(num_boxes, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(num_boxes,), dtype=tf.float32),
            ]
        )
        def func(self, boxes, scores):
            indices, out_scores, valid = tf.raw_ops.NonMaxSuppressionV5(
                boxes=boxes,
                scores=scores,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                soft_nms_sigma=0.0,
                pad_to_max_output_size=True,
            )
            return indices, out_scores, valid

    instance = NMSv5Module()
    cf = instance.func.get_concrete_function()
    mod = _get_mod_from_cfunc(cf)
    return mod, instance.func


def _make_valid_boxes(rng, n):
    """Generate n random boxes with y1<=y2, x1<=x2 using the given RNG."""
    raw = rng.random((n, 4), dtype=np.float32)
    return np.stack(
        [
            np.minimum(raw[:, 0], raw[:, 2]),  # y1
            np.minimum(raw[:, 1], raw[:, 3]),  # x1
            np.maximum(raw[:, 0], raw[:, 2]),  # y2
            np.maximum(raw[:, 1], raw[:, 3]),  # x2
        ],
        axis=1,
    ).astype(np.float32)


_NMS_V5_CASES = [
    pytest.param(
        6, 3, 0.5, 0.0,
        np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, 0.0, 1.0, 0.9],
            [0.5, 0.5, 1.5, 1.5],
            [0.0, 0.0, 0.3, 0.3],
        ], dtype=np.float32),
        np.array([0.9, 0.75, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
        id="basic",
    ),
    pytest.param(
        8, 4, 0.5, 0.4,
        _make_valid_boxes(np.random.default_rng(42), 8),
        np.random.default_rng(42).random(8, dtype=np.float32),
        id="score_threshold",
    ),
    pytest.param(
        5, 3, 0.5, 0.99,
        _make_valid_boxes(np.random.default_rng(0), 5),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        id="all_suppressed",
    ),
    pytest.param(
        6, 6, 0.1, 0.0,
        np.array([
            [0.0, 0.0, 0.4, 0.4],
            [0.5, 0.5, 0.9, 0.9],
            [0.1, 0.1, 0.5, 0.5],
            [0.6, 0.6, 1.0, 1.0],
            [0.0, 0.5, 0.4, 0.9],
            [0.5, 0.0, 0.9, 0.4],
        ], dtype=np.float32),
        np.array([0.9, 0.85, 0.7, 0.65, 0.6, 0.55], dtype=np.float32),
        id="iou_threshold",
    ),
    pytest.param(
        4, 10, 0.5, 0.0,
        np.array([
            [0.0, 0.0, 0.3, 0.3],
            [0.5, 0.5, 0.8, 0.8],
            [0.1, 0.1, 0.4, 0.4],
            [0.6, 0.6, 0.9, 0.9],
        ], dtype=np.float32),
        np.array([0.9, 0.85, 0.7, 0.65], dtype=np.float32),
        id="max_output_size_larger_than_boxes",
    ),
]


@pytest.mark.parametrize(
    "num_boxes,max_output_size,iou_threshold,score_threshold,boxes,scores",
    _NMS_V5_CASES,
)
def test_nms_v5(num_boxes, max_output_size, iou_threshold, score_threshold, boxes, scores):
    """NON_MAX_SUPPRESSION_V5: conversion smoke test + E2E correctness (nightly only)."""
    mod, tf_func = _build_nms_v5_mod(num_boxes, max_output_size, iou_threshold, score_threshold)
    _verify_nms_v5(mod, tf_func, boxes, scores)


def test_nms_v5_ir():
    """Verify the emitted Relax IR has correct structure for NON_MAX_SUPPRESSION_V5."""
    num_boxes = 6
    max_output_size = 3
    mod, _ = _build_nms_v5_mod(
        num_boxes=num_boxes,
        max_output_size=max_output_size,
        iou_threshold=0.5,
        score_threshold=0.0,
    )

    ir = mod.script()

    # Validate correct sorting/id indices are passed to valid_counts
    assert "score_index=0" in ir
    assert "id_index=-1" in ir
    # NMS size limit validation
    assert f"max_output_size={max_output_size}" in ir
    # Valid output shape must be () statically
    assert 'R.Tensor((), dtype="int32")' in ir
    # Bounding boxes / scores tensor bounds checks
    assert f"R.Tensor(({max_output_size},)" in ir


def _make_resize_expected(input_shape, output_size, method, coordinate_transformation_mode, rounding_method):
    """Build an Expected IRModule programmatically to avoid TVMScript variable scope limitations."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(
                relax.op.image.resize2d(
                    x,
                    size=relax.ShapeExpr([output_size[0], output_size[1]]),
                    roi=[0.0, 0.0, 0.0, 0.0],
                    layout="NHWC",
                    method=method,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    rounding_method=rounding_method,
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                    out_dtype="void",
                )
            )
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


@pytest.mark.parametrize(
    "input_shape, output_size, tf_op, coordinate_transformation_mode",
    [
        ((1, 4, 4, 1), [8, 8],   lambda x: tf.image.resize(x, [8, 8],   method="bilinear"),                                          "half_pixel"),
        ((1, 8, 8, 3), [4, 4],   lambda x: tf.image.resize(x, [4, 4],   method="bilinear"),                                          "half_pixel"),
        ((1, 4, 4, 1), [7, 7],   lambda x: tf.compat.v1.image.resize_bilinear(x, [7, 7], align_corners=True),                        "align_corners"),
        ((1, 4, 4, 2), [8, 8],   lambda x: tf.compat.v1.image.resize_bilinear(x, [8, 8], half_pixel_centers=True),                   "half_pixel"),
        ((2, 6, 6, 16), [12, 12], lambda x: tf.image.resize(x, [12, 12], method="bilinear"),                                         "half_pixel"),
        ((1, 5, 5, 3), [5, 5],   lambda x: tf.image.resize(x, [5, 5],   method="bilinear"),                                          "half_pixel"),
        ((1, 4, 8, 1), [8, 16],  lambda x: tf.image.resize(x, [8, 16],  method="bilinear"),                                          "half_pixel"),
    ],
)
def test_resize_bilinear(input_shape, output_size, tf_op, coordinate_transformation_mode):
    class ResizeBilinear(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(input_shape, output_size, "linear", coordinate_transformation_mode, "")
    verify(ResizeBilinear, expected)


@pytest.mark.parametrize(
    "input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method",
    [
        ((1, 2, 2, 1), [4, 4],   lambda x: tf.image.resize(x, [4, 4],   method="nearest"),                                "half_pixel",   "round_prefer_ceil"),
        ((1, 8, 8, 3), [4, 4],   lambda x: tf.image.resize(x, [4, 4],   method="nearest"),                                "half_pixel",   "round_prefer_ceil"),
        ((1, 4, 4, 1), [7, 7],   lambda x: tf.compat.v1.image.resize_nearest_neighbor(x, [7, 7], align_corners=True),     "align_corners", ""),
        ((4, 3, 3, 8), [6, 6],   lambda x: tf.image.resize(x, [6, 6],   method="nearest"),                                "half_pixel",   "round_prefer_ceil"),
        ((1, 4, 8, 1), [8, 16],  lambda x: tf.image.resize(x, [8, 16],  method="nearest"),                                "half_pixel",   "round_prefer_ceil"),
        ((1, 3, 3, 2), [3, 3],   lambda x: tf.image.resize(x, [3, 3],   method="nearest"),                                "half_pixel",   "round_prefer_ceil"),
    ],
)
def test_resize_nearest_neighbor(input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method):
    class ResizeNearest(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(input_shape, output_size, "nearest_neighbor", coordinate_transformation_mode, rounding_method)
    verify(ResizeNearest, expected)


def _make_reduce_expected(relax_op, input_shape, axes, keepdims, dtype):
    if axes is None:
        axes = list(range(len(input_shape)))
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(input_shape, dtype))
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(relax_op(x, axis=axes, keepdims=keepdims))
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


@pytest.mark.parametrize(
    "tf_op, relax_op",
    [
        (tf.reduce_sum, relax.op.sum),
        (tf.reduce_mean, relax.op.mean),
        (tf.reduce_max, relax.op.max),
        (tf.reduce_min, relax.op.min),
        (tf.reduce_prod, relax.op.prod),
    ],
)
@pytest.mark.parametrize(
    "input_shape, axes",
    [
        ((1, 8, 8, 3), 1),
        ((1, 8, 8, 3), [1, 2]),
        ((1, 8, 8, 3), -1),
        ((1, 8, 8, 3), None),
        ((30,), 0),
        ((2, 5, 2), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype", [tf.float32, tf.int32])
def test_reduction_ops(tf_op, relax_op, input_shape, axes, keepdims, dtype):
    class ReduceModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=dtype)])
        def func(self, x):
            return tf_op(x, axis=axes, keepdims=keepdims)

    relax_dtype = "float32" if dtype == tf.float32 else "int32"
    expected = _make_reduce_expected(relax_op, input_shape, axes, keepdims, relax_dtype)
    verify(ReduceModule, expected)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
