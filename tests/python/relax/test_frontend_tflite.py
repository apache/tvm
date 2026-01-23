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

import pytest
import tempfile
import tensorflow as tf
import numpy as np
import tflite.Model
import tvm
from tvm import relax
from tvm.script.parser import ir as I, relax as R, tir as T

from tvm.relax.frontend.tflite import from_tflite
from tf.keras import applications as keras_app

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def _get_mod_from_cfunc(cfunc):
    # print(cfunc.graph.as_graph_def())
    # for op in cfunc.graph.get_operations():
    #    if op.outputs:
    #        print(f"Op: {op.name}, Output Shape: {op.outputs[0].shape}")

    converter = tf.lite.TFLiteConverter.from_concrete_functions([cfunc])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = tflite.Model.Model.GetRootAsModel(converter.convert(), 0)
    mod = from_tflite(tflite_model)
    mod["main"] = mod["main"].without_attr("params")
    return mod


def verify(TestClass, expected=None):
    if isinstance(TestClass, type):
        cf = TestClass().func.get_concrete_function()
    else:
        cf = TestClass
    mod = _get_mod_from_cfunc(cf)

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

    if expected:
        tvm.ir.assert_structural_equal(mod, expected)


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
                lv: R.Tensor((1, 30), dtype="float32") = R.full(R.shape([1, 30]), y, dtype="void")
                gv: R.Tensor((1, 30), dtype="float32") = R.add(x, lv)
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
        def main(
            x: R.Tensor((2, 2), dtype="bool"), y: R.Tensor((2, 2), dtype="bool")
        ) -> R.Tensor((2, 2), dtype="bool"):
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


@pytest.mark.parametrize(
    "data, kernel, data_format, strides, padding",
    [
        ((1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "SAME"),
        ((1, 128, 128, 32), (3, 3, 32, 32), "NHWC", (1, 1, 1, 1), "VALID"),
        ((1, 32, 128, 128), (3, 3, 32, 32), "NCHW", (1, 1, 1, 1), "SAME"),
        ((1, 32, 128, 128), (3, 3, 32, 32), "NCHW", (1, 1, 1, 1), "VALID"),
    ],
)
def test_conv2d(data, kernel, data_format, strides, padding):
    class Conv2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data, dtype=tf.float32),
                tf.TensorSpec(shape=kernel, dtype=tf.float32),
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

    verify(Conv2DModule)


@pytest.mark.parametrize(
    "pool",
    [tf.nn.avg_pool2d, tf.nn.max_pool2d],
)
@pytest.mark.parametrize(
    "data, kernel, data_format, strides, padding",
    [
        ((1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "SAME"),
        ((1, 128, 128, 32), (2, 2), "NHWC", (1, 1, 1, 1), "VALID"),
        ((1, 32, 128, 128), (3, 3), "NCHW", (1, 1, 1, 1), "SAME"),
        ((1, 32, 128, 128), (3, 3), "NCHW", (1, 1, 1, 1), "VALID"),
    ],
)
def test_pool_2d(pool, data, kernel, data_format, strides, padding):
    class Pool2DModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=data, dtype=tf.float32),
            ]
        )
        def func(self, data):
            return pool(
                input=data,
                ksize=kernel,
                data_format=data_format,
                strides=strides,
                padding=padding,
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
        (keras_app.ConvNeXtTiny, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtSmall, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtBase, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtLarge, (1, 224, 224, 3)),
        # (keras_app.ConvNeXtXLarge, (1, 224, 224, 3)),
    ],
)
def test_networks(net, shape):
    class NetworkModule(tf.Module):
        def __init__(self):
            self.model = net(weights="imagenet", include_top=True)

        @tf.function
        def func(self, data):
            return self.model(data, training=False)

    model = NetworkModule()
    concrete_func = model.func.get_concrete_function(tf.TensorSpec(shape=shape, dtype=tf.float32))

    verify(concrete_func)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
