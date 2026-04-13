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
import tvm.relax.frontend.tflite.tflite_frontend as tflite_frontend
from tvm import relax
from tvm.relax.frontend.tflite import from_tflite
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tirx as T


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


@pytest.mark.parametrize(
    "input_shape, out_type",
    [
        ((2, 3, 4), tf.int32),
        ((5,), tf.int64),
        ((1, 1, 1, 1), tf.int32),
        ((), tf.int32),
        ((0, 3), tf.int64),
    ],
)
def test_shape(input_shape, out_type):
    """SHAPE conversion for static-rank non-quantized tensors."""

    class Shape(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf.shape(x, out_type=out_type)

    verify(Shape)


def test_shape_dynamic_dim():
    """SHAPE conversion with a dynamic input dimension."""

    class ShapeDynamic(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.shape(x, out_type=tf.int32)

    verify(ShapeDynamic)


@pytest.mark.parametrize(
    "start, limit, delta, dtype",
    [
        (0, 8, 2, tf.int32),
        (1, 9, 2, tf.int64),
        (0.0, 1.0, 0.2, tf.float32),
        (8, 0, -2, tf.int32),
        (0, 0, 1, tf.int32),
        (0, 7, 2, tf.int32),
        (0.0, -1.0, -0.25, tf.float32),
    ],
)
def test_range(start, limit, delta, dtype):
    """RANGE conversion with non-quantized constant scalar bounds."""

    class Range(tf.Module):
        @tf.function(input_signature=[])
        def func(self):
            return tf.range(start, limit, delta, dtype=dtype)

    verify(Range)


def test_range_dynamic_scalar_inputs_not_supported():
    """RANGE conversion currently rejects dynamic scalar inputs."""

    class RangeDynamic(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
        )
        def func(self, start, limit, delta):
            return tf.range(start, limit, delta, dtype=tf.int32)

    with pytest.raises(tvm.error.OpNotImplemented, match="dynamic scalar inputs"):
        verify(RangeDynamic)


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
        (tf.nn.log_softmax, R.nn.log_softmax),
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


def test_fully_connected():
    class FullyConnected(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 8), dtype=tf.float32)])
        def func(self, x):
            weight = tf.constant(np.arange(24, dtype=np.float32).reshape((3, 8)))
            bias = tf.constant(np.array([0.5, 1.0, -1.0], dtype=np.float32))
            out = tf.matmul(x, weight, transpose_b=True)
            return tf.nn.bias_add(out, bias)

    verify(FullyConnected)


def test_depthwise_conv2d():
    class DepthwiseConv2D(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(3, 3, 2, 1), dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            return tf.nn.depthwise_conv2d(
                input=data,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )

    verify(DepthwiseConv2D)


def test_transpose_conv():
    class TransposeConv(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(3, 3, 3, 2), dtype=tf.float32),
            ]
        )
        def func(self, data, kernel):
            output_shape = tf.constant([1, 8, 8, 3], dtype=tf.int32)
            return tf.nn.conv2d_transpose(
                input=data,
                filters=kernel,
                output_shape=output_shape,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )

    verify(TransposeConv)

def test_l2_pool2d():
    class L2Pool2D(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 8, 8, 2), dtype=tf.float32)])
        def func(self, data):
            squared = tf.math.square(data)
            pooled = tf.nn.avg_pool2d(squared, ksize=[2, 2], strides=[1, 1], padding="SAME")
            return tf.math.sqrt(pooled)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 2), dtype="float32")
        ) -> R.Tensor((1, 8, 8, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                squared = R.power(data, R.const(2.0, "float32"))
                pooled = R.nn.avg_pool2d(
                    squared,
                    pool_size=[2, 2],
                    strides=[1, 1],
                    padding=[0, 0, 1, 1],
                    layout="NHWC",
                )
                gv = R.sqrt(pooled)
                R.output(gv)
            return gv

    verify(L2Pool2D, Expected)


def test_l2_normalization():
    class L2Normalization(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.l2_normalize(x, axis=-1)

    verify(L2Normalization)


def test_slice():
    class Slice(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.slice(x, begin=[1, 1], size=[2, 2])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((2, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 2), dtype="float32") = R.strided_slice(
                    x, axes=[0, 1], begin=[1, 1], end=[3, 3]
                )
                R.output(gv)
            return gv

    verify(Slice, Expected)


def test_reverse_v2():
    class ReverseV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.reverse(x, axis=[1])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.flip(x, axis=1)
                R.output(gv)
            return gv

    verify(ReverseV2, Expected)

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

    tf_indices, tf_scores, tf_valid = tf_func(tf.constant(boxes_np), tf.constant(scores_np))
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


class _StubDetectionPostprocessTensor:
    def __init__(self, shape, name):
        self._shape = list(shape)
        self._name = name

    def Shape(self, index):
        return self._shape[index]

    def Name(self):
        return self._name

    def Type(self):
        return 0


class _StubDetectionPostprocessOp:
    def __init__(self, custom_options):
        self._custom_options = _encode_detection_postprocess_custom_options(custom_options)

    def CustomOptionsAsNumpy(self):
        return np.frombuffer(self._custom_options, dtype="uint8")


_DETECTION_POSTPROCESS_ANCHORS = np.array(
    [
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.2, 1.0, 1.0],
        [0.1, 0.1, 0.5, 0.5],
        [0.8, 0.8, 0.2, 0.2],
    ],
    dtype="float32",
)


def _encode_detection_postprocess_custom_options(custom_options):
    from flatbuffers import flexbuffers

    builder = flexbuffers.Builder()
    with builder.Map():
        for key, value in custom_options.items():
            if isinstance(value, bool):
                builder.Bool(key, value)
            elif isinstance(value, int):
                builder.Int(key, value)
            else:
                builder.Float(key, float(value))
    return bytes(builder.Finish())


def _make_detection_postprocess_tensor_wrapper(tensor_idx, shape, name):
    return tflite_frontend.TensorWrapper(
        tensor_idx,
        _StubDetectionPostprocessTensor(shape, name),
        None,
    )


def _build_detection_postprocess_mod(
    *,
    num_classes=1,
    max_detections=4,
    detections_per_class=4,
    use_regular_nms=False,
    nms_iou_threshold=0.5,
    nms_score_threshold=0.3,
    x_scale=10.0,
    y_scale=10.0,
    w_scale=5.0,
    h_scale=5.0,
    batch_size=2,
    num_anchors=4,
    input_num_classes=None,
):
    custom_options = {
        "num_classes": num_classes,
        "max_detections": max_detections,
        "detections_per_class": detections_per_class,
        "nms_iou_threshold": nms_iou_threshold,
        "nms_score_threshold": nms_score_threshold,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "w_scale": w_scale,
        "h_scale": h_scale,
        "use_regular_nms": use_regular_nms,
    }
    return _convert_detection_postprocess_with_options(
        custom_options,
        batch_size=batch_size,
        num_anchors=num_anchors,
        num_classes=num_classes,
        input_num_classes=input_num_classes,
    )


def _convert_detection_postprocess_with_options(
    custom_options,
    *,
    batch_size=2,
    num_anchors=4,
    num_classes=1,
    input_num_classes=None,
    build_module=True,
):
    input_num_classes = num_classes if input_num_classes is None else input_num_classes
    loc = relax.Var("loc", relax.TensorStructInfo((batch_size, num_anchors, 4), "float32"))
    cls = relax.Var(
        "cls", relax.TensorStructInfo((batch_size, num_anchors, input_num_classes), "float32")
    )
    inputs = [
        _make_detection_postprocess_tensor_wrapper(0, (batch_size, num_anchors, 4), "loc"),
        _make_detection_postprocess_tensor_wrapper(
            1, (batch_size, num_anchors, input_num_classes), "cls"
        ),
        _make_detection_postprocess_tensor_wrapper(2, (num_anchors, 4), "anchors"),
    ]
    converter = tflite_frontend.OperatorConverter.__new__(tflite_frontend.OperatorConverter)
    converter.bb = relax.BlockBuilder()
    converter.exp_tab = tflite_frontend.ExprTable()
    converter.get_input_tensors = lambda op: inputs
    converter.get_expr = lambda tensor_idx: {0: loc, 1: cls}[tensor_idx]
    converter.get_tensor_value = (
        lambda tensor: _DETECTION_POSTPROCESS_ANCHORS if tensor.tensor_idx == 2 else None
    )
    converter.get_tensor_type_str = lambda tensor_type: "float32"
    op = _StubDetectionPostprocessOp(custom_options)
    if not build_module:
        return converter.convert_detection_postprocess(op)
    bb = converter.bb
    with bb.function("main", [loc, cls]):
        with bb.dataflow():
            output = converter.convert_detection_postprocess(op)
            gv = bb.emit_output(output)
        bb.emit_func_output(gv)
    return bb.get()


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
        6,
        3,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.0, 1.0, 0.9],
                [0.5, 0.5, 1.5, 1.5],
                [0.0, 0.0, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.75, 0.6, 0.5, 0.4, 0.3], dtype=np.float32),
        id="basic",
    ),
    pytest.param(
        8,
        4,
        0.5,
        0.4,
        _make_valid_boxes(np.random.default_rng(42), 8),
        np.random.default_rng(42).random(8, dtype=np.float32),
        id="score_threshold",
    ),
    pytest.param(
        5,
        3,
        0.5,
        0.99,
        _make_valid_boxes(np.random.default_rng(0), 5),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        id="all_suppressed",
    ),
    pytest.param(
        6,
        6,
        0.1,
        0.0,
        np.array(
            [
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.9, 0.9],
                [0.1, 0.1, 0.5, 0.5],
                [0.6, 0.6, 1.0, 1.0],
                [0.0, 0.5, 0.4, 0.9],
                [0.5, 0.0, 0.9, 0.4],
            ],
            dtype=np.float32,
        ),
        np.array([0.9, 0.85, 0.7, 0.65, 0.6, 0.55], dtype=np.float32),
        id="iou_threshold",
    ),
    pytest.param(
        4,
        10,
        0.5,
        0.0,
        np.array(
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.5, 0.5, 0.8, 0.8],
                [0.1, 0.1, 0.4, 0.4],
                [0.6, 0.6, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
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


_DETECTION_POSTPROCESS_SMOKE_CASES = [
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 2,
            "detections_per_class": 2,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        2,
        False,
        id="basic_fast_nms",
    ),
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 3,
            "detections_per_class": 2,
            "use_regular_nms": True,
            "nms_iou_threshold": 0.45,
            "nms_score_threshold": 0.25,
            "batch_size": 2,
            "num_anchors": 4,
        },
        1,
        True,
        id="regular_nms_multi_batch",
    ),
]


_DETECTION_POSTPROCESS_SHAPE_CASES = [
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 5,
            "max_detections": 2,
            "detections_per_class": 2,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        id="wider_input_classes",
    ),
    pytest.param(
        {
            "num_classes": 2,
            "input_num_classes": 3,
            "max_detections": 4,
            "detections_per_class": 4,
            "use_regular_nms": False,
            "nms_iou_threshold": 0.5,
            "nms_score_threshold": 0.5,
            "batch_size": 1,
            "num_anchors": 4,
        },
        id="larger_max_detections",
    ),
]


@pytest.mark.parametrize(
    "build_kwargs,expected_topk_count,expected_keep_background",
    _DETECTION_POSTPROCESS_SMOKE_CASES,
)
def test_detection_postprocess_smoke(
    build_kwargs, expected_topk_count, expected_keep_background
):
    mod = _build_detection_postprocess_mod(**build_kwargs)
    ir = mod.script()

    assert "R.vision.multibox_transform_loc" in ir
    assert "R.vision.all_class_non_max_suppression" in ir
    assert 'output_format="tensorflow"' in ir
    assert "R.where" in ir
    assert "R.gather_elements" in ir
    assert "R.gather_nd" in ir
    assert ir.count("R.topk(") == expected_topk_count
    assert f"keep_background={expected_keep_background}" in ir
    expected_batch = build_kwargs["batch_size"]
    expected_max_detections = build_kwargs["max_detections"]
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((expected_batch, expected_max_detections, 4), "float32"),
                relax.TensorStructInfo((expected_batch, expected_max_detections), "float32"),
                relax.TensorStructInfo((expected_batch, expected_max_detections), "float32"),
                relax.TensorStructInfo((expected_batch,), "float32"),
            ]
        ),
    )

    legalized = relax.transform.LegalizeOps()(mod)
    legalized_ir = legalized.script()
    assert "R.vision.all_class_non_max_suppression(" not in legalized_ir
    assert "R.call_tir(" in legalized_ir
    tvm.ir.assert_structural_equal(legalized["main"].ret_struct_info, mod["main"].ret_struct_info)


@pytest.mark.parametrize("build_kwargs", _DETECTION_POSTPROCESS_SHAPE_CASES)
def test_detection_postprocess_shape_variations(build_kwargs):
    mod = _build_detection_postprocess_mod(**build_kwargs)
    batch_size = build_kwargs["batch_size"]
    num_anchors = build_kwargs["num_anchors"]
    input_num_classes = build_kwargs["input_num_classes"]
    max_detections = build_kwargs["max_detections"]

    tvm.ir.assert_structural_equal(
        mod["main"].params[1].struct_info,
        relax.TensorStructInfo((batch_size, num_anchors, input_num_classes), "float32"),
    )
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size, max_detections, 4), "float32"),
                relax.TensorStructInfo((batch_size, max_detections), "float32"),
                relax.TensorStructInfo((batch_size, max_detections), "float32"),
                relax.TensorStructInfo((batch_size,), "float32"),
            ]
        ),
    )

def _make_resize_expected(
    input_shape, output_size, method, coordinate_transformation_mode, rounding_method
):
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
        (
            (1, 4, 4, 1),
            [8, 8],
            lambda x: tf.image.resize(x, [8, 8], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 8, 8, 3),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 4, 4, 1),
            [7, 7],
            lambda x: tf.compat.v1.image.resize_bilinear(x, [7, 7], align_corners=True),
            "align_corners",
        ),
        (
            (1, 4, 4, 2),
            [8, 8],
            lambda x: tf.compat.v1.image.resize_bilinear(x, [8, 8], half_pixel_centers=True),
            "half_pixel",
        ),
        (
            (2, 6, 6, 16),
            [12, 12],
            lambda x: tf.image.resize(x, [12, 12], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 5, 5, 3),
            [5, 5],
            lambda x: tf.image.resize(x, [5, 5], method="bilinear"),
            "half_pixel",
        ),
        (
            (1, 4, 8, 1),
            [8, 16],
            lambda x: tf.image.resize(x, [8, 16], method="bilinear"),
            "half_pixel",
        ),
    ],
)
def test_resize_bilinear(input_shape, output_size, tf_op, coordinate_transformation_mode):
    class ResizeBilinear(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(
        input_shape, output_size, "linear", coordinate_transformation_mode, ""
    )
    verify(ResizeBilinear, expected)


@pytest.mark.parametrize(
    "input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method",
    [
        (
            (1, 2, 2, 1),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 8, 8, 3),
            [4, 4],
            lambda x: tf.image.resize(x, [4, 4], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 4, 4, 1),
            [7, 7],
            lambda x: tf.compat.v1.image.resize_nearest_neighbor(x, [7, 7], align_corners=True),
            "align_corners",
            "",
        ),
        (
            (4, 3, 3, 8),
            [6, 6],
            lambda x: tf.image.resize(x, [6, 6], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 4, 8, 1),
            [8, 16],
            lambda x: tf.image.resize(x, [8, 16], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
        (
            (1, 3, 3, 2),
            [3, 3],
            lambda x: tf.image.resize(x, [3, 3], method="nearest"),
            "half_pixel",
            "round_prefer_ceil",
        ),
    ],
)
def test_resize_nearest_neighbor(
    input_shape, output_size, tf_op, coordinate_transformation_mode, rounding_method
):
    class ResizeNearest(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def func(self, x):
            return tf_op(x)

    expected = _make_resize_expected(
        input_shape,
        output_size,
        "nearest_neighbor",
        coordinate_transformation_mode,
        rounding_method,
    )
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


def test_pad():
    class Pad(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]])

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((4, 7), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 7), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=0.0, pad_mode="constant"
                )
                R.output(gv)
            return gv

    verify(Pad, Expected)


def test_pad_v2():
    class PadV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]], constant_values=5.0)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((4, 7), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 7), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=5.0, pad_mode="constant"
                )
                R.output(gv)
            return gv

    verify(PadV2, Expected)


def test_mirror_pad():
    class MirrorPad(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 4), dtype=tf.float32)])
        def func(self, x):
            return tf.pad(x, [[1, 1], [2, 2]], mode="REFLECT")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((5, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((5, 8), dtype="float32") = R.nn.pad(
                    x, pad_width=[1, 1, 2, 2], pad_value=0.0, pad_mode="reflect"
                )
                R.output(gv)
            return gv

    verify(MirrorPad, Expected)


def test_topk_v2():
    class TopKV2(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(5,), dtype=tf.float32)])
        def func(self, x):
            return tf.math.top_k(x, k=3).values

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5,), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="int32")) = (
                    R.topk(x, k=3, axis=-1, ret_type="both", largest=True, dtype="int32")
                )
                gv: R.Tensor((3,), dtype="float32") = lv[0]
                R.output(gv)
            return gv

    verify(TopKV2, Expected)


def test_one_hot():
    class OneHot(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.int32)])
        def func(self, x):
            return tf.one_hot(x, depth=4)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="int32")) -> R.Tensor((3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 4), dtype="float32") = R.one_hot(
                    x,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=4,
                    axis=-1,
                )
                R.output(gv)
            return gv

    verify(OneHot, Expected)


def test_select():
    class Select(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(2, 3), dtype=tf.bool),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
            ]
        )
        def func(self, cond, x, y):
            return tf.where(cond, x, y)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond: R.Tensor((2, 3), dtype="bool"),
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.where(cond, x, y)
                R.output(gv)
            return gv

    verify(Select, Expected)


def test_depth_to_space():
    class DepthToSpace(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 2, 4, 8), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.depth_to_space(x, block_size=2)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2, 4, 8), dtype="float32"),
        ) -> R.Tensor((1, 4, 8, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 4, 2, 2, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 4, 2, 2, 2])
                )
                lv1: R.Tensor((1, 2, 2, 4, 2, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 3, 2, 4, 5]
                )
                gv: R.Tensor((1, 4, 8, 2), dtype="float32") = R.reshape(lv1, R.shape([1, 4, 8, 2]))
                R.output(gv)
            return gv

    verify(DepthToSpace, Expected)


def test_space_to_depth():
    class SpaceToDepth(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 4, 4, 2), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.space_to_depth(x, block_size=2)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4, 4, 2), dtype="float32"),
        ) -> R.Tensor((1, 2, 2, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 2, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 2, 2])
                )
                lv1: R.Tensor((1, 2, 2, 2, 2, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 3, 2, 4, 5]
                )
                gv: R.Tensor((1, 2, 2, 8), dtype="float32") = R.reshape(lv1, R.shape([1, 2, 2, 8]))
                R.output(gv)
            return gv

    verify(SpaceToDepth, Expected)


def test_leaky_relu():
    class LeakyReLU(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.leaky_relu(x, alpha=0.2)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.nn.leakyrelu(
                    x, alpha=0.20000000298023224
                )
                R.output(gv)
            return gv

    verify(LeakyReLU, Expected)


def test_hard_swish():
    class HardSwish(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return x * tf.nn.relu6(x + 3) / 6

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 30), dtype="float32") = R.add(x, R.const(3.0, dtype="float32"))
                lv1: R.Tensor((1, 30), dtype="float32") = R.clip(
                    lv, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(6.0))
                )
                lv2: R.Tensor((1, 30), dtype="float32") = R.multiply(x, lv1)
                gv: R.Tensor((1, 30), dtype="float32") = R.divide(
                    lv2, R.const(6.0, dtype="float32")
                )
                R.output(gv)
            return gv

    verify(HardSwish, Expected)


def test_relu_n1_to_1():
    class ReLU_N1_to_1(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.clip_by_value(x, -1.0, 1.0)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 30), dtype="float32")) -> R.Tensor((1, 30), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 30), dtype="float32") = R.clip(x, min=-1, max=1)
                R.output(gv)
            return gv

    verify(ReLU_N1_to_1, Expected)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
