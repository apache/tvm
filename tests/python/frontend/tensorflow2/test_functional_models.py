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
"""TF2 to relay converter test: tests basic examples"""

import tempfile
import tensorflow as tf
import numpy as np
import pytest
from common import compare_tf_tvm
from common import run_tf_code


class AddOne(tf.Module):
    """ simple function to test x=x+1; scalar as input"""

    def get_input(self):
        return np.array(1.0, dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def func(self, x):
        return x + 1


class AddOne2D(AddOne):
    """2D array as input"""

    def get_input(self):
        return np.ones((2, 2), dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x + 1


class AddOne2DConstant(AddOne):
    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    def get_input(self):
        return np.ones((2, 2), dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x + np.ones((2, 2), dtype="float32")


class SubOne2DConstant(tf.Module):
    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    def get_input(self):
        return np.ones((2, 2), dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x - np.ones((2, 2), dtype="float32")


class MulOne2DConstant(tf.Module):
    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    def get_input(self):
        return np.ones((2, 2), dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x * np.ones((2, 2), dtype="float32")


class DivOne2DConstant(tf.Module):
    """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

    def get_input(self):
        return np.ones((2, 2), dtype="float32")

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def func(self, x):
        return x / np.ones((2, 2), dtype="float32")


class StridedSlice(tf.Module):
    def get_input(self):
        return np.ones((3, 2, 3), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(3, 2, 3), dtype=tf.float32)])
    def func(self, x):
        return tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])


class Shape(tf.Module):
    def get_input(self):
        return np.ones((3, 2, 3), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(3, 2, 3), dtype=tf.float32)])
    def func(self, x):
        a = tf.ones_like(tf.raw_ops.Shape(input=x), dtype=tf.float32)
        return a + x


class Pack(tf.Module):
    def get_input(self):
        return np.ones((2, 3), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
    def func(self, x):
        return tf.raw_ops.Pack(values=[x, x], axis=0)


class Split(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b, c = tf.split(x, 3, axis=1)
        return tf.raw_ops.Pack(values=[a, b, c], axis=1)


class Maximum(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b = tf.split(x, 2, axis=1)
        return tf.math.maximum(a, b, name=None)


class Less(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b = tf.split(x, 2, axis=1)
        return tf.math.less(a, b, name=None)


class Equal(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b = tf.split(x, 2, axis=1)
        return tf.math.equal(a, b, name=None)


class Cast(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.cast(x, tf.int32)


class ExpandDims(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.expand_dims(x, axis=2)


class Transpose(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        x = tf.expand_dims(x, axis=2)
        return tf.transpose(x, perm=[0, 2, 1])


class Reshape(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.reshape(x, (1, 2, 15))


class Tanh(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.tanh(x)


class Sigmoid(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.sigmoid(x)


class Relu(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.nn.relu(x)


class Floor(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        return tf.math.floor(x)


class FloorMod(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b = tf.split(x, 2, axis=1)
        return tf.math.floormod(a, b)


class ConcatV2(tf.Module):
    def get_input(self):
        return np.ones((1, 30), dtype=np.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
    def func(self, x):
        a, b, c = tf.split(x, 3, axis=1)
        return tf.raw_ops.ConcatV2(values=[a, b, c], axis=1)


def _function_graph(TestClass):
    f = TestClass().func
    gdef = f.get_concrete_function().graph.as_graph_def()
    gdef_ops = list(set([n.op for n in gdef.node]))
    input_ = TestClass().get_input()
    output = run_tf_code(f, input_)
    return gdef, input_, output


def _model_graph(TestClass):
    model = TestClass()
    with tempfile.TemporaryDirectory() as model_path:
        tf.saved_model.save(model, model_path)
        imported = tf.saved_model.load(model_path)

    f = imported.signatures["serving_default"]
    gdef = f.graph.as_graph_def(add_shapes=True)

    input_ = model.get_input()
    output = run_tf_code(f, input_)
    return gdef, input_, output


def run_func_graph(TestClass, use_vm=False):
    compare_tf_tvm(*_function_graph(TestClass), vm=use_vm)


def run_model_graph(TestClass, output_sig=None):
    compare_tf_tvm(*_model_graph(TestClass), vm=True, output_sig=output_sig)


def run_all(TestClass):
    run_model_graph(TestClass)
    for use_vm in [True, False]:
        run_func_graph(TestClass, use_vm=use_vm)


def test_basic_ops():
    run_all(AddOne)
    run_all(AddOne2D)
    run_all(AddOne2DConstant)
    run_all(SubOne2DConstant)
    run_all(MulOne2DConstant)
    run_all(DivOne2DConstant)


def test_strided_slice():
    run_all(StridedSlice)


def test_shape():
    run_all(Shape)


def test_pack():
    run_all(Pack)


def test_split():
    run_all(Split)


def test_max():
    run_all(Maximum)


def test_less():
    run_all(Less)


def test_equal():
    run_all(Equal)


def test_floor():
    run_all(Floor)
    run_all(FloorMod)


def test_concat_v2():
    run_all(ConcatV2)


def test_cast():
    run_all(Cast)


def test_expand_dims():
    run_all(ExpandDims)


def test_transpose():
    run_all(Transpose)


def test_reshape():
    run_all(Reshape)


def test_tanh():
    run_all(Tanh)


def test_sigmoid():
    run_all(Sigmoid)


def test_relu():
    run_all(Relu)


if __name__ == "__main__":
    pytest.main([__file__])
