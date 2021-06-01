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


def run_func_graph(TestClass, runtime="vm"):
    compare_tf_tvm(*_function_graph(TestClass), runtime=runtime)


def run_model_graph(TestClass):
    compare_tf_tvm(*_model_graph(TestClass), runtime="vm")


def run_all(TestClass):
    run_model_graph(TestClass)
    for runtime_ in ["vm", "graph"]:
        run_func_graph(TestClass, runtime=runtime_)


def test_add_one():
    class AddOne(tf.Module):
        """simple function to test x=x+1; scalar as input"""

        def get_input(self):
            return np.array(1.0, dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
        def func(self, x):
            return x + 1

    run_all(AddOne)


def test_add_one_2d():
    class AddOne2D(tf.Module):
        """2D array as input"""

        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x + 1

    run_all(AddOne2D)


def test_add_one_2d_constant():
    class AddOne2DConstant(tf.Module):
        """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x + np.ones((2, 2), dtype="float32")

    run_all(AddOne2DConstant)


def test_sub_one_2d_constant():
    class SubOne2DConstant(tf.Module):
        """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x - np.ones((2, 2), dtype="float32")

    run_all(SubOne2DConstant)


def test_mul_one_2d_constant():
    class MulOne2DConstant(tf.Module):
        """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x * np.ones((2, 2), dtype="float32")

    run_all(MulOne2DConstant)


def test_div_one_2d_constant():
    class DivOne2DConstant(tf.Module):
        """2D array as input with 2D constant as well; 2D constant stored in params after convert"""

        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            return x / np.ones((2, 2), dtype="float32")

    run_all(DivOne2DConstant)


def test_strided_slice():
    class StridedSlice(tf.Module):
        def get_input(self):
            return np.ones((3, 2, 3), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])

    run_all(StridedSlice)


def test_split():
    class Split(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b, c = tf.split(x, 3, axis=1)
            return tf.raw_ops.Pack(values=[a, b, c], axis=1)

    run_all(Split)


def test_shape():
    class Shape(tf.Module):
        def get_input(self):
            return np.ones((3, 2, 3), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(3, 2, 3), dtype=tf.float32)])
        def func(self, x):
            a = tf.ones_like(tf.raw_ops.Shape(input=x), dtype=tf.float32)
            return a + x

    run_all(Shape)


def test_pack():
    class Pack(tf.Module):
        def get_input(self):
            return np.ones((2, 3), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 3), dtype=tf.float32)])
        def func(self, x):
            return tf.raw_ops.Pack(values=[x, x], axis=0)

    run_all(Pack)


def test_max():
    class Maximum(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf.math.maximum(a, b, name=None)

    run_all(Maximum)


def test_less():
    class Less(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf.math.less(a, b, name=None)

    run_all(Less)


def test_equal():
    class Equal(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf.math.equal(a, b, name=None)

    run_all(Equal)


def test_cast():
    class Cast(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.cast(x, tf.int32)

    run_all(Cast)


def test_expand_dims():
    class ExpandDims(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.expand_dims(x, axis=2)

    run_all(ExpandDims)


def test_transpose():
    class Transpose(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            x = tf.expand_dims(x, axis=2)
            return tf.transpose(x, perm=[0, 2, 1])

    run_all(Transpose)


def test_reshape():
    class Reshape(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.reshape(x, (1, 2, 15))

    run_all(Reshape)


def test_tanh():
    class Tanh(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.tanh(x)

    run_all(Tanh)


def test_sigmoid():
    class Sigmoid(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.sigmoid(x)

    run_all(Sigmoid)


def test_relu():
    class Relu(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.nn.relu(x)

    run_all(Relu)


def test_floor():
    class Floor(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            return tf.math.floor(x)

    run_all(Floor)


def test_floor_mod():
    class FloorMod(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b = tf.split(x, 2, axis=1)
            return tf.math.floormod(a, b)

    run_all(FloorMod)


def test_concat_v2():
    class ConcatV2(tf.Module):
        def get_input(self):
            return np.ones((1, 30), dtype=np.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 30), dtype=tf.float32)])
        def func(self, x):
            a, b, c = tf.split(x, 3, axis=1)
            return tf.raw_ops.ConcatV2(values=[a, b, c], axis=1)

    run_all(ConcatV2)


def test_multi_output():
    class MultiOutput(tf.Module):
        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            y = 2 * x
            return x, y

    run_model_graph(MultiOutput)


def test_if():
    class If(tf.Module):
        def get_input(self):
            return np.ones((2, 2), dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
        def func(self, x):
            @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
            def double(x):
                return 2 * x

            @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
            def triple(x):
                return 3 * x

            cond = True
            output = tf.raw_ops.If(
                cond=cond,
                input=[x],
                Tout=[tf.float32],
                output_shapes=[(2, 2)],
                then_branch=double.get_concrete_function(),
                else_branch=triple.get_concrete_function(),
            )
            return output[0]

    run_func_graph(If, runtime="vm")
    run_model_graph(If)


def test_stateless_while():
    class StatelessWhile(tf.Module):
        def get_input(self):
            return np.array([6], dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.float32)])
        def func(self, x):
            i = tf.constant(3.0)
            cond = lambda i: tf.less(i, x)
            body = lambda i: (tf.add(i, 2),)
            r = tf.while_loop(cond, body, [i])
            return r[0]

    run_func_graph(StatelessWhile, runtime="vm")
    run_model_graph(StatelessWhile)


def test_stateless_while_2var():
    class StatelessWhile2Var(tf.Module):
        def get_input(self):
            return np.array([20], dtype="float32")

        @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.float32)])
        def func(self, x):
            i = tf.constant(3.0)
            j = tf.constant(5.0)
            cond = lambda i, j: tf.less(i + j, x)
            body = lambda i, j: (tf.add(i, 2), tf.add(j, 3))
            r = tf.while_loop(cond, body, [i, j])
            return r

    run_model_graph(StatelessWhile2Var)


if __name__ == "__main__":
    pytest.main([__file__])
