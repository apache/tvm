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
import nnvm.symbol as sym
import nnvm.graph as graph

def test_dense():
    x = sym.Variable('x')
    x1 = sym.dense(x, units=3, name="dense")
    x2 = sym.flatten(x1)
    x3 = sym.softmax(x2)
    assert x3.list_input_names() == ['x', 'dense_weight', 'dense_bias']


def test_concatenate_split():
    x = sym.Variable('x')
    y = sym.Variable('y')
    y = sym.concatenate(x, y)
    assert y.list_input_names() == ['x', 'y']
    z = sym.split(y, indices_or_sections=10)
    assert len(z.list_output_names()) == 10
    z = sym.split(y, indices_or_sections=[10, 20])
    assert len(z.list_output_names()) == 3

def test_expand_dims():
    x = sym.Variable('x')
    y = sym.expand_dims(x, axis=1, num_newaxis=2)
    assert y.list_input_names() == ['x']


def test_unary():
    x = sym.Variable('x')
    x = sym.exp(x)
    x = sym.log(x)
    x = sym.sigmoid(x)
    x = sym.tanh(x)
    x = sym.relu(x)
    assert x.list_input_names() == ['x']


def test_batchnorm():
    x = sym.Variable('x')
    x = sym.batch_norm(x, name="bn")
    assert x.list_input_names() == [
        "x", "bn_gamma", "bn_beta", "bn_moving_mean", "bn_moving_var"]


if __name__ == "__main__":
    test_concatenate_split()
    test_expand_dims()
    test_dense()
    test_unary()
    test_batchnorm()
