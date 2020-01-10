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

def test_reshape():
    x = sym.Variable("x")
    y = sym.reshape(x, shape=(10, 20), name="y")
    assert(y.list_input_names() == ["x"])


def test_scalar_op():
    x = sym.Variable("x")
    y = (1 / (x * 2) - 1) ** 2
    assert(y.list_input_names() == ["x"])

def test_leaky_relu():
    x = sym.Variable("x")
    y = sym.leaky_relu(x, alpha=0.1)
    assert(y.list_input_names() == ["x"])

def test_prelu():
    x = sym.Variable("x")
    w = sym.Variable("w")
    y = sym.prelu(x, w)
    assert(y.list_input_names()[0] == 'x')
    assert(y.list_input_names()[1] == 'w')

if __name__ == "__main__":
    test_scalar_op()
    test_reshape()
    test_leaky_relu()
    test_prelu()
