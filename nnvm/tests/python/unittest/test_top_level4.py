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

def test_binary_broadcast():
    x = sym.Variable('x')
    y = sym.Variable('y')
    z = x + y
    z = x * y
    z = x - y
    z = x / y


def test_broadcast_to():
    x = sym.Variable('x')
    y = sym.broadcast_to(x, shape=(3, 3))
    assert y.list_input_names() == ["x"]


if __name__ == "__main__":
    test_binary_broadcast()
    test_broadcast_to()
