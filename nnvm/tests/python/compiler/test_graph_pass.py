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
"""Unittest cases for graph pass"""
import nnvm
import nnvm.compiler
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_infer_attr():
    x = sym.Variable("x")
    y = x * 2
    g = nnvm.graph.create(y)
    ishape, oshape = graph_util.infer_shape(g, x=(10,20))
    assert tuple(oshape[0]) == (10, 20)

    itype, otype = graph_util.infer_dtype(g, x="float32")
    assert otype[0] == "float32"

if __name__ == "__main__":
    test_infer_attr()
