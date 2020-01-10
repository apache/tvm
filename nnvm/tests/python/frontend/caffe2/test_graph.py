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
"""Test graph equality of caffe2 models."""
import nnvm
from nnvm.compiler import graph_util, graph_attr
from model_zoo import c2_squeezenet, squeezenet

def compare_graph(init, predict, nnvm_sym, ishape):
    caffe2_sym, params = nnvm.frontend.from_caffe2(init, predict)
    g1 = nnvm.graph.create(caffe2_sym)
    g2 = nnvm.graph.create(nnvm_sym)
    input_name = predict.external_input[0]
    ishapes = {input_name: ishape}
    graph_attr.set_shape_inputs(g1, ishapes)
    graph_attr.set_shape_inputs(g2, ishapes)
    g1 = g1.apply("InferShape").apply("SimplifyInference")
    g2 = g2.apply("InferShape").apply("SimplifyInference")
    graph_util.check_graph_equal(g1, g2)

def test_squeeze_net():
    symbol, params = squeezenet.get_workload(version='1.1')
    compare_graph(c2_squeezenet.init_net, c2_squeezenet.predict_net, symbol, ishape=(1, 3, 224, 224))


if __name__ == '__main__':
    test_squeeze_net()
