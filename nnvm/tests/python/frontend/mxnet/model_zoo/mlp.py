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

"""
a simple multilayer perceptron
"""
import mxnet as mx
import nnvm

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.softmax(data = fc3, name = 'softmax')
    return mlp

def get_symbol_nnvm(num_classes=10, **kwargs):
    data = nnvm.symbol.Variable('data')
    data = nnvm.sym.flatten(data=data)
    fc1  = nnvm.symbol.dense(data = data, name='fc1', units=128)
    act1 = nnvm.symbol.relu(data = fc1, name='relu1')
    fc2  = nnvm.symbol.dense(data = act1, name = 'fc2', units = 64)
    act2 = nnvm.symbol.relu(data = fc2, name='relu2')
    fc3  = nnvm.symbol.dense(data = act2, name='fc3', units=num_classes)
    mlp  = nnvm.symbol.softmax(data = fc3, name = 'softmax')
    return mlp
