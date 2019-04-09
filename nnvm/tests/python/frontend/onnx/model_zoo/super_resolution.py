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
"""NNVM symbol corresponding to super_resolution.onnx example."""
from nnvm import sym

def get_super_resolution():
    factor = 3
    size = 224
    data = sym.Variable(name='9')
    conv1 = sym.conv2d(data, channels=64, kernel_size=(5, 5), padding=(2, 2), use_bias=False)
    relu1 = sym.relu(conv1 + sym.expand_dims(sym.Variable(name='2', shape=(64)), axis=1, num_newaxis=2))
    conv2 = sym.conv2d(relu1, channels=64, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    relu2 = sym.relu(conv2 + sym.expand_dims(sym.Variable(name='4', shape=(64)), axis=1, num_newaxis=2))
    conv3 = sym.conv2d(relu2, channels=32, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    relu3 = sym.relu(conv3 + sym.expand_dims(sym.Variable(name='6', shape=(32)), axis=1, num_newaxis=2))
    conv4 = sym.conv2d(relu3, channels=factor**2, kernel_size=(3, 3), padding=(1, 1), use_bias=False)
    conv4 = conv4 + sym.expand_dims(sym.Variable(name='8', shape=(factor**2)), axis=1, num_newaxis=2)
    # TODO(zhreshold): allow shape inference for batch size > 1
    r1 = sym.reshape(conv4, shape=(1, 1, factor, factor, size, size))
    t1 = sym.transpose(r1, axes=(0, 1, 4, 2, 5, 3))
    r2 = sym.reshape(t1, shape=(1, 1, size * factor, size * factor))
    return r2
