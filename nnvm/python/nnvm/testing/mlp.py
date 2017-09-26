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
from .. import symbol as sym
from . utils import create_workload

def get_symbol(num_classes=1000):
    data = sym.Variable('data')
    data = sym.flatten(data=data)
    fc1 = sym.dense(data=data, name='fc1', units=128)
    act1 = sym.relu(data=fc1, name='relu1')
    fc2 = sym.dense(data=act1, name='fc2', units=64)
    act2 = sym.relu(data=fc2, name='relu2')
    fc3 = sym.dense(data=act2, name='fc3', units=num_classes)
    mlp = sym.softmax(data=fc3, name='softmax')
    return mlp

def get_workload(batch_size, num_classes=1000, image_shape=(3, 224, 224), dtype="float32"):
    """Get benchmark workload for a simple multilayer perceptron

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol(num_classes=num_classes)
    return create_workload(net, batch_size, image_shape, dtype)
