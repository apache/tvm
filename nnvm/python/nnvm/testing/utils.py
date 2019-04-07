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
"""Helper utility to create common workload for testing."""
from __future__ import absolute_import as _abs

import numpy as np
import tvm
from ..compiler import graph_util
from ..import graph
from . init import Xavier

def create_workload(net, batch_size, image_shape=(3, 224, 224),
                    dtype="float32", initializer=None, seed=0):
    """Helper function to create benchmark workload for input network

    Parameters
    ----------
    net : nnvm.Symbol
        The selected network symbol to use

    batch_size : int
        The batch size used in the model

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    initializer : Initializer
        The initializer used

    seed : int
        The seed used in initialization.

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    if image_shape is None:
        image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    params = {}
    g = graph.create(net)
    input_shapes, _ = graph_util.infer_shape(g, data=data_shape)
    shape_dict = dict(zip(g.index.input_names, input_shapes))
    np.random.seed(seed)
    initializer = initializer if initializer else Xavier()
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.zeros(v).astype(dtype)
        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
    return net, params
