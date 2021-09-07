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

""" Hexagon testing infrastructure """

import tvm
import numpy


def ceildiv(o, d):
    return tvm.tir.floordiv(o + d - 1, d)


def get_packed_layout(logical_shape_nhwc, block_shape, packed_C=True):
    shape = [logical_shape_nhwc[0]]
    off_h, off_w, off_c = block_shape
    shape.append(ceildiv(logical_shape_nhwc[1], off_h))
    shape.append(ceildiv(logical_shape_nhwc[2], off_w))
    if packed_C:
        shape.append(ceildiv(logical_shape_nhwc[3], off_c))
        shape.extend(block_shape)
    else:
        shape.extend([off_h, off_w, logical_shape_nhwc[-1]])
    return shape


def build_and_run(inputs, func, target, target_host, *args, **kwargs):
    s, placeholders, binds = func(*args, **kwargs)

    func = tvm.build(s, placeholders, target=target, target_host=target_host, binds=binds)
    dev = tvm.device(target)
    tensors = []
    for tensor in inputs:
        tensors.append(tvm.nd.array(tensor, dev))
    tensors.append(
        tvm.nd.array(
            numpy.zeros([i.value for i in placeholders[-1].shape], dtype=placeholders[-1].dtype),
            dev,
        )
    )
    func(*tensors)

    return tensors[-1].asnumpy()
