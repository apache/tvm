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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Upsampling in python"""
import math
import numpy as np

def upsample_nearest(arr, scale):
    """ Populate the array by scale factor"""
    h, w = arr.shape
    out_h = math.floor(h * scale[0])
    out_w = math.floor(w * scale[1])
    out = np.empty((out_h, out_w))
    for y in range(out_h):
        for x in range(out_w):
            in_y = math.floor(y / scale[0])
            in_x = math.floor(x / scale[1])
            out[y, x] = arr[in_y, in_x]
    return out

def upsampling_python(data, scale, layout='NCHW'):
    """ Python version of scaling using nearest neighbour """

    ishape = data.shape
    if layout == 'NCHW':
        oshape = (ishape[0], ishape[1], math.floor(ishape[2]*scale[0]), math.floor(ishape[3]*scale[1]))
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[1]):
                output_np[b, c, :, :] = upsample_nearest(data[b, c, :, :], scale)
        return output_np
    if layout == 'NHWC':
        oshape = (ishape[0], math.floor(ishape[1]*scale[0]), math.floor(ishape[1]*scale[1]), ishape[3])
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[3]):
                output_np[b, :, :, c] = upsample_nearest(data[b, :, :, c], scale)
        return output_np
    raise ValueError("not support this layout {} yet".format(layout))
