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
import numpy as np

def upsample_nearest(arr, scale):
    """ Populate the array by scale factor"""
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)

def upsampling_python(data, scale, layout='NCHW'):
    """ Python version of scaling using nearest neighbour """

    ishape = data.shape
    if layout == 'NCHW':
        oshape = (ishape[0], ishape[1], ishape[2]*scale, ishape[3]*scale)
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[1]):
                output_np[b, c, :, :] = upsample_nearest(data[b, c, :, :], scale)
        return output_np
    if layout == 'NHWC':
        oshape = (ishape[0], ishape[1]*scale, ishape[1]*scale, ishape[3])
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[3]):
                output_np[b, :, :, c] = upsample_nearest(data[b, :, :, c], scale)
        return output_np
    raise ValueError("not support this layout {} yet".format(layout))
