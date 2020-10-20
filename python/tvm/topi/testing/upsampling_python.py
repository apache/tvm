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
from tvm.topi.util import nchw_pack_layout


def upsample_nearest(arr, scale):
    """ Populate the array by scale factor"""
    h, w = arr.shape
    out_h = int(round(h * scale[0]))
    out_w = int(round(w * scale[1]))
    out = np.empty((out_h, out_w))
    for y in range(out_h):
        for x in range(out_w):
            in_y = math.floor(y / scale[0])
            in_x = math.floor(x / scale[1])
            out[y, x] = arr[in_y, in_x]
    return out


def upsampling_python(data, scale, layout="NCHW"):
    """ Python version of scaling using nearest neighbour """

    ishape = data.shape
    if layout == "NCHW":
        oshape = (
            ishape[0],
            ishape[1],
            int(round(ishape[2] * scale[0])),
            int(round(ishape[3] * scale[1])),
        )
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[1]):
                output_np[b, c, :, :] = upsample_nearest(data[b, c, :, :], scale)
        return output_np
    # NCHWinic
    if nchw_pack_layout(layout):
        oshape = (
            ishape[0],
            ishape[1],
            int(round(ishape[2] * scale[0])),
            int(round(ishape[3] * scale[1])),
            ishape[4],
            ishape[5],
        )
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for ib in range(oshape[4]):
                for c in range(oshape[1]):
                    for ic in range(oshape[5]):
                        output_np[b, c, :, :, ib, ic] = upsample_nearest(
                            data[b, c, :, :, ib, ic], scale
                        )
        return output_np

    if layout == "NHWC":
        oshape = (
            ishape[0],
            int(round(ishape[1] * scale[0])),
            int(round(ishape[2] * scale[1])),
            ishape[3],
        )
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[3]):
                output_np[b, :, :, c] = upsample_nearest(data[b, :, :, c], scale)
        return output_np
    raise ValueError("not support this layout {} yet".format(layout))


def upsample3d_nearest(arr, scale):
    """ Populate the array by scale factor"""
    d, h, w = arr.shape
    out_d = int(round(d * scale[0]))
    out_h = int(round(h * scale[1]))
    out_w = int(round(w * scale[2]))
    out = np.empty((out_d, out_h, out_w))
    for z in range(out_d):
        for y in range(out_h):
            for x in range(out_w):
                in_z = math.floor(z / scale[0])
                in_y = math.floor(y / scale[1])
                in_x = math.floor(x / scale[2])
                out[z, y, x] = arr[in_z, in_y, in_x]
    return out


def upsampling3d_python(data, scale, layout="NCDHW"):
    """ Python version of 3D scaling using nearest neighbour """

    ishape = data.shape
    if layout == "NCDHW":
        oshape = (
            ishape[0],
            ishape[1],
            int(round(ishape[2] * scale[0])),
            int(round(ishape[3] * scale[1])),
            int(round(ishape[4] * scale[2])),
        )
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[1]):
                output_np[b, c, :, :, :] = upsample3d_nearest(data[b, c, :, :, :], scale)
        return output_np
    if layout == "NDHWC":
        oshape = (
            ishape[0],
            int(round(ishape[1] * scale[0])),
            int(round(ishape[2] * scale[1])),
            int(round(ishape[3] * scale[2])),
            ishape[4],
        )
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[4]):
                output_np[b, :, :, :, c] = upsample3d_nearest(data[b, :, :, :, c], scale)
        return output_np
    raise ValueError("not support this layout {} yet".format(layout))
