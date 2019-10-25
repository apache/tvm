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
"""TVM operator upsampling compute."""
from __future__ import absolute_import
import topi
import tvm
from ..util import simplify


def upsampling(data, scale_h, scale_w, layout="NCHW", method='nearest_neighbor',
               align_corners=False):
    """Perform upsampling on the data.
       Nearest neighbor and bilinear upsampling are supported.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale_h : float
        Scaling factor for height

    scale_w : float
        Scaling factor for width

    layout : string, optional
        either "NCHW" or "NHWC"

    method : {"bilinear", "nearest_neighbor", "bicubic"}
        Method to be used for upsampling.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale_h, in_width*scale_w]
        or [batch, in_height*scale, in_width*scale, channel]
    """
    base_layout = layout[0:4]
    if base_layout == "NCHW":
        out_shape = (simplify(topi.cast(tvm.round(data.shape[2] * scale_h), data.shape[2].dtype)),
                     simplify(topi.cast(tvm.round(data.shape[3] * scale_w), data.shape[3].dtype)))
    elif layout == "NHWC":
        out_shape = (simplify(topi.cast(tvm.round(data.shape[1] * scale_h), data.shape[1].dtype)),
                     simplify(topi.cast(tvm.round(data.shape[2] * scale_w), data.shape[2].dtype)))

    else:
        raise ValueError("not support this layout {} yet".format(layout))
    return topi.image.resize(data, out_shape, layout=layout,
                             method=method, align_corners=align_corners)
