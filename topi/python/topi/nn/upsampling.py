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
from ..util import simplify


def upsampling(data, scale, layout="NCHW", method='NEAREST_NEIGHBOR'):
    """Perform upsampling on the data.
       Nearest neighbor and bilinear upsampling are supported.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale : int
        Scaling factor

    layout : string, optional
        either "NCHW" or "NHWC"

    method : {"BILINEAR", "NEAREST_NEIGHBOR"}
        Method to be used for upsampling.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """
    base_layout = layout[0:4]
    if base_layout == "NCHW":
        out_shape = (simplify(data.shape[2] * scale), simplify(data.shape[3] * scale))
    elif layout == "NHWC":
        out_shape = (simplify(data.shape[1] * scale), simplify(data.shape[2] * scale))
    else:
        raise ValueError("not support this layout {} yet".format(layout))
    return topi.cpp.nn.upsampling(data, out_shape, layout, method)
