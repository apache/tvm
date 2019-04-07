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
"""TVM operator input resize compute."""
from __future__ import absolute_import
import topi

def resize(data, size, layout="NCHW", align_corners=False, method="BILINEAR"):
    """Perform resize operation on the data.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    size: Tuple
        Output resolution scale to

    layout: string, optional
        either "NCHW" or "NHWC"

    align_corners: Boolean, optional
        To preserve the values at the corner pixels

    method: {"BILINEAR", "NEAREST_NEIGHBOR"}
        Method to be used for resizing.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """
    return topi.cpp.image.resize(data, size, layout, align_corners, method)
