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
# pylint: disable=no-else-return, invalid-name, unused-argument, too-many-arguments, consider-using-in
"""Backend compiler related feature registration"""

from __future__ import absolute_import

from tvm.te.hybrid import script
from ...op import register_shape_func
from ...op import register_broadcast_schedule

# pad
register_broadcast_schedule("dyn.nn.pad")

#####################
#  Shape functions  #
#####################

@script
def _dyn_pad_shape_func(data, pad_width):
    ndim = len(data.shape)
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = int64(pad_width[i, 0] + pad_width[i, 1] + data.shape[i])
    return out

@register_shape_func("dyn.nn.pad", True)
def pad_shape_func(attrs, inputs, data):
    """
    Shape function for dynamic pad op.
    """
    return [_dyn_pad_shape_func(inputs[0], inputs[1])]
