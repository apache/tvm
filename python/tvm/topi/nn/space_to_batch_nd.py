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
# pylint: disable=invalid-name
"""TVM operator space_to_batch_nd compute."""
from __future__ import absolute_import
from . import cpp


def space_to_batch_nd(data, block_shape, pad_before, pad_after, pad_value=0.0):
    """Perform batch to space transformation on the data

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D Tensor with shape [batch, spatial_shape, remaining_shapes],
        where spatial_shape has M dimensions.

    block_shape : list of ints
        list of size [M] where M is number of spatial dims, specifies block
        size for each spatial dimension.

    pad_before : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        zero-padding size before each spatial dimension.

    pad_after : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        zero-padding size after each spatial dimension.

    pad_value : float, optional
        The value used for padding.

    Returns
    -------
    output : tvm.te.Tensor
    """

    return cpp.nn.space_to_batch_nd(data, block_shape, pad_before, pad_after, pad_value)
