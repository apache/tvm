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
"""TVM operator batch_to_space_nd compute."""
from __future__ import absolute_import
from . import cpp


def batch_to_space_nd(data, block_shape, crop_begin_list, crop_end_list):
    """Perform space to batch transformation on the data

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D Tensor with shape [batch, spatial_shape, remaining_shapes],
        where spatial_shape has M dimensions.

    block_size : list of ints
        list of size [M] where M is number of spatial dims, specifies block
        size for each spatial dimension.

    crop_begin_list : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        begin crop size for each spatial dimension.

    crop_end_list : list of ints
        list of shape [M] where M is number of spatial dims, specifies
        end crop size for each spatial dimension.

    Returns
    -------
    output : tvm.te.Tensor
    """

    return cpp.nn.batch_to_space_nd(data, block_shape, crop_begin_list, crop_end_list)
