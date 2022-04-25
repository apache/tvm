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

"""TOPI Testing Util functions.

Used to verify the correctness of operators in TOPI .
"""
from __future__ import absolute_import as _abs

from .conv1d_ncw_python import conv1d_ncw_python, group_conv1d_ncw_python
from .conv2d_hwcn_python import conv2d_hwcn_python
from .conv2d_nchw_python import conv2d_nchw_python
from .conv2d_nhwc_python import conv2d_nhwc_python
from .conv3d_ncdhw_python import conv3d_ncdhw_python
from .conv3d_ndhwc_python import conv3d_ndhwc_python
from .conv3d_transpose_ncdhw_python import conv3d_transpose_ncdhw_python
from .conv2d_transpose_python import conv2d_transpose_nchw_python, conv2d_transpose_nhwc_python
from .conv1d_transpose_ncw_python import conv1d_transpose_ncw_python
from .correlation_nchw_python import correlation_nchw_python
from .deformable_conv2d_python import deformable_conv2d_nchw_python, deformable_conv2d_nhwc_python
from .depthwise_conv2d_python import (
    depthwise_conv2d_python_nchw,
    depthwise_conv2d_python_nhwc,
    depthwise_conv2d_python_nchwc,
)
from .dilate_python import dilate_python
from .softmax_python import softmax_python, log_softmax_python
from .resize_python import resize1d_python, resize2d_python, resize3d_python
from .reorg_python import reorg_python
from .roi_align_python import roi_align_nchw_python, roi_align_nhwc_python
from .roi_pool_python import roi_pool_nchw_python
from .lrn_python import lrn_python
from .l2_normalize_python import l2_normalize_python
from .gather_python import gather_python
from .gather_nd_python import gather_nd_python
from .strided_slice_python import strided_slice_python, strided_set_python
from .batch_matmul import batch_matmul
from .batch_norm import batch_norm
from .slice_axis_python import slice_axis_python
from .sequence_mask_python import sequence_mask
from .poolnd_python import poolnd_python
from .pool_grad_python import pool_grad_nchw
from .one_hot import one_hot
from .depth_to_space import depth_to_space_python
from .space_to_depth import space_to_depth_python
from .crop_and_resize_python import crop_and_resize_python
from .common import (
    compare_numpy_tvm,
    get_injective_schedule,
    get_reduce_schedule,
    get_broadcast_schedule,
    get_elemwise_schedule,
    get_conv2d_nchw_implement,
    dispatch,
)
from .adaptive_pool_python import adaptive_pool
from .grid_sample_python import affine_grid_python, grid_sample_python
from .matrix_set_diag import matrix_set_diag
from .space_to_batch_nd import space_to_batch_nd_python
from .batch_to_space_nd import batch_to_space_nd_python
from .nll_loss import nll_loss
from .dense import dense
from .searchsorted import searchsorted_ref
from .conv2d_backcward_weight_python import conv2d_backward_weight_python
