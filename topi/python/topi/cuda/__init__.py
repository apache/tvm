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

# pylint: disable=redefined-builtin, wildcard-import
"""CUDA specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv1d import *
from .conv1d_transpose_ncw import *
from .conv2d import *
from .conv2d_hwcn import *
from .conv2d_int8 import *
from .conv2d_winograd import *
from .depthwise_conv2d import *
from .group_conv2d_nchw import *
from . import conv2d_alter_op
from .conv2d_transpose_nchw import *
from .deformable_conv2d import *
from .conv3d import *
from .conv3d_winograd import *
from . import conv3d_alter_op
from .reduction import schedule_reduce
from .softmax import *
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .dense import *
from .pooling import *
from .nn import schedule_lrn
from .batch_matmul import *
from .vision import *
from .ssd import *
from .nms import get_valid_counts, non_max_suppression
from .rcnn import *
from .sort import *
from .conv2d_nhwc_tensorcore import *
from .conv3d_ndhwc_tensorcore import *
from .dense_tensorcore import *
