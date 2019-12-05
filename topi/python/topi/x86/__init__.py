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
"""x86 specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .conv2d import schedule_conv2d, schedule_conv2d_nhwc
from .binarize_pack import schedule_binarize_pack
from .binary_dense import schedule_binary_dense
from .nn import *
from .conv2d_int8 import *
from .injective import *
from .reduction import *
from .pooling import schedule_pool, schedule_adaptive_pool
from .bitserial_conv2d import schedule_bitserial_conv2d
from .bitserial_dense import schedule_bitserial_dense
from .depthwise_conv2d import schedule_depthwise_conv2d_NCHWc
from .dense import _schedule_dense, _schedule_dense_pack, _schedule_dense_nopack
from .batch_matmul import schedule_batch_matmul
from .roi_align import roi_align_nchw
from .conv2d_transpose import _schedule_conv2d_transpose_nchw
from .sparse import *
from .conv2d_alter_op import *
