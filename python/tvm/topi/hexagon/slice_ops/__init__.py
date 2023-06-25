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

""" Computes and Schedules for Hexagon slice ops. """

from .avg_pool2d import avg_pool2d_NHWC, avg_pool2d_NCHW, avg_pool2d_schedule
from .max_pool2d import max_pool2d_compute, max_pool2d_STIR_schedule
from .add_subtract_multiply import *
from .argmax import argmax_compute, argmax_schedule
from .batch_flatten import batch_flatten_compute, batch_flatten_stir_schedule
from .softmax_slice import *
from .clip import *
from .cast import (
    cast_f16_f32_compute,
    cast_f16_f32_schedule,
    cast_f32_f16_compute,
    cast_f32_f16_schedule,
)
from .conv2d import *
from .reshape import reshape_compute, reshape_stir_schedule
from .relu import relu_compute, relu_stir_schedule
from .tanh import tanh_te_compute, tanhf16_schedule
from .dwconv2d import *
from .depth_to_space import d2s_compute, d2s_schedule
from .global_avg_pool2d import *
from .dense import *
