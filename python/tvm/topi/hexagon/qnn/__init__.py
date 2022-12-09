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

""" Computes and schedules for Hexagon quantized ops """

from .avg_pool2d import qnn_avg_pool2d_compute, qnn_avg_pool2d_schedule
from .qadd_qsub_qmul import *
from .dequantize import (
    dequantize_compute,
    dequantize_schedule,
)

from .quantize import quantize_compute, tir_quantize_schedule
from .nn import *
from .qdepthwise_conv2d_slice import qdepthwise_conv2d_compute, qdepthwise_conv2d_schedule
from .adaptive_avg_pool1d import *
