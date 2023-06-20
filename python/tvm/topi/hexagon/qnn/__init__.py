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

from .adaptive_avg_pool1d import *
from .avg_pool2d import *
from .conv2d_alter_op import *
from .dense_alter_op import *
from .dequantize import dequantize_compute, dequantize_schedule
from .global_avg_pool2d import *
from .nn import *
from .qadd_qsub_qmul import *
from .qdense import *
from .qdepthwise_conv2d_slice import qdepthwise_conv2d_compute, qdepthwise_conv2d_schedule
from .quantize import quantize_compute, tir_quantize_schedule
