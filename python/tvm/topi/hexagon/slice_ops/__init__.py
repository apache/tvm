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

from .avg_pool2d import avg_pool2d_compute, avg_pool2d_STIR_schedule
from .add_subtract_multiply import *
from .cast import (
    cast_f16_f32_compute,
    cast_f16_f32_schedule,
    cast_f32_f16_compute,
    cast_f32_f16_schedule,
)
from .softmax_slice import *
