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
"""Common utility for topi test"""

import tvm
from tvm import topi

_injective_schedule = {
    "generic": topi.generic.schedule_injective,
    "cpu": topi.x86.schedule_injective,
    "arm_cpu": topi.arm_cpu.schedule_injective,
    "gpu": topi.cuda.schedule_injective,
    "hls": topi.hls.schedule_injective,
}

_reduce_schedule = {
    "generic": topi.generic.schedule_reduce,
    "cpu": topi.x86.schedule_reduce,
    "gpu": topi.cuda.schedule_reduce,
    "hls": topi.cuda.schedule_reduce,
}


def dispatch(target, dispatch_map):
    if isinstance(target, str):
        target = tvm.target.Target(target)
    assert isinstance(target, tvm.target.Target)
    for key in target.keys:
        if key in dispatch_map:
            return dispatch_map[key]
    return dispatch_map["generic"]


def get_injective_schedule(target):
    return dispatch(target, _injective_schedule)


def get_reduce_schedule(target):
    return dispatch(target, _reduce_schedule)


get_broadcast_schedule = get_injective_schedule
get_elemwise_schedule = get_injective_schedule

_conv2d_nchw_implement = {
    "generic": (topi.nn.conv2d_nchw, topi.generic.schedule_conv2d_nchw),
    "cpu": (topi.x86.conv2d_nchw, topi.x86.schedule_conv2d_nchw),
    "arm_cpu": (
        topi.arm_cpu.conv2d_nchw_spatial_pack,
        topi.arm_cpu.schedule_conv2d_nchw_spatial_pack,
    ),
    "gpu": (topi.cuda.conv2d_nchw, topi.cuda.schedule_conv2d_nchw),
    "mali": (topi.mali.conv2d_nchw_spatial_pack, topi.mali.schedule_conv2d_nchw_spatial_pack),
    "bifrost": (
        topi.bifrost.conv2d_nchw_spatial_pack,
        topi.bifrost.schedule_conv2d_nchw_spatial_pack,
    ),
    "intel_graphics": (topi.intel_graphics.conv2d_nchw, topi.intel_graphics.schedule_conv2d_nchw),
    "hls": (topi.nn.conv2d_nchw, topi.hls.schedule_conv2d_nchw),
}


def get_conv2d_nchw_implement(target):
    return dispatch(target, _conv2d_nchw_implement)
