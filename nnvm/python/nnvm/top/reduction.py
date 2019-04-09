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
# pylint: disable=invalid-name, unused-argument
"""Reduction ops"""
from __future__ import absolute_import

import tvm
import topi
import topi.cuda
from . import registry as reg
from .registry import OpPattern

def _schedule_reduce(_, outs, target):
    """Generic schedule for reduce"""
    with tvm.target.create(target):
        return topi.generic.schedule_reduce(outs)


_fschedule_reduce = tvm.convert(_schedule_reduce)

def _compute_reduce(f):
    """auxiliary function"""
    def _compute(attrs, inputs, out_info):
        axis = attrs.get_int_tuple("axis")
        keepdims = attrs.get_bool("keepdims")
        if axis:
            return f(inputs[0], axis=axis, keepdims=keepdims)
        return f(inputs[0], keepdims=keepdims)
    return _compute

# sum
reg.register_pattern("sum", OpPattern.COMM_REDUCE)
reg.register_schedule("sum", _fschedule_reduce)

# max
reg.register_pattern("max", OpPattern.COMM_REDUCE)
reg.register_schedule("max", _fschedule_reduce)

# min
reg.register_pattern("min", OpPattern.COMM_REDUCE)
reg.register_schedule("min", _fschedule_reduce)

# collapse sum
reg.register_pattern("collapse_sum", OpPattern.COMM_REDUCE)
reg.register_schedule("collapse_sum", _fschedule_reduce)

# argmax
reg.register_pattern("argmax", OpPattern.COMM_REDUCE)
reg.register_schedule("argmax", _fschedule_reduce)

# argmin
reg.register_pattern("argmin", OpPattern.COMM_REDUCE)
reg.register_schedule("argmin", _fschedule_reduce)

# mean
reg.register_pattern("mean", OpPattern.COMM_REDUCE)
reg.register_schedule("mean", _fschedule_reduce)

# product
reg.register_pattern("prod", OpPattern.COMM_REDUCE)
reg.register_schedule("prod", _fschedule_reduce)
