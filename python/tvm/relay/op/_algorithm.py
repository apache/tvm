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
"Definition of classic algorithms"
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import topi
from topi.util import get_const_int
from ..op import OpPattern, register_compute, register_schedule, register_pattern


@register_schedule("argsort")
def schedule_argsort(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_argsort(outs)


@register_compute("argsort")
def compute_argsort(attrs, inputs, _, target):
    """Compute definition of argsort"""
    axis = get_const_int(attrs.axis)
    is_ascend = bool(get_const_int(attrs.is_ascend))
    dtype = attrs.dtype
    return [topi.argsort(inputs[0], axis=axis, is_ascend=is_ascend, dtype=dtype)]


register_pattern("argsort", OpPattern.OPAQUE)


@register_schedule("topk")
def schedule_topk(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_topk(outs)


@register_compute("topk")
def compute_topk(attrs, inputs, _, target):
    """Compute definition of argsort"""
    k = get_const_int(attrs.k)
    axis = get_const_int(attrs.axis)
    ret_type = attrs.ret_type
    is_ascend = bool(get_const_int(attrs.is_ascend))
    dtype = attrs.dtype
    out = topi.topk(inputs[0], k, axis, ret_type, is_ascend, dtype)
    out = out if isinstance(out, list) else [out]
    return out


register_pattern("topk", OpPattern.OPAQUE)
