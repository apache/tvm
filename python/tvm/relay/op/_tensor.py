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
#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import topi
from .op import register_compute, register_schedule, register_pattern
from .op import schedule_injective, OpPattern

schedule_broadcast = schedule_injective
schedule_elemwise = schedule_injective

register_schedule("log", schedule_broadcast)
register_schedule("exp", schedule_broadcast)
register_schedule("sqrt", schedule_broadcast)
register_schedule("rsqrt", schedule_broadcast)
register_schedule("sigmoid", schedule_broadcast)
register_schedule("floor", schedule_broadcast)
register_schedule("ceil", schedule_broadcast)
register_schedule("trunc", schedule_broadcast)
register_schedule("round", schedule_broadcast)
register_schedule("sign", schedule_broadcast)
register_schedule("abs", schedule_broadcast)
register_schedule("tanh", schedule_broadcast)
register_schedule("logical_not", schedule_broadcast)
register_schedule("negative", schedule_broadcast)
register_schedule("copy", schedule_broadcast)

register_schedule("add", schedule_broadcast)
register_schedule("subtract", schedule_broadcast)
register_schedule("multiply", schedule_broadcast)
register_schedule("divide", schedule_broadcast)
register_schedule("power", schedule_injective)
register_schedule("mod", schedule_broadcast)
register_schedule("logical_and", schedule_broadcast)
register_schedule("logical_or", schedule_broadcast)
register_schedule("equal", schedule_broadcast)
register_schedule("not_equal", schedule_broadcast)
register_schedule("less", schedule_broadcast)
register_schedule("less_equal", schedule_broadcast)
register_schedule("greater", schedule_broadcast)
register_schedule("greater_equal", schedule_broadcast)
register_schedule("maximum", schedule_injective)
register_schedule("minimum", schedule_injective)
register_schedule("right_shift", schedule_injective)
register_schedule("left_shift", schedule_injective)
register_schedule("shape_of", schedule_injective)

# zeros
@register_compute("zeros")
def zeros_compute(attrs, inputs, output_type, target):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 0.0)]

register_schedule("zeros", schedule_broadcast)
register_pattern("zeros", OpPattern.ELEMWISE)

# zeros_like
@register_compute("zeros_like")
def zeros_like_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 0.0)]

register_schedule("zeros_like", schedule_broadcast)

# ones
@register_compute("ones")
def ones_compute(attrs, inputs, output_type, target):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 1.0)]

register_schedule("ones", schedule_broadcast)
register_pattern("ones", OpPattern.ELEMWISE)

# ones_like
@register_compute("ones_like")
def ones_like(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 1.0)]

register_schedule("ones_like", schedule_broadcast)

# clip
@register_compute("clip")
def clip_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.clip(inputs[0], attrs.a_min, attrs.a_max)]

register_schedule("clip", schedule_elemwise)
