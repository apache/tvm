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
"""Backend compiler related feature registration"""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import
from . import op as _reg
from ._reduce import _schedule_reduce
from .op import OpPattern

schedule_injective = _reg.schedule_injective
schedule_broadcast = _reg.schedule_injective
schedule_concatenate = _reg.schedule_concatenate


_reg.register_schedule("collapse_sum_like", _schedule_reduce)
_reg.register_schedule("broadcast_to", schedule_broadcast)
_reg.register_schedule("broadcast_to_like", schedule_broadcast)
_reg.register_schedule("expand_dims", schedule_broadcast)
_reg.register_schedule("squeeze", schedule_injective)
_reg.register_schedule("reshape", schedule_injective)
_reg.register_schedule("reshape_like", schedule_injective)
_reg.register_schedule("full", schedule_injective)
_reg.register_schedule("full_like", schedule_injective)
_reg.register_schedule("arange", schedule_injective)
_reg.register_schedule("reverse", schedule_injective)
_reg.register_schedule("repeat", schedule_broadcast)
_reg.register_schedule("tile", schedule_broadcast)
_reg.register_schedule("cast", schedule_injective)
_reg.register_schedule("strided_slice", schedule_injective)
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_schedule("split", schedule_injective)
_reg.register_schedule("take", schedule_injective)
_reg.register_schedule("transpose", schedule_injective)
_reg.register_schedule("where", schedule_broadcast)
_reg.register_schedule("stack", schedule_injective)
_reg.register_schedule("concatenate", schedule_concatenate)
_reg.register_schedule("_contrib_reverse_reshape", schedule_injective)
_reg.register_schedule("gather_nd", schedule_injective)
_reg.register_schedule("sequence_mask", schedule_injective)


# layout_transform
_reg.register_schedule("layout_transform", schedule_injective)
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)
