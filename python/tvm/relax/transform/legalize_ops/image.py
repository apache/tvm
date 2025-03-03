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
"""Default legalization function for image operators."""
from tvm import topi
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


@register_legalize("relax.image.resize2d")
def _image_resize2d(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.image.resize2d,
        call.args[0],
        roi=call.attrs.roi,
        size=call.args[1],
        layout=call.attrs.layout,
        method=call.attrs.method,
        coordinate_transformation_mode=call.attrs.coordinate_transformation_mode,
        rounding_method=call.attrs.rounding_method,
        bicubic_alpha=call.attrs.cubic_alpha,
        bicubic_exclude=call.attrs.cubic_exclude,
        extrapolation_value=call.attrs.extrapolation_value,
    )
