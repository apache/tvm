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
"""Default legalization function for vision network related operators."""
from tvm import topi
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


@register_legalize("relax.vision.all_class_non_max_suppression")
def _vision_all_class_non_max_suppression(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.vision.all_class_non_max_suppression,
        call.args[0],
        call.args[1],
        call.args[2],
        call.args[3],
        call.args[4],
        output_format=call.attrs.output_format,
    )
