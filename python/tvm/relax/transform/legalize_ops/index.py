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
"""Default legalization function for index operators."""
import logging

from tvm import topi, tir
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


@register_legalize("relax.take")
def _take(bb: BlockBuilder, call: Call) -> Expr:
    # Currently Relax `take` operator doesn't provide the mode choices and
    # requires input indices to be in range.
    # We use fast mode, which leads to runtime error whenever some index is
    # out of bound.
    return bb.call_te(topi.take, call.args[0], call.args[1], call.attrs.axis, mode="fast")


@register_legalize("relax.strided_slice")
def _strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    if not all(
        isinstance(call.args[0].struct_info.shape.values[i.value], tir.IntImm)
        for i in call.attrs.axes
    ):
        logging.info(
            "Cases where an axis with symbolic length is sliced are not able "
            "to be legalized through TOPI"
        )
        return call

    strides = (
        [tir.IntImm("int64", 1)] * len(call.attrs.axes)
        if call.attrs.strides is None
        else call.attrs.strides
    )
    return bb.call_te(
        topi.strided_slice,
        call.args[0],
        call.attrs.begin,
        call.attrs.end,
        strides,
        call.attrs.axes,
        slice_mode="end",
    )
