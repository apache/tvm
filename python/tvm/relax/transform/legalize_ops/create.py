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
"""Default legalization function for creation operators."""
from typing import Optional

import numpy as np

from tvm import tir, topi

from ...block_builder import BlockBuilder
from ...expr import Call, Expr, PrimValue, const
from .common import LegalizeFunc, _try_convert_to_scalar_const, register_legalize


def _full(is_like: bool, fill_value: Optional[float], primfunc_name: str) -> LegalizeFunc:
    def full_call_te(bb: BlockBuilder, call: Call) -> Expr:
        _fill_value = (
            _try_convert_to_scalar_const(call.args[1], python_native=True)
            if fill_value is None
            else fill_value
        )

        return bb.call_te(
            topi.full,
            call.args[0].struct_info.shape if is_like else call.args[0],
            call.struct_info.dtype,
            _fill_value,
            primfunc_name_hint=primfunc_name,
        )

    return full_call_te


def _tril_triu(is_upper: bool, primfunc_name: str) -> LegalizeFunc:
    def tril_triu_call_te(bb: BlockBuilder, call: Call) -> Expr:
        data, k = call.args
        return bb.call_te(
            topi.trilu,
            data,
            k,
            upper=is_upper,
            primfunc_name_hint=primfunc_name,
        )

    return tril_triu_call_te


register_legalize("relax.full", _full(is_like=False, fill_value=None, primfunc_name="full"))
register_legalize("relax.full_like", _full(is_like=True, fill_value=None, primfunc_name="full"))
register_legalize("relax.ones", _full(is_like=False, fill_value=1.0, primfunc_name="ones"))
register_legalize("relax.ones_like", _full(is_like=True, fill_value=1.0, primfunc_name="ones"))
register_legalize("relax.zeros", _full(is_like=False, fill_value=0.0, primfunc_name="zeros"))
register_legalize("relax.zeros_like", _full(is_like=True, fill_value=0.0, primfunc_name="zeros"))
register_legalize("relax.tril", _tril_triu(is_upper=False, primfunc_name="tril"))
register_legalize("relax.triu", _tril_triu(is_upper=True, primfunc_name="triu"))


@register_legalize("relax.arange")
def _arange(bb: BlockBuilder, call: Call) -> Expr:
    assert len(call.args) == 3
    assert all([isinstance(x, PrimValue) for x in call.args])
    start, end, step = [x.value for x in call.args]
    dtype = call.attrs.dtype

    def is_const_scalar(x: PrimValue):
        return isinstance(x.value, (tir.IntImm, tir.FloatImm))

    if all([is_const_scalar(x) for x in call.args]):
        return const(np.arange(start.value, end.value, step.value, dtype=dtype), dtype=dtype)
    else:
        return bb.call_te(topi.arange, start, end, step, dtype)
