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
"""Default legalization function for binary operators."""
from tvm import topi
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import TEFunc, LegalizeFunc, _try_convert_to_scalar_const, register_legalize


def _binary(te_func: TEFunc) -> LegalizeFunc:
    """A common wrapper util for the legalization of binary operators.

    It detects if one of the binary op arguments is a constant scalar. It so,
    it extracts the scalar value to simplify the generated PrimFunc.
    """

    def binary_call_te(bb: BlockBuilder, call: Call) -> Expr:
        # To simplify the created PrimFunc, we first check if arg1 is a constant scalar.
        # If it is not, we then check if arg0 is a constant scalar.
        arg0 = call.args[0]
        arg1 = _try_convert_to_scalar_const(call.args[1])
        if isinstance(arg1, Expr):  # type: ignore
            arg0 = _try_convert_to_scalar_const(arg0)
        return bb.call_te(te_func, arg0, arg1)

    return binary_call_te


register_legalize("relax.add", _binary(topi.add))
register_legalize("relax.divide", _binary(topi.divide))
register_legalize("relax.floor_divide", _binary(topi.floor_divide))
register_legalize("relax.multiply", _binary(topi.multiply))
register_legalize("relax.power", _binary(topi.power))
register_legalize("relax.subtract", _binary(topi.subtract))
register_legalize("relax.equal", _binary(topi.equal))

register_legalize("relax.greater", _binary(topi.greater))
register_legalize("relax.greater_equal", _binary(topi.greater_equal))
register_legalize("relax.less", _binary(topi.less))
register_legalize("relax.less_equal", _binary(topi.less_equal))
register_legalize("relax.not_equal", _binary(topi.not_equal))

register_legalize("relax.maximum", _binary(topi.maximum))
register_legalize("relax.minimum", _binary(topi.minimum))

# bitwise
register_legalize("relax.bitwise_and", _binary(topi.bitwise_and))
register_legalize("relax.bitwise_or", _binary(topi.bitwise_or))
register_legalize("relax.bitwise_xor", _binary(topi.bitwise_xor))

# logical
register_legalize("relax.logical_and", _binary(topi.logical_and))
register_legalize("relax.logical_or", _binary(topi.logical_or))
register_legalize("relax.logical_xor", _binary(topi.logical_xor))
