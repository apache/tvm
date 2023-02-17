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
"""Common functionality for legalization."""
from typing import Callable, Union

import tvm
from tvm import te
from ...block_builder import BlockBuilder
from ...expr import Call, Expr, Constant


##################### Types #####################


# The function type of a TE function, which accepts TE Tensors and
# other attributes, and returns the output TE Tensor.
TEFunc = Callable[..., te.Tensor]

# The function type of a legalization function, which takes a
# BlockBuilder and the Call to be legalized, and outputs the legalization
# result Expr.
LegalizeFunc = Callable[[BlockBuilder, Call], Expr]


##################### Utilities #####################


def _try_convert_to_scalar_const(expr: Expr) -> Union[Expr, bool, float, int]:
    """Check if the input Expr is a scalar constant.
    If it is, return its plain value.
    If it is not, return the input expr.

    Parameters
    ----------
    expr : Expr
        The expr to be checked and converted.

    Returns
    --–----
    ret : Union[Expr, bool, float, int]
        Return a Python native value (int/float/bool) if the given
        expr is a scalar constant. Or return the input itself
        if it is not.
    """
    if isinstance(expr, Constant) and expr.struct_info.ndim == 0:
        return expr.data.numpy()[()].item()
    else:
        return expr


def _call_topi_without_attr(te_func: TEFunc) -> LegalizeFunc:
    """A common wrapper util for the ops who has no attributes and whose
    legalization is simply passing its arguments to some TE function.
    """
    return lambda bb, call: bb.call_te(te_func, *call.args)


##################### Decorators #####################

_LEGALIZE_ATTR_NAME = "FLegalize"


def register_legalize(op_name: str, legal_func: LegalizeFunc = None):
    """Register legal transformation function for a Relax op.

    Parameters
    ----------
    op_name : str
        The name of the operator

    legal_func: function (bb: BlockBuilder, call: Call) -> new_expr: Expr
        The function for transforming an expr to another expr.
    """
    return tvm.ir.register_op_attr(op_name, _LEGALIZE_ATTR_NAME, legal_func)
