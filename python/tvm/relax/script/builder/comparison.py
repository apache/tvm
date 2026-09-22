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
"""Concrete Relax tensor and primitive comparison construction."""

import numbers

from tvm import ir, relax
from tvm.script.ir_builder.base import at
from tvm.tirx.script.builder import comparison as primitive


def _comparison(lhs, rhs, tensor_operation, primitive_operation, span):
    if any(isinstance(value, ir.Expr) and not ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = relax.const(lhs) if isinstance(lhs, numbers.Number) else lhs
        rhs = relax.const(rhs) if isinstance(rhs, numbers.Number) else rhs
        return at(span, tensor_operation(lhs, rhs))
    return primitive_operation(lhs, rhs, span=span)


def lt(lhs, rhs, *, span=None):
    """Construct a Relax less-than comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.less, primitive.lt, span)


def le(lhs, rhs, *, span=None):
    """Construct a Relax less-than-or-equal comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.less_equal, primitive.le, span)


def gt(lhs, rhs, *, span=None):
    """Construct a Relax greater-than comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.greater, primitive.gt, span)


def ge(lhs, rhs, *, span=None):
    """Construct a Relax greater-than-or-equal comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.greater_equal, primitive.ge, span)


def eq(lhs, rhs, *, span=None):
    """Construct a Relax equality comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.equal, primitive.eq, span)


def ne(lhs, rhs, *, span=None):
    """Construct a Relax inequality comparison in written operand order."""
    return _comparison(lhs, rhs, relax.op.not_equal, primitive.ne, span)
