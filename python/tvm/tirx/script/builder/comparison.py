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
"""Primitive comparison construction and shared comparison-chain bindings."""

from tvm import ir, tirx
from tvm.ir.prim import _ffi_api
from tvm.script.ir_builder.base import source_span


def _operand(value):
    return value.var if isinstance(value, tirx.IterVar) else value


def _comparison_chain(comparisons, operands, conjunction, bind):
    """Bind original chain operands progressively around short-circuit tests."""
    import tvm_ffi

    if len(operands) != len(comparisons) + 1:
        raise ValueError("A comparison chain requires one more operand than comparison")
    replacements = []
    bindings = []
    for operand in operands:
        if isinstance(operand, tvm_ffi.ObjectConvertible):
            operand = operand.asobject()
        if not isinstance(operand, ir.Expr) or isinstance(operand, ir.Var):
            bindings.append(None)
            continue
        previous = next((var for value, var in replacements if value.same_as(operand)), None)
        if previous is not None:
            bindings.append(None)
            continue
        variable = ir.Var("chain_operand", operand.ty)
        replacements.append((operand, variable))
        bindings.append((variable, operand))

    def replace(value, mutator):
        for original, variable in replacements:
            if value.same_as(original):
                return variable
        return mutator.default_mutate(value)

    conditions = []
    for comparison in comparisons:
        if isinstance(comparison, tvm_ffi.ObjectConvertible):
            comparison = comparison.asobject()
        conditions.append(
            tvm_ffi.structural_mutate(comparison, [(ir.Expr, replace)])
            if isinstance(comparison, ir.Expr)
            else comparison
        )
    result = conditions[-1]
    for index in range(len(conditions) - 1, -1, -1):
        if index < len(conditions) - 1:
            result = conjunction(conditions[index], result)
        if bindings[index + 1] is not None:
            variable, value = bindings[index + 1]
            result = bind(variable, value, result)
    if bindings[0] is not None:
        variable, value = bindings[0]
        result = bind(variable, value, result)
    return result


def lt(lhs, rhs, *, span=None):
    """Construct a primitive less-than comparison in written operand order."""
    return _ffi_api._OpLT(_operand(lhs), _operand(rhs), source_span(span))


def le(lhs, rhs, *, span=None):
    """Construct a primitive less-than-or-equal comparison in written operand order."""
    return _ffi_api._OpLE(_operand(lhs), _operand(rhs), source_span(span))


def gt(lhs, rhs, *, span=None):
    """Construct a primitive greater-than comparison in written operand order."""
    return _ffi_api._OpGT(_operand(lhs), _operand(rhs), source_span(span))


def ge(lhs, rhs, *, span=None):
    """Construct a primitive greater-than-or-equal comparison in written operand order."""
    return _ffi_api._OpGE(_operand(lhs), _operand(rhs), source_span(span))


def eq(lhs, rhs, *, span=None):
    """Construct a primitive equality comparison in written operand order."""
    return _ffi_api._OpEQ(_operand(lhs), _operand(rhs), source_span(span))


def ne(lhs, rhs, *, span=None):
    """Construct a primitive inequality comparison in written operand order."""
    return _ffi_api._OpNE(_operand(lhs), _operand(rhs), source_span(span))
