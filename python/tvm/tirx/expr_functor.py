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
"""
TIR expression functors in Python.

This module implements the visitor and mutator patterns for TIR expressions.
"""

from collections.abc import Callable
from typing import TypeVar

import tvm
from tvm.ir import PrimExpr, Range
from tvm.tirx import IterVar

T = TypeVar("T")


def _visit_array(arr: list[T], callback: Callable[[T], None]) -> None:
    """Visit elements in an array using a callback function.

    Parameters
    ----------
    arr : List[T]
        The array to be visited
    callback : Callable[[T], None]
        The callback function
    """
    for item in arr:
        callback(item)


class ExprFunctor:
    """An abstract visitor over Expr, with visiting function defined for each Expr type."""

    def __init__(self):
        self._dispatch_map = {
            "tirx.Var": self.visit_var_,
            "tirx.SizeVar": self.visit_size_var_,
            "tirx.BufferLoad": self.visit_buffer_load_,
            "tirx.ProducerLoad": self.visit_producer_load_,
            "tirx.Let": self.visit_let_,
            "tirx.Call": self.visit_call_,
            "tirx.Add": self.visit_add_,
            "tirx.Sub": self.visit_sub_,
            "tirx.Mul": self.visit_mul_,
            "tirx.Div": self.visit_div_,
            "tirx.Mod": self.visit_mod_,
            "tirx.FloorDiv": self.visit_floordiv_,
            "tirx.FloorMod": self.visit_floormod_,
            "tirx.Min": self.visit_min_,
            "tirx.Max": self.visit_max_,
            "tirx.EQ": self.visit_eq_,
            "tirx.NE": self.visit_ne_,
            "tirx.LT": self.visit_lt_,
            "tirx.LE": self.visit_le_,
            "tirx.GT": self.visit_gt_,
            "tirx.GE": self.visit_ge_,
            "tirx.And": self.visit_and_,
            "tirx.Or": self.visit_or_,
            "tirx.Reduce": self.visit_reduce_,
            "tirx.Cast": self.visit_cast_,
            "tirx.Not": self.visit_not_,
            "tirx.Select": self.visit_select_,
            "tirx.Ramp": self.visit_ramp_,
            "tirx.Broadcast": self.visit_broadcast_,
            "tirx.Shuffle": self.visit_shuffle_,
            "tirx.IntImm": self.visit_int_imm_,
            "tirx.FloatImm": self.visit_float_imm_,
            "tirx.StringImm": self.visit_string_imm_,
        }

    def visit_expr(self, expr: PrimExpr):
        """Apply the visitor to an expression.

        Parameters
        ----------
        expr : PrimExpr
            The expression to be visited.

        Returns
        -------
        result : Any
            The result of the visit.
        """
        if expr is None:
            return None

        key = expr.__class__.__name__
        if key.endswith("Node"):
            key = key[:-4]  # Remove the "Node" suffix

        key = "tirx." + key
        if key in self._dispatch_map:
            return self._dispatch_map[key](expr)

        return self.visit_expr_default_(expr)

    def visit_var_(self, op):
        """Default visitor for Var node."""
        return None

    def visit_size_var_(self, op):
        """Default visitor for SizeVar node."""
        return self.visit_var_(op)

    def visit_buffer_load_(self, op):
        """Default visitor for BufferLoad node."""
        return self.visit_expr_default_(op)

    def visit_producer_load_(self, op):
        """Default visitor for ProducerLoad node."""
        return self.visit_expr_default_(op)

    def visit_let_(self, op):
        """Default visitor for Let node."""
        return self.visit_expr_default_(op)

    def visit_call_(self, op):
        """Default visitor for Call node."""
        return self.visit_expr_default_(op)

    def visit_add_(self, op):
        """Default visitor for Add node."""
        return self.visit_expr_default_(op)

    def visit_sub_(self, op):
        """Default visitor for Sub node."""
        return self.visit_expr_default_(op)

    def visit_mul_(self, op):
        """Default visitor for Mul node."""
        return self.visit_expr_default_(op)

    def visit_div_(self, op):
        """Default visitor for Div node."""
        return self.visit_expr_default_(op)

    def visit_mod_(self, op):
        """Default visitor for Mod node."""
        return self.visit_expr_default_(op)

    def visit_floordiv_(self, op):
        """Default visitor for FloorDiv node."""
        return self.visit_expr_default_(op)

    def visit_floormod_(self, op):
        """Default visitor for FloorMod node."""
        return self.visit_expr_default_(op)

    def visit_min_(self, op):
        """Default visitor for Min node."""
        return self.visit_expr_default_(op)

    def visit_max_(self, op):
        """Default visitor for Max node."""
        return self.visit_expr_default_(op)

    def visit_eq_(self, op):
        """Default visitor for EQ node."""
        return self.visit_expr_default_(op)

    def visit_ne_(self, op):
        """Default visitor for NE node."""
        return self.visit_expr_default_(op)

    def visit_lt_(self, op):
        """Default visitor for LT node."""
        return self.visit_expr_default_(op)

    def visit_le_(self, op):
        """Default visitor for LE node."""
        return self.visit_expr_default_(op)

    def visit_gt_(self, op):
        """Default visitor for GT node."""
        return self.visit_expr_default_(op)

    def visit_ge_(self, op):
        """Default visitor for GE node."""
        return self.visit_expr_default_(op)

    def visit_and_(self, op):
        """Default visitor for And node."""
        return self.visit_expr_default_(op)

    def visit_or_(self, op):
        """Default visitor for Or node."""
        return self.visit_expr_default_(op)

    def visit_reduce_(self, op):
        """Default visitor for Reduce node."""
        return self.visit_expr_default_(op)

    def visit_cast_(self, op):
        """Default visitor for Cast node."""
        return self.visit_expr_default_(op)

    def visit_not_(self, op):
        """Default visitor for Not node."""
        return self.visit_expr_default_(op)

    def visit_select_(self, op):
        """Default visitor for Select node."""
        return self.visit_expr_default_(op)

    def visit_ramp_(self, op):
        """Default visitor for Ramp node."""
        return self.visit_expr_default_(op)

    def visit_broadcast_(self, op):
        """Default visitor for Broadcast node."""
        return self.visit_expr_default_(op)

    def visit_shuffle_(self, op):
        """Default visitor for Shuffle node."""
        return self.visit_expr_default_(op)

    def visit_int_imm_(self, op):
        """Default visitor for IntImm node."""
        return self.visit_expr_default_(op)

    def visit_float_imm_(self, op):
        """Default visitor for FloatImm node."""
        return self.visit_expr_default_(op)

    def visit_string_imm_(self, op):
        """Default visitor for StringImm node."""
        return self.visit_expr_default_(op)

    def visit_expr_default_(self, op):
        """Default visitor implementation."""
        raise NotImplementedError(f"Do not have a default for {op.__class__.__name__}")

    def __call__(self, expr):
        """Call visitor on expression.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        Returns
        -------
        result : Any
            The result of visiting.
        """
        return self.visit_expr(expr)


class ExprVisitor(ExprFunctor):
    """A visitor over Expr.

    This is a visitor that recursively traverses an expression. Subclasses can
    override the visit methods to customize the behavior.
    """

    def visit_var_(self, op):
        """Visitor implementation for Var."""
        pass

    def visit_size_var_(self, op):
        """Visitor implementation for SizeVar."""
        self.visit_var_(op)

    def visit_buffer_load_(self, op):
        """Visitor implementation for BufferLoad."""

        def _visit_indices(index):
            self.visit_expr(index)

        _visit_array(op.indices, _visit_indices)

    def visit_producer_load_(self, op):
        """Visitor implementation for ProducerLoad."""

        def _visit_indices(index):
            self.visit_expr(index)

        _visit_array(op.indices, _visit_indices)

    def visit_let_(self, op):
        """Visitor implementation for Let."""
        self.visit_expr(op.value)
        self.visit_expr(op.body)

    def visit_call_(self, op):
        """Visitor implementation for Call."""

        def _visit_arg(arg):
            self.visit_expr(arg)

        _visit_array(op.args, _visit_arg)

    def _visit_binary_op(self, op):
        """Helper to visit binary operators."""
        self.visit_expr(op.a)
        self.visit_expr(op.b)

    def visit_add_(self, op):
        """Visitor implementation for Add."""
        self._visit_binary_op(op)

    def visit_sub_(self, op):
        """Visitor implementation for Sub."""
        self._visit_binary_op(op)

    def visit_mul_(self, op):
        """Visitor implementation for Mul."""
        self._visit_binary_op(op)

    def visit_div_(self, op):
        """Visitor implementation for Div."""
        self._visit_binary_op(op)

    def visit_mod_(self, op):
        """Visitor implementation for Mod."""
        self._visit_binary_op(op)

    def visit_floordiv_(self, op):
        """Visitor implementation for FloorDiv."""
        self._visit_binary_op(op)

    def visit_floormod_(self, op):
        """Visitor implementation for FloorMod."""
        self._visit_binary_op(op)

    def visit_min_(self, op):
        """Visitor implementation for Min."""
        self._visit_binary_op(op)

    def visit_max_(self, op):
        """Visitor implementation for Max."""
        self._visit_binary_op(op)

    def visit_eq_(self, op):
        """Visitor implementation for EQ."""
        self._visit_binary_op(op)

    def visit_ne_(self, op):
        """Visitor implementation for NE."""
        self._visit_binary_op(op)

    def visit_lt_(self, op):
        """Visitor implementation for LT."""
        self._visit_binary_op(op)

    def visit_le_(self, op):
        """Visitor implementation for LE."""
        self._visit_binary_op(op)

    def visit_gt_(self, op):
        """Visitor implementation for GT."""
        self._visit_binary_op(op)

    def visit_ge_(self, op):
        """Visitor implementation for GE."""
        self._visit_binary_op(op)

    def visit_and_(self, op):
        """Visitor implementation for And."""
        self._visit_binary_op(op)

    def visit_or_(self, op):
        """Visitor implementation for Or."""
        self._visit_binary_op(op)

    def visit_int_imm_(self, op):
        """Visitor implementation for IntImm."""
        pass

    def visit_float_imm_(self, op):
        """Visitor implementation for FloatImm."""
        pass

    def visit_string_imm_(self, op):
        """Visitor implementation for StringImm."""
        pass

    def visit_reduce_(self, op):
        """Visitor implementation for Reduce."""

        def _visit_iter_var(iv):
            self.visit_expr(iv.dom.min)
            self.visit_expr(iv.dom.extent)

        def _visit_source(source):
            self.visit_expr(source)

        _visit_array(op.axis, _visit_iter_var)
        _visit_array(op.source, _visit_source)

        if op.init:
            _visit_array(op.init, _visit_source)

        self.visit_expr(op.condition)

    def visit_cast_(self, op):
        """Visitor implementation for Cast."""
        self.visit_expr(op.value)

    def visit_not_(self, op):
        """Visitor implementation for Not."""
        self.visit_expr(op.a)

    def visit_select_(self, op):
        """Visitor implementation for Select."""
        self.visit_expr(op.condition)
        self.visit_expr(op.true_value)
        self.visit_expr(op.false_value)

    def visit_ramp_(self, op):
        """Visitor implementation for Ramp."""
        self.visit_expr(op.base)
        self.visit_expr(op.stride)
        self.visit_expr(op.lanes)

    def visit_shuffle_(self, op):
        """Visitor implementation for Shuffle."""

        def _visit_expr(expr):
            self.visit_expr(expr)

        _visit_array(op.indices, _visit_expr)
        _visit_array(op.vectors, _visit_expr)

    def visit_broadcast_(self, op):
        """Visitor implementation for Broadcast."""
        self.visit_expr(op.value)
        self.visit_expr(op.lanes)


class ExprMutator(ExprFunctor):
    """A mutator over Expr.

    This is a mutator that recursively transforms an expression. Subclasses can
    override the visit methods to customize the behavior.
    """

    def visit_var_(self, op):
        """Mutator implementation for Var."""
        return op

    def visit_size_var_(self, op):
        """Mutator implementation for SizeVar."""
        return self.visit_var_(op)

    def visit_buffer_load_(self, op):
        """Mutator implementation for BufferLoad."""
        indices = [self.visit_expr(index) for index in op.indices]

        if all(old_index is new_index for old_index, new_index in zip(op.indices, indices)):
            return op
        else:
            return tvm.tirx.BufferLoad(op.buffer, indices, op.predicate)

    def visit_producer_load_(self, op):
        """Mutator implementation for ProducerLoad."""
        indices = [self.visit_expr(index) for index in op.indices]

        if all(old_index is new_index for old_index, new_index in zip(op.indices, indices)):
            return op
        else:
            return tvm.tirx.ProducerLoad(op.producer, indices)

    def visit_let_(self, op):
        """Mutator implementation for Let."""
        var = self.visit_var_(op.var)
        value = self.visit_expr(op.value)
        body = self.visit_expr(op.body)

        if var is op.var and value is op.value and body is op.body:
            return op
        else:
            return tvm.tirx.Let(var, value, body)

    def visit_call_(self, op):
        """Mutator implementation for Call."""
        args = [self.visit_expr(arg) for arg in op.args]

        if all(old_arg is new_arg for old_arg, new_arg in zip(op.args, args)):
            return op
        else:
            return tvm.tirx.Call(op.dtype, op.op, args)

    def _mutate_binary_op(self, op_cls, op):
        """Helper to mutate binary operators."""
        a = self.visit_expr(op.a)
        b = self.visit_expr(op.b)

        if a is op.a and b is op.b:
            return op
        else:
            return op_cls(a, b)

    def visit_add_(self, op):
        """Mutator implementation for Add."""
        return self._mutate_binary_op(tvm.tirx.Add, op)

    def visit_sub_(self, op):
        """Mutator implementation for Sub."""
        return self._mutate_binary_op(tvm.tirx.Sub, op)

    def visit_mul_(self, op):
        """Mutator implementation for Mul."""
        return self._mutate_binary_op(tvm.tirx.Mul, op)

    def visit_div_(self, op):
        """Mutator implementation for Div."""
        return self._mutate_binary_op(tvm.tirx.Div, op)

    def visit_mod_(self, op):
        """Mutator implementation for Mod."""
        return self._mutate_binary_op(tvm.tirx.Mod, op)

    def visit_floordiv_(self, op):
        """Mutator implementation for FloorDiv."""
        return self._mutate_binary_op(tvm.tirx.FloorDiv, op)

    def visit_floormod_(self, op):
        """Mutator implementation for FloorMod."""
        return self._mutate_binary_op(tvm.tirx.FloorMod, op)

    def visit_min_(self, op):
        """Mutator implementation for Min."""
        return self._mutate_binary_op(tvm.tirx.Min, op)

    def visit_max_(self, op):
        """Mutator implementation for Max."""
        return self._mutate_binary_op(tvm.tirx.Max, op)

    def visit_eq_(self, op):
        """Mutator implementation for EQ."""
        return self._mutate_binary_op(tvm.tirx.EQ, op)

    def visit_ne_(self, op):
        """Mutator implementation for NE."""
        return self._mutate_binary_op(tvm.tirx.NE, op)

    def visit_lt_(self, op):
        """Mutator implementation for LT."""
        return self._mutate_binary_op(tvm.tirx.LT, op)

    def visit_le_(self, op):
        """Mutator implementation for LE."""
        return self._mutate_binary_op(tvm.tirx.LE, op)

    def visit_gt_(self, op):
        """Mutator implementation for GT."""
        return self._mutate_binary_op(tvm.tirx.GT, op)

    def visit_ge_(self, op):
        """Mutator implementation for GE."""
        return self._mutate_binary_op(tvm.tirx.GE, op)

    def visit_and_(self, op):
        """Mutator implementation for And."""
        return self._mutate_binary_op(tvm.tirx.And, op)

    def visit_or_(self, op):
        """Mutator implementation for Or."""
        return self._mutate_binary_op(tvm.tirx.Or, op)

    def visit_int_imm_(self, op):
        """Mutator implementation for IntImm."""
        return op

    def visit_float_imm_(self, op):
        """Mutator implementation for FloatImm."""
        return op

    def visit_string_imm_(self, op):
        """Mutator implementation for StringImm."""
        return op

    def visit_reduce_(self, op):
        """Mutator implementation for Reduce."""

        def _mutate_iter_var(iv):
            old_dom = iv.dom
            new_min = self.visit_expr(old_dom.min)
            new_extent = self.visit_expr(old_dom.extent)

            if new_min is old_dom.min and new_extent is old_dom.extent:
                return iv
            else:
                new_dom = Range.FromMinExtent(new_min, new_extent)
                return IterVar(new_dom, iv.var, iv.iter_type, iv.thread_tag)

        axis = [_mutate_iter_var(iv) for iv in op.axis]
        source = [self.visit_expr(e) for e in op.source]
        init = [self.visit_expr(e) for e in op.init] if op.init else []
        condition = self.visit_expr(op.condition)

        axis_unchanged = all(old_iv is new_iv for old_iv, new_iv in zip(op.axis, axis))
        source_unchanged = all(old_e is new_e for old_e, new_e in zip(op.source, source))
        init_unchanged = (
            True if not op.init else all(old_e is new_e for old_e, new_e in zip(op.init, init))
        )
        condition_unchanged = condition is op.condition

        if axis_unchanged and source_unchanged and init_unchanged and condition_unchanged:
            return op
        else:
            return tvm.tirx.Reduce(op.combiner, source, axis, condition, op.value_index, init)

    def visit_cast_(self, op):
        """Mutator implementation for Cast."""
        value = self.visit_expr(op.value)

        if value is op.value:
            return op
        else:
            return tvm.tirx.Cast(op.dtype, value)

    def visit_not_(self, op):
        """Mutator implementation for Not."""
        a = self.visit_expr(op.a)

        if a is op.a:
            return op
        else:
            return tvm.tirx.Not(a)

    def visit_select_(self, op):
        """Mutator implementation for Select."""
        condition = self.visit_expr(op.condition)
        true_value = self.visit_expr(op.true_value)
        false_value = self.visit_expr(op.false_value)

        if (
            condition is op.condition
            and true_value is op.true_value
            and false_value is op.false_value
        ):
            return op
        else:
            return tvm.tirx.Select(condition, true_value, false_value)

    def visit_ramp_(self, op):
        """Mutator implementation for Ramp."""
        base = self.visit_expr(op.base)
        stride = self.visit_expr(op.stride)
        lanes = self.visit_expr(op.lanes)

        if base is op.base and stride is op.stride and lanes is op.lanes:
            return op
        else:
            return tvm.tirx.Ramp(base, stride, lanes)

    def visit_broadcast_(self, op):
        """Mutator implementation for Broadcast."""
        value = self.visit_expr(op.value)
        lanes = self.visit_expr(op.lanes)

        if value is op.value and lanes is op.lanes:
            return op
        else:
            return tvm.tirx.Broadcast(value, lanes)

    def visit_shuffle_(self, op):
        """Mutator implementation for Shuffle."""
        vectors = [self.visit_expr(v) for v in op.vectors]

        vectors_unchanged = all(old_v is new_v for old_v, new_v in zip(op.vectors, vectors))

        if vectors_unchanged:
            return op
        else:
            return tvm.tirx.Shuffle(vectors, op.indices)
