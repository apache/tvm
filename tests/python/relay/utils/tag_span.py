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
import tvm
from tvm import relay, tir
from tvm.relay import expr as _expr
from tvm.relay.expr_functor import ExprVisitor


def _set_span(expr, src):
    if isinstance(expr, _expr.Call):
        return _expr.CallWithFields(
            expr, expr.op, expr.args, expr.attrs, expr.type_args, None, _create_span(src)
        )
    elif isinstance(expr, _expr.Var):
        return _expr.VarWithFields(expr, expr.vid, expr.type_annotation, None, _create_span(src))
    elif isinstance(expr, _expr.TupleGetItem):
        return _expr.TupleGetItemWithFields(
            expr, expr.tuple_value, expr.index, None, _create_span(src)
        )
    elif isinstance(expr, _expr.Constant):
        return _expr.ConstantWithFields(expr, expr.data, None, _create_span(src))
    elif isinstance(expr, _expr.Tuple):
        return _expr.TupleWithFields(expr, expr.fields, None, _create_span(src))
    elif isinstance(expr, _expr.TupleWrapper):
        return _expr.TupleWrapper(_set_span(expr.tuple_value, src), expr.size)

    assert False, f"unsupported type {type(expr)}"


def _create_span(src):
    if isinstance(src, list):
        tmp_list = []
        for s in src:
            if isinstance(s, str):
                tmp_list.append(_create_span(s))
            elif isinstance(s, relay.Span):
                tmp_list.append(s)
            elif isinstance(s, relay.SequentialSpan):
                tmp_list.extend(s.spans)
            elif s is None:
                tmp_list.append(s)
            else:
                assert False, f"unsupported type {type(s)}"
        return relay.SequentialSpan(tmp_list)
    return relay.Span(relay.SourceName(src), 0, 0, 0, 0)


def _collect_spans(objref):
    class Collector:
        def __init__(self):
            self._spans = []

        def collect(self, objref):
            if hasattr(objref, "span"):
                self._spans.append(objref.span)

        @property
        def get_spans(self):
            return self._spans

    pov = None
    if isinstance(objref, relay.Expr):
        pov = relay.analysis.post_order_visit
    elif isinstance(objref, (tir.Stmt, tir.expr.PrimExprWithOp)):
        pov = tir.stmt_functor.post_order_visit
    else:
        assert False, f"unsupported type {type(objref)}"

    c = Collector()
    pov(objref, c.collect)
    return c.get_spans


def _verify_span(lhs, rhs):
    lhs_spans, rhs_spans = _collect_spans(lhs), _collect_spans(rhs)

    assert len(lhs_spans) == len(rhs_spans)

    for i in range(len(lhs_spans)):
        tvm.ir.assert_structural_equal(lhs_spans[i], rhs_spans[i])


def _verify_structural_equal_with_span(lhs, rhs, assert_mode=False, map_free_vars=False):
    if isinstance(lhs, relay.Var) and isinstance(rhs, relay.Var):
        # SEqualReduce compares the vid of Var type. Threrfore we only compare span here.
        _verify_span(lhs, rhs)
        return

    if assert_mode:
        tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars)
    else:
        tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars)

    _verify_span(lhs, rhs)
