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
"""tvm.contrib.msc.core.utils.expr"""

import copy
from typing import Dict, List

import tvm
from tvm import relax
from tvm.relax import PyExprVisitor
from tvm.contrib.msc.core import _ffi_api


def legalize_expr_name(name: str, symbols: List[str] = None, dst: str = "_") -> str:
    """Legalize expr name

    Parameters
    ----------
    name: str
        The source name.
    symbols: list<str>
        The symbols to be replaced.
    dst: str
        The symbol for replace.

    Returns
    -------
    name: str
        The legialized name.
    """

    symbols = symbols or ["::", "/", "."]
    for sym in symbols:
        name = name.replace(sym, dst)
    return name.strip(dst)


def get_expr_name(expr: relax.Expr) -> str:
    """Get name hint for expr

    Parameters
    ----------
    expr: Expr
        The Expr of relax.

    Returns
    -------
    name: str
        The name_hint of expr
    """

    name = _ffi_api.SpanGetAttr(expr.span, _ffi_api.ToAttrKey("name"))
    if not name and isinstance(expr, relax.Var):
        return expr.name_hint
    return name


def make_span(kwargs: Dict[str, str], span: relax.Span = None) -> relax.Span:
    """Make a span from kwargs

    Parameters
    ----------
    kwargs: dict<str, str>
        The attrs in span.
    span: relax.Span
        The source span.

    Returns
    -------
    span: relax.Span
        The span.
    """

    span = span or relax.Span(tvm.ir.SourceName(""), 0, 0, 0, 0)
    for k, v in kwargs.items():
        span = _ffi_api.SpanSetAttr(span, _ffi_api.ToAttrKey(k), v)
    return span


def set_expr_name(expr: relax.Expr, name: str):
    """Set the name for expr

    Parameters
    ----------
    expr: Expr
        The Expr of relax.
    name: str
        The name.

    Returns
    -------
    expr: Expr
        The expr with name.
    """

    expr.span = make_span({"name": name}, expr.span)
    return expr


def get_expr_layout(expr: relax.Expr) -> str:
    """Get layout for expr

    Parameters
    ----------
    expr: Expr
        The Expr of relax.

    Returns
    -------
    layout: str
        The layout of expr
    """

    return _ffi_api.SpanGetAttr(expr.span, _ffi_api.ToAttrKey("layout"))


def get_span_attrs(mod: tvm.IRModule) -> dict:
    """Extract the span attributes from relax.Function.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.

    Returns
    -------
    attrs: dict
    """

    @relax.expr_functor.visitor
    class SpanVisitor(PyExprVisitor):
        """Visitor for get attributes in span"""

        def extract(self, expr: relax.Expr) -> dict:
            self._span_info = {}
            self._local_funcs = {}
            if isinstance(expr, relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, relax.BindingBlock):
                self.visit_binding_block(expr)
            return self._span_info

        def _update_attrs(self, expr: relax.Expr, name: str = "") -> None:
            if not expr.span:
                return
            name = name or get_expr_name(expr)
            if not name:
                return
            self._span_info[name] = dict(_ffi_api.SpanGetAttrs(expr.span))

        def visit_var_binding_(self, binding: relax.VarBinding) -> None:
            if isinstance(binding.value, relax.expr.Function):
                self._local_funcs[binding.var] = binding.value
            elif (
                isinstance(binding.value, relax.expr.Call) and binding.value.op in self._local_funcs
            ):
                cache_info = copy.deepcopy(self._span_info)
                func_info = self.extract(self._local_funcs[binding.value.op])
                self._span_info = cache_info
                self._span_info[binding.value.op.name_hint] = func_info
            else:
                super().visit_var_binding_(binding)
                self._update_attrs(binding.value, binding.var.name_hint)

        def visit_constant_(self, op: relax.Constant) -> None:
            super().visit_constant_(op)
            self._update_attrs(op)

        def visit_var_(self, op: relax.Var) -> None:
            super().visit_var_(op)
            self._update_attrs(op, op.name_hint)

    return {v.name_hint: SpanVisitor().extract(mod[v]) for v in mod.functions}


def msc_script(mod: tvm.IRModule, script: str = "") -> str:
    """Add span attrs after lines.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    script: string
        The script to be replaced

    Returns
    -------
    script: string
        The replaced script
    """

    script = script or str(mod)
    attrs = get_span_attrs(mod)
    cur_attr, lines = {}, []
    for line in script.split("\n"):
        if line.strip().startswith("def "):
            func_name = line.strip().split("def ")[1].split("(")[0]
            cur_attr = attrs.get(func_name, {})
        if ": " in line:
            v_name = line.strip().split(": ")[0]
            if v_name in cur_attr:
                line += (
                    " # "
                    + ", ".join(["{}={}".format(k, v) for k, v in cur_attr[v_name].items()])
                    + " #"
                )
        lines.append(line)
    return "\n".join(lines)
