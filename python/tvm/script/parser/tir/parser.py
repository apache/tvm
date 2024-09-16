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
"""The base parser for tir"""

import contextlib
from functools import partial
from typing import Any

import tvm
from tvm.ir import GlobalVar, PrimType
from tvm.tir import Buffer, IterVar, PrimExpr, Var

from ...ir_builder import ir as I
from ...ir_builder import tir as T
from ...ir_builder.base import IRBuilder
from ...ir_builder.base import IRBuilderFrame as Frame
from .._core import Parser, dispatch, doc


def bind_with_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing with statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        with T.grid(128, 128, 18) as i, j, k.

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, (Buffer, Var)):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in with statement")
        raise NotImplementedError


def bind_for_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing for statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        for i, j, k in T.grid(128, 128, 128).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, (list, tuple, tvm.ir.Array)):
        for i, v in enumerate(value):
            bind_for_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Var):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in for statement")
        raise NotImplementedError


def bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing assign statement.
    e.g. binding vi, vj, vk with T.axis.remap("SSR", [i, j, k]), when parsing
        vi, vj, vk = T.axis.remap("SSR", [i, j, k]).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, T.meta_var):
        return value.value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            bind_assign_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        IRBuilder.name(var_name, res)
        return res
    elif isinstance(value, (Buffer, IterVar)) or (
        isinstance(value, Var) and not self.var_table.exist(value)
    ):
        IRBuilder.name(var_name, value)
        return value
    else:
        value = tvm.runtime.convert(value)
        frame = T.LetStmt(value)
        var = frame.var
        IRBuilder.name(var_name, var)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return var


def find_decorator_annotation(node: doc.FunctionDef, annotation: str, default: bool = True) -> bool:
    """
    Check the value of given annotation (argument name) in the prim_func decorator.
    Returns the value of the annotation if present, otherwise giving the default value.
    """
    # look for the named argument in the prim_func decorator
    for dec in node.decorator_list:
        if not isinstance(dec, doc.Call) or dec.func.attr != "prim_func":
            continue
        for keyword in dec.keywords:
            if keyword.arg == annotation:
                return keyword.value.value
    return default


@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    """The for visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.For
        The doc AST for node.
    """
    for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.frame.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of the following: "
            "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            self.eval_assign(target=node.target, source=iters, bind_value=bind_for_value)
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="While")
def visit_while(self: Parser, node: doc.While) -> None:
    """The while visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.While
        The doc AST while node.
    """
    with self.var_table.with_frame():
        cond = self.eval_expr(node.test)
        with T.While(cond):
            self.visit_body(node.body)


@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    """The assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assign
        The doc AST assign node.
    """
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]

    if isinstance(node.value, doc.Subscript):
        check_slices = []
        if isinstance(node.value.slice, doc.Slice):
            check_slices = [node.value.slice]
        elif isinstance(node.value.slice, doc.Tuple):
            for p in node.value.slice.elts:
                if isinstance(p, doc.Slice):
                    check_slices.append(p)
        for s in check_slices:
            if not s.step and s.upper and s.lower:
                s.step = doc.Constant(
                    1,
                    None,
                    1,
                    1,
                    s.upper.lineno,
                    s.upper.end_col_offset + 1,
                    s.upper.lineno,
                    s.upper.end_col_offset + 2,
                )

    rhs = self.eval_expr(node.value)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = self.eval_expr(lhs.slice)
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AugAssign")
def visit_aug_assign(self: Parser, node: doc.AugAssign) -> None:
    """The augmented assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AugAssign
        The doc AST augmented assign node.
    """
    lhs_pos = (
        node.target.lineno,
        node.target.col_offset,
        node.target.end_lineno,
        node.target.end_col_offset,
    )
    rhs_pos = (
        node.value.lineno,
        node.value.col_offset,
        node.value.end_lineno,
        node.value.end_col_offset,
    )
    node.target.ctx = doc.Load(*lhs_pos)
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(*lhs_pos), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(*rhs_pos), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)
    lhs = node.target
    lhs.ctx = doc.Store(*lhs_pos)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tir", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    """The annotated assign visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AnnAssign
        The doc AST annotated assign node.
    """
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_var = self.visit_tvm_annotation(node.annotation)
    if not isinstance(ann_var, Var):
        self.report_error(node.annotation, "Annotation should be Var")
    self.eval_assign(target=lhs, source=ann_var, bind_value=bind_assign_value)
    frame = T.LetStmt(rhs, var=ann_var)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    """The with visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.With
        The doc AST with node.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, Frame):
                self.report_error(
                    item.context_expr, "Invalid context expression in the with-statement."
                )
            rhs = stack.enter_context(frame)
            if item.optional_vars is not None:
                self.eval_assign(target=item.optional_vars, source=rhs, bind_value=bind_with_value)
        self.visit_body(node.body)


@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    """The function definition visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.FunctionDef
        The doc AST function definition node.
    """
    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})
    privacy = find_decorator_annotation(node, "private", default=False)
    self.function_annotations = None
    with self.var_table.with_frame():
        self.var_table.add("range", T.serial)
        with T.prim_func(is_private=privacy):
            T.func_name(node.name)
            if node.returns is not None:
                ret_type = self.eval_expr(node.returns)
                if callable(ret_type):
                    ret_type = PrimType(ret_type().dtype)
                T.func_ret(ret_type)
            with self.with_dispatch_token("tir"):
                # TODO: handle different types of arguments:
                # - vararg: arg | None
                # - kwonlyargs: list[arg]
                # - kw_defaults: list[expr | None]
                # - kwarg: arg | None
                # - defaults: list[expr]
                # - posonlyargs: list[arg]
                for arg in node.args.args:
                    if arg.annotation is None:
                        self.report_error(arg, "Type annotation required for function parameters.")
                    try:
                        ann = self.eval_expr(arg.annotation)
                        if callable(ann):
                            ann = ann()
                    except Exception:  # pylint: disable=broad-except
                        ann = func_annotation.get(arg.arg, None)
                        if ann is None:
                            raise
                    param = T.arg(arg.arg, ann)
                    self.var_table.add(arg.arg, param)
                self.visit_body(node.body)
    self.function_annotations = supplied_annotation


@dispatch.register(token="tir", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    """The TVM annotation visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.expr
        The doc AST expr node.
    """
    annotation = self.eval_expr(node)
    if callable(annotation):
        annotation = annotation()
    return annotation


@dispatch.register(token="tir", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    """The expr statement visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Expr
        The doc AST Expr node.
    """

    res = self.eval_expr(node.value)
    if res is None:
        pass
    elif isinstance(res, Frame):
        res.add_callback(partial(res.__exit__, None, None, None))
        res.__enter__()
    elif isinstance(res, PrimExpr):
        T.evaluate(res)
    elif isinstance(res, (int, bool)):
        T.evaluate(tvm.tir.const(res))
    elif isinstance(res, (tvm.relay.Call, tvm.relax.Call)) and not res.args:
        # Using GlobalVar.__call__ with no arguments is ambiguous, as
        # each IR has a different function Call representation.  If
        # this occurs, convert to the TIR representation.
        T.evaluate(tvm.tir.call_tir(res.op))
    elif isinstance(res, str):
        # Ignore docstrings
        pass
    elif isinstance(res, tvm.tir.stmt.BufferStore):
        T.buffer_store(res.buffer, res.value, res.indices, res.predicate)
    else:
        self.report_error(node, f"Parsing resulted in unexpected type {type(res)}")


@dispatch.register(token="tir", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    """The if visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.If
        The doc AST if node.
    """
    with self.var_table.with_frame():
        predicate = self.eval_expr(node.test)
        if isinstance(predicate, (PrimExpr, tvm.tir.expr.ExprOp)):
            with T.If(self.eval_expr(node.test)):
                with T.Then():
                    with self.var_table.with_frame():
                        self.visit_body(node.body)
                if node.orelse:
                    with T.Else():
                        with self.var_table.with_frame():
                            self.visit_body(node.orelse)
        elif isinstance(predicate, bool):
            if predicate:
                with self.var_table.with_frame():
                    self.visit_body(node.body)
            elif node.orelse:
                with self.var_table.with_frame():
                    self.visit_body(node.orelse)
        else:
            self.report_error(
                node.test, f"If condition must be a boolean expression, but got {predicate}"
            )


@dispatch.register(token="tir", type_name="Assert")
def visit_assert(self: Parser, node: doc.Assert) -> None:
    """The assert visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assert
        The doc AST assert node.
    """
    cond = self.eval_expr(node.test)
    msg = self.eval_expr(node.msg)
    frame = T.Assert(cond, msg)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tir", type_name="Return")
def visit_return(self: Parser, node: doc.Return) -> None:
    """The return visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """
    value = self.eval_expr(node.value)
    if value is None:
        self.report_error(node, "Expression to be returned must be a PrimExpr")
    T.evaluate(tvm.tir.ret(value))


@dispatch.register(token="tir", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> GlobalVar:
    """The function declaration step for tir

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """

    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})

    ret_type = None
    with self.var_table.with_frame():
        if node.returns is not None:
            ret_type = self.eval_expr(node.returns)
            if callable(ret_type):
                ret_type = PrimType(ret_type().dtype)

        arg_annotations = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation required for function parameters.")
            try:
                ann = self.eval_expr(arg.annotation)
                if callable(ann):
                    ann = ann()
            except Exception:  # pylint: disable=broad-except
                ann = func_annotation.get(arg.arg, None)
                if ann is None:
                    raise

            IRBuilder.name(arg.arg, ann)
            arg_annotations.append(ann)

    func_signature = tvm.tir.PrimFunc(arg_annotations, None, ret_type=ret_type)
    return I.decl_function(node.name, func_signature)
