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
# pylint: disable=missing-docstring

import functools
import numbers
from typing import Any, Optional

from tvm import relax, tir
from tvm.ir import structural_equal
from tvm.relax import StructInfo
from tvm.relax.utils import convert_to_expr
from tvm.script.ir_builder.relax.frame import BlockFrame

from ...ir_builder import ir as I
from ...ir_builder import relax as R
from ...ir_builder.base import IRBuilder
from .._core import Parser, dispatch, doc
from .entry import MatchCastPair, StructInfoProxy, TupleProxy


def bind_assign_value(
    self: Parser,
    node: doc.expr,
    var_name: str,
    value: Any,
    anno_sinfo: Optional[StructInfo] = None,
) -> Any:
    var_table = self.var_table.get()

    if isinstance(value, tir.Var):
        if value.name and var_name != value.name:
            self.report_error(
                node,
                "Cannot define TIR variables with different names. The LHS of binding should "
                "has the same name provided in RHS.",
            )
        if var_name in var_table:
            prev_value = var_table[var_name]
            if not isinstance(prev_value, tir.Var):
                self.report_error(
                    node,
                    "Cannot redefine a non-TIR-variable object to a TIR variable. Please "
                    "define the TIR variable with another name.",
                )
            if prev_value.dtype != value.dtype:
                self.report_error(
                    node,
                    "Expected the same dtype for TIR vars "
                    f"but got {value.dtype} vs {prev_value.dtype}",
                )
            return prev_value
        IRBuilder.name(var_name, value)
        return value

    if isinstance(value, tuple):
        value = convert_to_expr(value)
    if isinstance(value, numbers.Number):
        value = R.const(value)

    if isinstance(value, relax.Expr):
        var = R.emit(value, anno_sinfo)
    elif isinstance(value, MatchCastPair):
        if anno_sinfo is not None and not structural_equal(anno_sinfo, value.struct_info):
            self.report_error(
                node, "Cannot specify inconsistent annotation for a match cast pair. "
            )
        var = R.emit_match_cast(value.value, value.struct_info)
    else:
        raise TypeError(f"Unsupported type {type(value)} in assignment")

    IRBuilder.name(var_name, var)
    return var


def eval_struct_info_proxy(self: Parser, node: doc.expr) -> StructInfoProxy:
    try:
        annotation = self.eval_expr(node)
        if annotation is None:
            return TupleProxy([])
        if callable(annotation):
            annotation = annotation()
        if isinstance(annotation, StructInfoProxy):
            return annotation
        else:
            raise TypeError(f"Expected StructInfoProxy but got {type(annotation)}.")
    except Exception as err:
        self.report_error(node, str(err))
        raise err


def eval_struct_info(self: Parser, node: doc.expr, eval_str: bool = False) -> StructInfo:
    var_table = self.var_table.get() if eval_str else None
    try:
        return eval_struct_info_proxy(self, node).as_struct_info(var_table)
    except Exception as err:
        self.report_error(node, str(err))
        raise err


def collect_symbolic_var_from_params(self: Parser, node: doc.FunctionDef) -> None:
    # Collect symbolic vars from parameters
    symbolic_vars = set()
    for arg in node.args.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_sinfo_proxy = eval_struct_info_proxy(self, arg.annotation)
        symbolic_vars.update(param_sinfo_proxy.get_symbolic_vars())

    # Define symbolic vars to the current var_table frame
    for var_name in symbolic_vars:
        self.var_table.add(var_name, tir.Var(var_name, "int64"), allow_shadowing=False)


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        with self.with_dispatch_token("relax"):
            with R.function():
                R.func_name(node.name)
                collect_symbolic_var_from_params(self, node)

                if node.returns is not None:
                    ann_sinfo = eval_struct_info(self, node.returns, eval_str=True)
                    R.func_ret_struct_info(ann_sinfo)

                self.visit(node.args)
                self.visit_body(node.body)


@dispatch.register(token="relax", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> None:
    with self.var_table.with_frame():
        collect_symbolic_var_from_params(self, node)

        if node.returns is None:
            # Use ObjectStructInfo as unknown return type
            # NOTE: Cannot use VoidStructInfo here because the return type can be refined later.
            ret_sinfo = relax.ObjectStructInfo()
        else:
            ret_sinfo = eval_struct_info(self, node.returns, eval_str=True)
        params = []
        params_sinfo = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation is required for function parameters.")
            param_sinfo = eval_struct_info(self, arg.annotation, eval_str=True)
            params_sinfo.append(param_sinfo)
            params.append(relax.Var(arg.arg, param_sinfo))

    func_signature = relax.Function.create_empty(params, ret_sinfo)
    global_var = I.decl_function(node.name, func_signature)
    self.var_table.add(node.name, global_var)


@dispatch.register(token="relax", type_name="pre_token_switch")
def pre_token_switch(self: Parser, node: doc.Expr) -> None:  # pylint: disable=unused-argument
    ir_builder = IRBuilder()
    ir_builder.__enter__()


@dispatch.register(token="relax", type_name="post_token_switch")
def post_token_switch(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder.current()
    result = ir_builder.get()
    ir_builder.__exit__(None, None, None)
    var = R.emit(result)
    IRBuilder.name(node.name, var)
    self.var_table.add(node.name, var, allow_shadowing=False)


@dispatch.register(token="relax", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    value = self.eval_expr(node.value)
    if value is not None:
        self.report_error(node, f"Unsupported Expr stmt type {value}.")


@dispatch.register(token="relax", type_name="arguments")
def visit_arguments(self: Parser, node: doc.arguments) -> None:
    arg: doc.arg
    for arg in node.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_sinfo = eval_struct_info(self, arg.annotation, eval_str=True)
        param = R.arg(arg.arg, param_sinfo)

        self.var_table.add(arg.arg, param)


@dispatch.register(token="relax", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr) -> StructInfo:
    return eval_struct_info(self, node, eval_str=False)


@dispatch.register(token="relax", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    # Currently only `with R.dataflow()` is supported
    if len(node.items) != 1:
        self.report_error(node, "Only one item is allowed.")
    item = node.items[0]
    if item.optional_vars is not None:
        self.report_error(
            item.context_expr,
            "Relax syntax doesn't allow binding expressions in `with` to variables",
        )
    frame = self.eval_expr(item.context_expr)
    with self.var_table.with_frame():
        with frame:
            self.visit(node.body)
    if isinstance(frame, BlockFrame) and frame.is_dataflow:
        output_vars = frame.output_vars
        for var in output_vars:
            self.var_table.add(var.name_hint, var, allow_shadowing=True)


@dispatch.register(token="relax", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=bind_assign_value,
        allow_shadowing=True,
    )


@dispatch.register(token="relax", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    lhs = node.target
    rhs = self.eval_expr(node.value)
    anno_sinfo = self.visit_tvm_annotation(node.annotation)
    self.eval_assign(
        target=lhs,
        source=rhs,
        bind_value=functools.partial(bind_assign_value, anno_sinfo=anno_sinfo),
        allow_shadowing=True,
    )


@dispatch.register(token="relax", type_name="Return")
def visit_return(self: Parser, node: doc.Assign) -> None:
    value = self.eval_expr(node.value)
    value = convert_to_expr(value)
    R.func_ret_value(value)


@dispatch.register(token="relax", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    if node.orelse is None:
        raise ValueError("Else statements are required for relax dialect.")
    with R.If(self.eval_expr(node.test)) as if_frame:
        with self.var_table.with_frame():
            with R.Then():
                self.visit_body(node.body)
        with self.var_table.with_frame():
            with R.Else():
                self.visit_body(node.orelse)
    self.var_table.add(if_frame.var_name, if_frame.var, allow_shadowing=True)
