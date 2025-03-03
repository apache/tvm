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
# pylint: disable=missing-docstring, unused-argument

import functools
import numbers
from typing import Any, Dict, Optional

from tvm import relax, tir
from tvm.ir import GlobalVar, structural_equal
from tvm.relax import Expr, StructInfo
from tvm.relax.utils import convert_to_expr
from tvm.script.ir_builder.relax.frame import BlockFrame

from ...ir_builder import ir as I
from ...ir_builder import relax as R
from ...ir_builder.base import IRBuilder
from .._core import Parser, dispatch, doc
from .entry import (
    MatchCastPair,
    StructInfoProxy,
    _normalize_struct_info_proxy,
    _normalize_struct_info,
)


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
            if not isinstance(value, type(prev_value)):
                self.report_error(
                    node,
                    f"Expected the same IR type for TIR vars "
                    f"but existing value {type(value)} is mismatched "
                    f"to previous {type(prev_value)}",
                )
            value = prev_value
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
        return value
        # raise TypeError(f"Unsupported type {type(value)} in assignment")

    IRBuilder.name(var_name, var)
    return var


def eval_struct_info_proxy(self: Parser, node: doc.expr) -> StructInfoProxy:
    try:
        annotation = self.eval_expr(node)
        return _normalize_struct_info_proxy(annotation)
    except Exception as err:  # pylint: disable=broad-except
        self.report_error(node, err)
        raise


def eval_struct_info(self: Parser, node: doc.expr, eval_str: bool = False) -> StructInfo:
    var_table = self.var_table.get() if eval_str else None
    try:
        struct_info = self.eval_expr(node)
        return _normalize_struct_info(struct_info, var_table)
    except Exception as err:  # pylint: disable=broad-except
        self.report_error(node, err)
        raise


def is_called(node: Any, func_name: str) -> bool:
    # Check if it calls into a func
    if isinstance(node, doc.Call):
        # Recursive call was found
        if isinstance(node.func, doc.Name) and node.func.id == func_name:
            return True
    elif isinstance(node, (list, tuple)):
        for stmt in node:
            if is_called(stmt, func_name):
                return True
    elif isinstance(node, (doc.AnnAssign, doc.Assign, doc.Return, doc.Expr)):
        return is_called(node.value, func_name)
    elif isinstance(node, doc.With):
        return is_called(node.body, func_name)
    elif isinstance(node, doc.If):
        smts = []
        if node.body is not None:
            smts = smts + list(node.body)
        if node.orelse is not None:
            smts = smts + list(node.orelse)
        return is_called(smts, func_name)
    return False


def is_recursive(node: doc.FunctionDef) -> bool:
    # Check if it is a recursive function
    for stmt in node.body:
        if is_called(stmt, node.name):
            return True
    return False


def collect_symbolic_var_from_prelude(
    self: Parser, node: doc.FunctionDef, symbolic_vars: Dict[str, tir.Var]
) -> Dict[str, tir.Var]:
    prelude_vars = {}
    for stmt in node.body:
        if isinstance(stmt, doc.Assign) and all(
            isinstance(target, doc.Name) and target.id in symbolic_vars for target in stmt.targets
        ):
            values = self.eval_expr(stmt.value)

            try:
                iter(values)
            except TypeError:
                values = [values]

            assert len(stmt.targets) == len(values)
            for target, value in zip(stmt.targets, values):
                name = target.id
                prelude_vars[name] = value

    return {**symbolic_vars, **prelude_vars}


def collect_symbolic_var_from_params(self: Parser, node: doc.FunctionDef) -> None:
    # Collect symbolic vars from parameters
    symbolic_vars = {}
    for arg in node.args.args:
        if arg.annotation is None:
            self.report_error(arg, "Type annotation is required for function parameters.")
        param_sinfo_proxy = eval_struct_info_proxy(self, arg.annotation)

        for var_name in param_sinfo_proxy.get_symbolic_vars():
            if var_name not in symbolic_vars:
                symbolic_vars[var_name] = tir.Var(var_name, "int64")

    # Update symbolic vars based on
    symbolic_vars = collect_symbolic_var_from_prelude(self, node, symbolic_vars)

    # Define symbolic vars to the current var_table frame
    for var_name, var in symbolic_vars.items():
        self.var_table.add(var_name, var, allow_shadowing=False)


@dispatch.register(token="relax", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    is_inner_function = self.inside_function
    self.inside_function = True

    # reserve a var for local function
    func_val = self.var_table.get().get(node.name)
    if not func_val and is_recursive(node):
        collect_symbolic_var_from_params(self, node)
        if node.returns is None:
            ret_sinfo = relax.TupleStructInfo([])
        else:
            ret_sinfo = eval_struct_info(self, node.returns, eval_str=True)
        params_sinfo = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation is required for function parameters.")
            param_sinfo = eval_struct_info(self, arg.annotation, eval_str=True)
            params_sinfo.append(param_sinfo)
        # created a var for the local function, the same var could be used for recursive call
        local_func_var = relax.Var(node.name, relax.FuncStructInfo(params_sinfo, ret_sinfo))
        self.var_table.add(node.name, local_func_var)

    purity = find_decorator_annotation(node, "pure")
    # treat the function as private if we are inside another function
    # or if it has a privacy annotation
    privacy = is_inner_function or find_decorator_annotation(node, "private", default=False)

    with self.var_table.with_frame():
        with self.with_dispatch_token("relax"):
            with R.function(is_pure=purity, is_private=privacy):
                R.func_name(node.name)
                collect_symbolic_var_from_params(self, node)

                if node.returns is not None:
                    ann_sinfo = eval_struct_info(self, node.returns, eval_str=True)
                    R.func_ret_struct_info(ann_sinfo)

                self.visit(node.args)

                for stmt in node.body:
                    if isinstance(stmt, doc.FunctionDef):
                        if not stmt.decorator_list:
                            self.report_error(stmt, "Function must be decorated")
                        dec = self.eval_expr(stmt.decorator_list[-1])
                        # inline prim_func was found
                        if dec.dispatch_token == "tir":
                            self.report_error(stmt, "inline prim_func is disallowed in Relax IR")

                self.visit_body(node.body)
    self.inside_function = is_inner_function


def find_decorator_annotation(node: doc.FunctionDef, annotation: str, default: bool = True) -> bool:
    """
    Check the value of given annotation (argument name) in the function decorator.
    Returns the value of the annotation if present, otherwise giving the default value.
    """
    # look for the named argument in the function decorator
    for dec in node.decorator_list:
        if not isinstance(dec, doc.Call) or dec.func.attr != "function":
            continue
        for keyword in dec.keywords:
            if keyword.arg == annotation:
                return keyword.value.value
    return default


@dispatch.register(token="relax", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> GlobalVar:
    with self.var_table.with_frame():
        collect_symbolic_var_from_params(self, node)

        if node.returns is None:
            # Use ObjectStructInfo as unknown return type
            # NOTE: Cannot use VoidStructInfo here because the return type can be refined later.
            ret_sinfo = relax.ObjectStructInfo()
        else:
            ret_sinfo = eval_struct_info(self, node.returns, eval_str=True)
        params = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation is required for function parameters.")
            param_sinfo = eval_struct_info(self, arg.annotation, eval_str=True)
            params.append(relax.Var(arg.arg, param_sinfo))

    is_pure = find_decorator_annotation(node, "pure")

    func_signature = relax.Function.create_empty(params, ret_sinfo, is_pure=is_pure)
    return I.decl_function(node.name, func_signature)


@dispatch.register(token="relax", type_name="pre_visit_local_function")
def pre_visit_local_function(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder()
    ir_builder.__enter__()


@dispatch.register(token="relax", type_name="post_visit_local_function")
def post_visit_local_function(self: Parser, node: doc.Expr) -> None:
    ir_builder = IRBuilder.current()
    result = ir_builder.get()
    ir_builder.__exit__(None, None, None)
    # reuse var if it is reserved
    reserved_var = self.var_table.get().get(node.name)
    if reserved_var:
        var = R.emit_var_binding(relax.VarBinding(reserved_var, result))
    else:
        var = R.emit(result)
    IRBuilder.name(node.name, var)
    self.var_table.add(node.name, var, allow_shadowing=False)


@dispatch.register(token="relax", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    value = self.eval_expr(node.value)
    if isinstance(value, relax.Expr):
        var = R.emit(value)
        IRBuilder.name("_", var)
        is_void_value = (
            isinstance(var.struct_info, relax.TupleStructInfo) and len(var.struct_info.fields) == 0
        )

        if not is_void_value:
            self.report_error(
                node,
                f"Non-void relax expressions must be bound to a variable, "
                f"but expression of type {var.struct_info} was used as a statement.",
            )

    elif value is not None:
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


@dispatch.register(token="relax", type_name="enter_token")
def enter_token(self: Parser) -> Dict[str, Any]:
    def relax_call(self, *args) -> Expr:
        args = [convert_to_expr(arg) if isinstance(arg, tuple) else arg for arg in args]

        if all(isinstance(x, Expr) for x in args):
            return relax.Call(self, args)
        arg_types = [type(x) for x in args]
        raise RuntimeError(
            "Do not know how to handle GlobalVar.__call__ for types {}".format(arg_types)
        )

    context = {"GlobalVar.__call__": GlobalVar.__call__}
    GlobalVar.__call__ = relax_call
    return context


@dispatch.register(token="relax", type_name="exit_token")
def exit_token(self: Parser, context: Dict[str, Any]) -> None:
    assert "GlobalVar.__call__" in context
    GlobalVar.__call__ = context.get("GlobalVar.__call__")
