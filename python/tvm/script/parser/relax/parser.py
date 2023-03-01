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
from ..core.parser import VarTable as return_var_table
from .entry import MatchCastPair, StructInfoProxy, TupleProxy


# An global list to record all exprs to return
return_expr_list = []


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

    with self.var_table.with_frame():
        with self.with_dispatch_token("relax"):
            with R.function():
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
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation is required for function parameters.")
            param_sinfo = eval_struct_info(self, arg.annotation, eval_str=True)
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
    print("Entering post_token_switch")
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
    """
    TODO (yongwww):
    issue 1): Save all values into a global list, and add into global_info in the end of parsing  -> Status: wip
          => we can just have a single api like add_return_global_info into the ReturnGlobalInfo,
             Solution: 
            [x]o1: Save all return values in a global list, and assembly it in the end of parsing,
                   don't allow user to provide it. Ignore if it exists
               o2: Create an IRModuleNode::GetGlobalInfo(String name), plus UpdateGlobalInfo should help do the modification
                   But how to expose it to parser? doesn't work, hard to expose to ir_builder
               o3: add ModuleGetGlobalInfos and ModuleUpdateGlobalInfos in src/script/ir_builder/ir/ir.cc
                   and python/tvm/script/ir_builder/ir/ir.py
                   how to reassembly the ReturnGlobalInfo is a problem, before the fetch returnGlobalInfo is a runtime.Object
                   seems there is no way to update it, so give up o3

                   Solution: expose get elements of ReturnGlobalInfo into IR-builder


    issue 2): global issue was required explicitly at the beggining of the ir_module,
              need to figure out a way to update/create a return global info at any point  -> Status: todo
              Solution: No matter if the tvmscript has explicitly feed the module_gloabl_info or not, and one for return!

    issue 3): need to hide the return global info, it shouldn't be visible to users,
              it might crash the exiting test cases -> Status: todo
              Solution: solution in 2) should help fix test cases, since we will have return_global_info anyway,
                the only concern is that the ordering of return_exprs, topological ordering for relax func parsing
                should fix it too. And it just potentially impact test structural_equal, no functionality impacted!
    
    Conclusion: 
        1) The best way is to add "Bool return_body" in SeqExpr, but we need to keep IR constrained at this moment
        2) Introduce func_info in relax function level, similar to global info, but it will introduce return_func_info
           into Function, and the IR is affected, then prefer option 1)
        So, I decided to move forward with GlobalInfo, because it is already there.
    """

    return_expr_list.append(value)
    print("Entering return visit")
    # use var_table to record the return exprs
    ginfos = I.module_get_global_infos()
    print("the current global info: ", ginfos)
    ret_ginfo = I.return_global_info(return_expr_list)
    # str "relax_return_exprs" was reserved as key for return exprs in global_info
    ginfos["return_exprs"] = [ret_ginfo]
    I.module_update_global_infos(ginfos)

    R.ret_value(value)  # TODO(yongwww): probably we can remove R.ret_value as well


@dispatch.register(token="relax", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    if node.orelse is None:
        raise ValueError("Else statements are required for relax dialect.")
    with R.If(self.eval_expr(node.test)) as if_frame:
        with self.var_table.with_frame():
            with R.Then():
                print("Entering R.Then")
                self.visit_body(node.body)
        with self.var_table.with_frame():
            with R.Else():
                print("Entering R.Else")
                self.visit_body(node.orelse)
    self.var_table.add(if_frame.var_name, if_frame.var, allow_shadowing=True)
