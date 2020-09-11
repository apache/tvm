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
"""Hybrid Script Parser For TIR"""
# pylint: disable=invalid-name, missing-docstring, inconsistent-return-statements, no-else-return
# pylint: disable=unnecessary-comprehension, unused-argument, import-outside-toplevel
# pylint: disable=unused-import
import json
import numbers
import operator
from typed_ast import ast3 as ast

import tvm._ffi
from tvm import tir
from tvm._ffi.base import TVMError
from tvm.ir import GlobalVar
from tvm.tir import all as _all
from tvm.tir import expr as _expr

from . import scope_emitter, special_stmt, scope_handler, intrin
from .meta_unparser import MetaUnparser
from .registry import Registry


class HybridParserError(RuntimeError):
    """Hybrid Parser Runtime Error"""


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to TIR
    Notes for extension:
    1. To support new types of AST nodes. Add a function visit_xxx().
    2. To support new functions
        We divide allowed function calls in hybrid script into 3 categories,
        which is scope_handler, intrin and special_stmt.
        1) scope_handler: scope_handler functions correspond to StmtNodes without body, which can be
        further classified into 2 categories: with scope handler can for scope handlers
        2) intrin: intrin functions corresponds to the remaining IRNodes (StmtNodes without body,
        PrimExprNodes and more)
        3) special_stmt: special_stmt functions don't correspond to an IRNode in the AST directly.
        It is usually used for some information that is not suitable to be printed directly.
        When visiting With node, we check with_scope registry.
        When visiting For node, we check for_scope registry.
    """

    _binop_maker = {
        ast.Add: tir.Add,
        ast.Sub: tir.Sub,
        ast.Mult: tir.Mul,
        ast.Div: tir.Div,
        ast.FloorDiv: tir.FloorDiv,
        ast.Mod: tir.FloorMod,
        ast.BitOr: operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
        ast.Gt: tir.GT,
        ast.GtE: tir.GE,
        ast.Lt: tir.LT,
        ast.LtE: tir.LE,
        ast.Eq: tir.EQ,
        ast.NotEq: tir.NE,
        ast.And: tir.And,
        ast.Or: tir.Or,
    }

    _unaryop_maker = {ast.USub: operator.neg, ast.Invert: operator.invert, ast.Not: tir.Not}

    def __init__(self, src, base_lienno):
        self.params = None
        self.buffer_map = None
        self.dict_attr = None
        self.scope_emitter = None

        self.src = src.split("\n")
        self.base_lineno = base_lienno
        self.current_lineno = 0
        self.current_col_offset = 0
        self.meta = None

        self.functions = {}

        self._in_with_func_arg = False
        self._assign_target = None

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.params = []  # parameter list
        self.buffer_map = {}  # buffer map
        self.dict_attr = {}  # dict attr
        self.scope_emitter = scope_emitter.ScopeEmitter(self)  # scope emitter

    @staticmethod
    def is_meta(node):
        """Judge whether an AST node is META"""
        return (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__tvm_meta__"
        )

    def init_meta(self, meta_dict):
        if meta_dict is not None:
            self.meta = tvm.ir.load_json(json.dumps(meta_dict))

    def visit(self, node):
        """Override method in ast.NodeVisitor"""
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset

        if hasattr(node, "lineno"):
            self.current_lineno = self.base_lineno + node.lineno - 1
        if hasattr(node, "col_offset"):
            self.current_col_offset = node.col_offset

        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visit_res = visitor(node)

        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset

        return visit_res

    def wrap_line_col(self, message, lineno, col_offset):
        """Wrap the message with line number and column offset"""
        src_line = self.src[lineno - self.base_lineno]
        leading_space = len(src_line) - len(src_line.lstrip(" "))
        col_offset = col_offset - leading_space
        src_line = src_line[leading_space:]
        return (
            "\n  "
            + src_line
            + "\n  "
            + " " * col_offset
            + "^\n"
            + "ParserError in line "
            + str(lineno)
            + " : "
            + message
        )

    def report_error(self, message, lineno=None, col_offset=None):
        """Report an error occur in line lineno and column col_offset
        Parameters
        ----------
        message : str
            Error message
        lineno : int
            Line number of error line
        col_offset : int
            Column offset of error line
        """

        if lineno is None:
            lineno = self.current_lineno
        if col_offset is None:
            col_offset = self.current_col_offset
        raise HybridParserError(self.wrap_line_col(message, lineno, col_offset))

    def get_type_name(self, vtype):
        if (
            isinstance(vtype, ast.Attribute)
            and isinstance(vtype.value, ast.Name)
            and vtype.value.id == "ty"
        ):
            return vtype.attr
        self.report_error("invalid type annotation")

    def get_body(self):
        body = []
        while len(self.scope_emitter.node_stack[-1]) > 0:
            res = self.visit(self.scope_emitter.node_stack[-1].pop())
            if res is not None:
                body.append(res)
        return tvm.tir.SeqStmt(body) if len(body) > 1 else body[0]

    def parse_type(self, vtype):
        """ Parse type annotation AST into Type object """
        if isinstance(vtype, ast.NameConstant) and vtype.value is None:
            return tvm.ir.TupleType([])
        elif isinstance(vtype, ast.Attribute):
            return tvm.ir.PrimType(self.get_type_name(vtype))
        elif isinstance(vtype, ast.Subscript) and isinstance(vtype.slice, ast.Index):
            type_name = self.get_type_name(vtype.value)
            if isinstance(vtype.slice.value, ast.Tuple):
                args = [self.parse_type(element) for element in vtype.slice.value.elts]
            else:
                args = [self.parse_type(vtype.slice.value)]
            if type_name == "Ptr":
                return tvm.ir.PointerType(*args)
            elif type_name == "Tuple":
                return tvm.ir.TupleType(args)

        self.report_error("invalid type annotation")

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """

        self.report_error(type(node).__name__ + " stmt is not supported now")

    def visit_Module(self, node):
        """Module visitor
        AST abstract grammar:
            Module(stmt* body, type_ignore* type_ignore)
        By now we support two format of hybrid script shown below.

        Example
        -------
        1. Generate a Function(If the code is printed, then it may bring meta)
        .. code-block:: python

            import tvm

            @tvm.hybrid.script
            def A(...):
                ...

            # call hybrid parser when call this function, get a Function
            func = A

        2. Generate an IRModule
        .. code-block:: python

            import tvm

            @tvm.hybrid.script
            class MyMod():
               def A(...):
                  ...

               def B(...):
                   ...

                __tvm_meta__ = ...

            # call hybrid parser during construction, get an IRModule
            mod = MyMod()
        """

        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        elif len(node.body) == 2:
            if isinstance(node.body[0], ast.Assign):
                node.body[0], node.body[1] = node.body[1], node.body[0]
            if isinstance(node.body[0], ast.FunctionDef) and HybridParser.is_meta(node.body[1]):
                # function with meta
                self.init_meta(MetaUnparser().visit(node.body[1].value))
                return self.visit(node.body[0])
        self.report_error(
            "Only one-function, one-class or function-with-meta source code is allowed"
        )

    def visit_ClassDef(self, node):
        """ClassDef visitor
        AST abstract grammar:
            ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body,
                     expr* decorator_list)
        """

        # parse meta
        count = False
        for body_element in node.body:
            if isinstance(body_element, ast.FunctionDef):
                pass
            elif HybridParser.is_meta(body_element) and not count:
                count = True
                self.init_meta(MetaUnparser().visit(body_element.value))
            else:
                self.report_error("invalid class member")

        # parse member functions
        for body_element in node.body:
            if isinstance(body_element, ast.FunctionDef):
                self.visit(body_element)
        from .utils import create_module

        return create_module(self.functions)

    def visit_FunctionDef(self, node):
        """FunctionDef visitor
        AST abstract grammar:
            FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list,
                        expr? returns, string? type_comment)
            arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                         expr* kw_defaults, arg? kwarg, expr* defaults)
            arg = (identifier arg, expr? annotation, string? type_comment)
        """

        self.init_function_parsing_env()
        # add parameters of function
        for arg in node.args.args:
            arg_var = tvm.te.var(arg.arg, self.parse_type(arg.annotation))
            self.scope_emitter.update_symbol(arg.arg, arg_var)
            self.params.append(arg_var)

        # visit the body of function
        self.scope_emitter.node_stack[-1].extend(reversed(node.body))

        # fetch the body and return a tir.PrimFunc
        func = tvm.tir.PrimFunc(
            self.params,
            self.get_body(),
            ret_type=self.parse_type(node.returns),
            buffer_map=self.buffer_map,
            attrs=tvm.ir.make_node("DictAttrs", **self.dict_attr),
        )
        self.functions[GlobalVar(node.name)] = func
        return func

    def visit_Assign(self, node):
        """Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)
        By now only 2 types of Assign is supported:
            1. special stmts that appear as assign stmt
                1.1 Buffer = tir.buffer_bind()/tir.buffer_decl()
                1.2 Var = tir.var()
            2. (BufferStore) Buffer[PrimExpr, PrimExpr, ..., PrimExpr] = PrimExpr
            3. (Store) Var[PrimExpr] = PrimExpr
        """

        if not len(node.targets) == 1:
            self.report_error("Only one-valued assignment is supported now")
        target = node.targets[0]

        if isinstance(target, ast.Name):
            # scenario 1
            self._assign_target = target.id
            rhs = self.visit(node.value)
            if not isinstance(node.value, ast.Call):
                self.report_error("Unsupported Assign stmt")
            self.scope_emitter.update_symbol(target.id, rhs)
        elif isinstance(target, ast.Subscript):
            # scenario 2&3
            symbol, indexes = self.visit(target)
            self._assign_target = (symbol, indexes)
            rhs = self.visit(node.value)
            if isinstance(symbol, tvm.tir.Buffer):
                return tvm.tir.BufferStore(symbol, tvm.runtime.convert(rhs), indexes)
            else:
                if len(indexes) != 1:
                    self.report_error("Invalid Store stmt")
                return tvm.tir.Store(
                    symbol, tvm.runtime.convert(rhs), indexes[0], tvm.runtime.convert(True)
                )
        else:
            self.report_error("Unsupported Assign stmt")

    def visit_AnnAssign(self, node):
        """AnnAssign visitor
        AST abstract grammar:
            AnnAssign(expr target, expr annotation, expr? value, int simple)
        Corresponds to concise mode of with tir.let()
        """

        if isinstance(node.target, ast.Name):
            value = self.visit(node.value)
            var = tvm.te.var(node.target.id, self.parse_type(node.annotation))
            self.scope_emitter.update_symbol(var.name, var)
            return tvm.tir.LetStmt(var, value, self.visit(self.scope_emitter.node_stack[-1].pop()))
        else:
            self.report_error("Unsupported AnnAssign stmt")

    def visit_Assert(self, node):
        """Assert visitor
        AST abstract grammar:
            Assert(expr test, expr? msg)
        Corresponds to concise mode of with tir.assert()
        """

        condition = self.visit(node.test)
        if node.msg is None:
            self.report_error("Message of AssertStmt can't be None")
        message = self.visit(node.msg)
        return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), self.get_body())

    def visit_For(self, node):
        """For visitor
        AST abstract grammar:
            For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        By now only 1 type of For is supported:
            1. for name in tir.range(begin, end, for_type)
        """

        if not isinstance(node.target, ast.Name):
            self.report_error("The loop variable should be a name variable")
        # check node.iter, which is a tir Call
        if not isinstance(node.iter, ast.Call):
            self.report_error("The loop iter should be a Call")
        if (
            not isinstance(node.iter.func, ast.Attribute)
            or not isinstance(node.iter.func.value, ast.Name)
            or node.iter.func.value.id != "tir"
        ):
            self.report_error("The loop iter Call should be tir.name()")

        func_name = node.iter.func.attr
        # collect arguments
        args = [self.visit(arg) for arg in node.iter.args]
        kw_args = [self.visit(keyword) for keyword in node.iter.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        # All the functions supported in For stmt are registered in scope_handler.ForScope
        if func_name not in Registry.for_scope:
            self.report_error(
                "Function " + func_name + " used in For stmt is not supported now",
                self.current_lineno,
                node.iter.col_offset,
            )

        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + node.iter.lineno - 1,
            node.iter.col_offset,
        )
        res = Registry.for_scope.get(func_name)(self, node, args, kw_args)
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def visit_With(self, node):
        """With visitor
        AST abstract grammar:
            With(withitem* items, stmt* body, string? type_comment)
            withitem = (expr context_expr, expr? optional_vars)
        By now only 1 type of With is supported:
            1. with tir.let/tir.Assert()/tir.attr()/tir.allocate()/tir.realize()
        """

        if len(node.items) != 1:
            self.report_error("Only one with element is supported now")
        if not isinstance(node.items[0].context_expr, ast.Call):
            self.report_error("The context expression of with should be a Call")
        func_call = node.items[0].context_expr
        if (
            not isinstance(func_call.func, ast.Attribute)
            or not isinstance(func_call.func.value, ast.Name)
            or func_call.func.value.id != "tir"
        ):
            self.report_error("The context expression of with should be tir.name()")

        func_name = func_call.func.attr
        # collect arguments
        args = [self.visit(arg) for arg in func_call.args]
        kw_args = [self.visit(keyword) for keyword in func_call.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        if func_name not in Registry.with_scope:
            self.report_error("Function " + func_name + " used in With stmt is not supported now")

        # All the functions supported in With stmt are registered in scope_handler.WithScope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + func_call.lineno - 1,
            func_call.col_offset,
        )
        res = Registry.with_scope.get(func_name)(self, node, args, kw_args)
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def visit_If(self, node):
        """If visitor
        AST abstract grammar:
            If(expr test, stmt* body, stmt* orelse)
        """

        condition = self.visit(node.test)
        # then body
        self.scope_emitter.new_scope()
        self.scope_emitter.node_stack[-1].extend(reversed(node.body))
        then_body = self.get_body()
        self.scope_emitter.pop_scope()

        # else body
        if len(node.orelse) > 0:
            self.scope_emitter.new_scope()
            self.scope_emitter.node_stack[-1].extend(reversed(node.orelse))
            else_body = self.get_body()
            self.scope_emitter.pop_scope()
        else:
            else_body = None
        return tvm.tir.IfThenElse(condition, then_body, else_body)

    def visit_Call(self, node):
        """Call visitor
        AST abstract grammar:
            Call(expr func, expr* args, keyword* keywords)
            keyword = (identifier? arg, expr value)
        """

        # collect arguments
        args = [self.visit(arg) for arg in node.args]
        kw_args = [self.visit(keyword) for keyword in node.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        maybe_intrin = False
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "tir":
                func_name = node.func.attr
                maybe_intrin = True
            else:
                self.report_error("Unsupported Attribute typed function call")
        else:
            self.report_error("Unsupported function call")

        if func_name in Registry.special_stmt:
            return Registry.special_stmt.get(func_name)(self, node, args, kw_args)
        if func_name in Registry.intrin:
            return Registry.intrin.get(func_name)(self, node, args, kw_args)
        if func_name in Registry.with_scope:
            return Registry.with_scope.get(func_name)(self, node, args, kw_args)
        if maybe_intrin:
            return tvm.tir.Call(kw_args["dtype"], tvm.ir.op.Op.get("tir." + func_name), args)

        self.report_error("Function " + func_name + " is not supported now")

    def visit_Expr(self, node):
        """Expr visitor
        AST abstract grammar:
            Expr(expr value)

        Now only 2 types of Expr stmt is allowed:
            1. Concise mode of with scope handlers
                tir.attr()/tir.assert()/tir.allocate()/tir.realize()
            2. special stmts appear as a call
                tir.set_func_attr()
        """

        if not isinstance(node.value, ast.Call):
            self.report_error("Unsupported Expr stmt")
        return self.visit(node.value)

    def visit_BinOp(self, node):
        """BinOp visitor
        AST abstract grammar:
            BinOp(expr left, operator op, expr right)
        """

        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not isinstance(node.op, tuple(HybridParser._binop_maker.keys())):
            self.report_error("BinOp " + str(type(node.op)) + " is not supported now")
        return HybridParser._binop_maker[type(node.op)](lhs, rhs)

    def visit_Compare(self, node):
        """Compare visitor
        AST abstract grammar:
            Compare(expr left, expr right, ops=)
        """

        ops = [self.visit(node.left)]
        ops += [self.visit(comparator) for comparator in node.comparators]
        res = []
        for i in range(len(node.ops)):
            lhs = ops[i]
            rhs = ops[i + 1]
            res.append(HybridParser._binop_maker[type(node.ops[i])](lhs, rhs))
        return _all(*res)

    def visit_BoolOp(self, node):
        """BoolOp visitor
        AST abstract grammar:
            BoolOp(boolop op, expr* values)
        """

        values = [self.visit(value) for value in node.values]
        return HybridParser._binop_maker[type(node.op)](*values)

    def visit_UnaryOp(self, node):
        """UnaryOp visitor
        AST abstract grammar:
            UnaryOp(unaryop op, expr operand)
        """

        operand = self.visit(node.operand)
        if not isinstance(node.op, tuple(HybridParser._unaryop_maker.keys())):
            self.report_error("UnaryOp " + str(type(node.op)) + " is not supported now")
        return HybridParser._unaryop_maker[type(node.op)](operand)

    def visit_Subscript(self, node):
        """Subscript visitor
        AST abstract grammar:
            Subscript(expr value, slice slice, expr_context ctx)
            slice = Slice(expr? lower, expr? upper, expr? step)
                    | ExtSlice(slice* dims)
                    | Index(expr value)
        By now only 2 types of Subscript are supported:
            1. Buffer[index, index, ...], Buffer element access(BufferLoad & BufferStore)
               Var[index] Buffer element access()
            2. meta[type_key][index], Meta info access
        """

        if isinstance(node.value, (ast.Name, ast.Attribute)):
            symbol = self.visit(node.value)
            if isinstance(node.slice, ast.Index):
                # BufferLoad & BufferStore
                if isinstance(node.slice.value, ast.Tuple):
                    # Buffer/Var[index, index, ...]
                    indexes = [self.visit(element) for element in node.slice.value.elts]
                else:
                    # Buffer/Var[index]
                    indexes = [self.visit(node.slice.value)]
                if isinstance(node.ctx, ast.Load):
                    if isinstance(symbol, tir.expr.Var):
                        return tvm.tir.Load("float32", symbol, indexes, True)
                    else:
                        return tvm.tir.BufferLoad(symbol, indexes)
                else:
                    return symbol, indexes
            else:
                # Buffer Region, now used in tir.realize(buffer[bounds])
                doms = []
                slice_nodes = []
                if isinstance(node.slice, ast.Slice):
                    # Buffer[begin:end]
                    slice_nodes.append(node.slice)
                elif isinstance(node.slice, ast.ExtSlice):
                    # Buffer[begin:end, begin:end]
                    slice_nodes.extend(node.slice.dims)

                for dim in slice_nodes:
                    if not hasattr(dim, "step"):
                        self.report_error("slice of Buffer Region ought to be begin:end")
                    if dim.step is not None:
                        self.report_error("step is not allowed in Buffer Region")
                    upper = self.visit(dim.upper)
                    lower = self.visit(dim.lower)
                    extent = upper - lower
                    if isinstance(extent, _expr.PrimExpr):
                        ana = tvm.arith.Analyzer()
                        extent = ana.simplify(extent)
                    doms.append(tvm.ir.Range.from_min_extent(lower, extent))
                return symbol, doms

        elif (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "meta"
        ):
            # meta[type_key][index]
            if not (
                isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num)
            ) or not (
                isinstance(node.value.slice, ast.Index)
                and isinstance(node.value.slice.value, ast.Name)
            ):
                self.report_error("The meta access format ought to be meta[type_key][index]")
            type_key = node.value.slice.value.id
            index = node.slice.value.n
            node_list = self.meta[type_key]
            if node_list is None:
                self.report_error("type_key " + type_key + " in meta not found")
            if len(node_list) <= index:
                self.report_error("index " + index + " out of range " + len(node_list))
            return node_list[index]
        else:
            self.report_error("Only buffer variable and meta can be subscriptable")

    def visit_Name(self, node):
        """Name visitor
        AST abstract grammar:
            Name(identifier id, expr_context ctx)
        """

        name = node.id
        symbol = self.scope_emitter.lookup_symbol(name)
        if symbol is None:
            self.report_error("Unknown symbol %s" % name)
        return symbol

    def visit_Attribute(self, node):
        """Attribute visitor
        AST abstract grammar:
            Attribute(expr value, identifier attr, expr_context ctx)
        """

        if not isinstance(node.value, ast.Name):
            self.report_error("The value of Attribute ought to a Name")
        name = node.value.id
        symbol = self.scope_emitter.lookup_symbol(name)
        if symbol is None or not isinstance(symbol, tvm.tir.Buffer):
            self.report_error("Unsupported Attribute expression")
        if not hasattr(symbol, node.attr):
            self.report_error("Type " + type(symbol) + " has not attr " + node.attr)
        return getattr(symbol, node.attr)

    def visit_Dict(self, node):
        """Dict visitor
        AST abstract grammar:
            Dict(expr* keys, expr* values)
        """

        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]

        return {key: value for key, value in zip(keys, values)}

    def visit_Tuple(self, node):
        """Tuple visitor
        AST abstract grammar:
            Tuple(expr* elts, expr_context ctx)
        """

        return tuple(self.visit(element) for element in node.elts)

    def visit_List(self, node):
        """List visitor
        AST abstract grammar:
            List(expr* elts, expr_context ctx)
        """

        return [self.visit(element) for element in node.elts]

    def visit_keyword(self, node):
        """Keyword visitor
        AST abstract grammar:
            keyword = (identifier? arg, expr value)
        """

        return node.arg, self.visit(node.value)

    def visit_NameConstant(self, node):
        return tvm.runtime.convert(node.value)

    def visit_Constant(self, node):
        return tvm.runtime.convert(node.value)

    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            self.report_error("The data type should be one of (int, float)")
        return tvm.tir.const(node.n, dtype)

    def visit_Str(self, node):
        return node.s


def from_source(src, func_lineno=0):
    """Parse the src into TIR

    Parameters
    ----------
    src : str
        Pruned source of original script
    func_lineno : Optional[int]
        The line number of the first line of the script to be parsed
    Returns
    -------
    functions : PrimFunc or IRModule
        The PrimFunc or IRModule in IR.
    """

    root = ast.parse(src)
    parser = HybridParser(src, func_lineno)

    try:
        return parser.visit(root)
    except HybridParserError as e:
        raise e
    except TVMError as e:
        # TVM internal c++ error, we have to process the error message and inject line info
        inject_e = str(e).split("\n")
        msg = inject_e[-1].split(":", maxsplit=1)[1].strip()
        inject_e = inject_e[:-1]
        inject_e.extend(
            parser.wrap_line_col(msg, parser.current_lineno, parser.current_col_offset).split("\n")
        )
        inject_e[-1] = "TVM" + inject_e[-1][6:]
        raise TVMError("\n".join(inject_e))
    except Exception as e:
        inject_e = parser.wrap_line_col(str(e), parser.current_lineno, parser.current_col_offset)
        raise HybridParserError(inject_e)


tvm._ffi._init_api("tvm.hybrid.parser")
