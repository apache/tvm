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
"""TVM Script Parser For TIR"""
# pylint: disable=invalid-name, missing-docstring, inconsistent-return-statements, no-else-return
# pylint: disable=unnecessary-comprehension, unused-argument
# pylint: disable=relative-beyond-top-level
import json
import operator
import inspect
from typed_ast import ast3 as ast

import tvm
from tvm import IRModule
from tvm._ffi.base import TVMError
from tvm.ir import GlobalVar
from tvm.tir import all as _all
from tvm.tir import expr as _expr

from . import context_maintainer, ty
from .meta_unparser import MetaUnparser
from .registry import Registry
from .intrin import Intrin
from .special_stmt import SpecialStmt
from .scope_handler import ScopeHandler, WithScopeHandler, ForScopeHandler
from . import _ffi_api


class CallArgumentReader(object):
    """A helper class which read required argument from passed arguments"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_pos_only_arg(self, pos, name):
        """Get corresponding position only function argument from argument list"""
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs:
            self.parser.report_error(self.func_name + " misses argument " + name)
        else:
            arg = self.kwargs[name]

        return arg

    def get_kwarg(self, pos, name, default):
        """Get corresponding keyword function argument from argument list
        If user doesn't provide the argument, set it to default value
        """
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name in self.kwargs:
            arg = self.kwargs[name]
        else:
            return default

        return arg

    def get_varargs(self, pos):
        """Get corresponding variable argument from argument list"""
        if len(self.args) >= pos and len(self.kwargs) == 0:
            return self.args[pos - 1 :]
        return []


class TVMScriptParserError(RuntimeError):
    """TVM script Parser Runtime Error"""


class TVMScriptParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to TIR
    Notes for extension:
    1. To support new types of AST nodes. Add a function visit_xxx().
    2. To support new functions
        We divide allowed function calls in TVM script into 3 categories,
        which is intrin, scope_handler and special_stmt.
        1) intrin functions ought to have return value.
        User can also register intrin category function into parser.
        2) scope_handler functions have no return value and accepts parser and AST node
        as its arguments, which is used in for scope and with scope.
        3) special_stmt functions have return value and accepts parser and AST node as its arguments
        When visiting Call node, we check special_stmt registry at first. If no registered function
        is found, we then check intrin.
        When visiting With node, we check with_scope registry.
        When visiting For node, we check for_scope registry.
    """

    _binop_maker = {
        ast.Add: tvm.tir.Add,
        ast.Sub: tvm.tir.Sub,
        ast.Mult: tvm.tir.Mul,
        ast.Div: tvm.tir.Div,
        ast.FloorDiv: tvm.tir.FloorDiv,
        ast.Mod: tvm.tir.FloorMod,
        ast.BitOr: operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
        ast.Gt: tvm.tir.GT,
        ast.GtE: tvm.tir.GE,
        ast.Lt: tvm.tir.LT,
        ast.LtE: tvm.tir.LE,
        ast.Eq: tvm.tir.EQ,
        ast.NotEq: tvm.tir.NE,
        ast.And: tvm.tir.And,
        ast.Or: tvm.tir.Or,
    }

    _unaryop_maker = {ast.USub: operator.neg, ast.Invert: operator.invert, ast.Not: tvm.tir.Not}

    def __init__(self, src, base_lienno):
        self.context = None

        self.src = src.split("\n")
        self.base_lineno = base_lienno
        self.current_lineno = 0
        self.current_col_offset = 0
        self.meta = None

        self.functions = {}

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.context = context_maintainer.ContextMaintainer(self)  # scope emitter

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
        raise TVMScriptParserError(self.wrap_line_col(message, lineno, col_offset))

    def parse_body(self):
        body = []
        while len(self.context.node_stack[-1]) > 0:
            res = self.visit(self.context.node_stack[-1].pop())
            if res is not None:
                body.append(res)
        return tvm.tir.SeqStmt(body) if len(body) > 1 else body[0]

    def parse_arg_list(self, func, node_call):
        assert isinstance(node_call, ast.Call)
        # collect arguments
        args = [self.visit(arg) for arg in node_call.args]
        kw_args = [self.visit(keyword) for keyword in node_call.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
        # get the name and parameter list of func
        if isinstance(func, (Intrin, ScopeHandler, SpecialStmt)):
            func_name, param_list = func.signature()
        else:
            print(func)
            raise Exception("Internal Error")
        # check arguments and parameter list and get a list of arguments
        reader = CallArgumentReader(func_name, args, kw_args, self)
        pos_only, kwargs, varargs = param_list
        internal_args = list()
        for i, arg_name in enumerate(pos_only):
            internal_args.append(reader.get_pos_only_arg(i + 1, arg_name))
        for i, arg_info in enumerate(kwargs):
            arg_name, default = arg_info
            internal_args.append(reader.get_kwarg(i + 1 + len(pos_only), arg_name, default=default))
        if varargs is not None:
            internal_args.extend(reader.get_varargs(len(pos_only) + len(kwargs) + 1))
        return internal_args

    def parse_type(self, type_node):
        """ Parse type """
        if type_node is None:
            self.report_error("missing type annotation")
        res_type = self.visit(type_node)
        return tvm.ir.TupleType([]) if res_type is None else res_type.evaluate()

    def generic_visit(self, node):
        """Override method in ast.NodeVisitor.
        To directly filter out invalidate type of stmt.
        """

        self.report_error(type(node).__name__ + " AST node is not supported now")

    def visit_Module(self, node):
        """Module visitor
        AST abstract grammar:
            Module(stmt* body, type_ignore* type_ignore)
        By now we support two format of TVM script shown below.

        Example
        -------
        1. Generate a PrimFunc (If the code is printed, then it may also contain metadata)
        .. code-block:: python

            import tvm

            @tvm.script
            def A(...):
                ...

            # returns a PrimFunc
            func = A

        2. Generate an IRModule
        .. code-block:: python

            import tvm

            @tvm.script
            class MyMod():
               def A(...):
                  ...

               def B(...):
                   ...

                __tvm_meta__ = ...

            # returns an IRModule
            mod = MyMod()
        """

        if len(node.body) == 1 and isinstance(node.body[0], (ast.ClassDef, ast.FunctionDef)):
            # class or single function
            return self.visit(node.body[0])
        elif len(node.body) == 2:
            if isinstance(node.body[0], ast.Assign):
                node.body[0], node.body[1] = node.body[1], node.body[0]
            if isinstance(node.body[0], ast.FunctionDef) and TVMScriptParser.is_meta(node.body[1]):
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
            elif TVMScriptParser.is_meta(body_element) and not count:
                count = True
                self.init_meta(MetaUnparser().visit(body_element.value))
            else:
                self.report_error("invalid class member")

        # parse member functions
        for body_element in node.body:
            if isinstance(body_element, ast.FunctionDef):
                self.visit(body_element)

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
        self.context.new_scope(nodes=node.body)

        # add parameters of function
        for arg in node.args.args:
            arg_var = tvm.te.var(arg.arg, self.parse_type(arg.annotation))
            self.context.update_symbol(arg.arg, arg_var)
            self.context.func_params.append(arg_var)

        # fetch the body and return a tir.PrimFunc
        func = tvm.tir.PrimFunc(
            self.context.func_params,
            self.parse_body(),
            ret_type=self.parse_type(node.returns),
            buffer_map=self.context.func_buffer_map,
            attrs=tvm.ir.make_node("DictAttrs", **self.context.func_dict_attr),
        )
        self.functions[GlobalVar(node.name)] = func

        self.context.pop_scope()
        return func

    def visit_Assign(self, node):
        """Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)

        By now 3 patterns of Assign is supported:
            1. special stmts with return value
                1.1 Buffer = tir.match_buffer()/tir.buffer_decl()
                1.2 Var = tir.var()
                1.3 Var = tir.env_thread()
            2. (BufferStore) Buffer[PrimExpr, PrimExpr, ..., PrimExpr] = PrimExpr
            3. (Store)       Var[PrimExpr] = PrimExpr
            4. with scope handlers with concise scoping and var def
                4.1 var = tir.allocate()
        """

        if not len(node.targets) == 1:
            self.report_error("Only one-valued assignment is supported now")

        if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            # Pattern 1 & Pattern 4
            func = self.visit(node.value.func)
            arg_list = self.parse_arg_list(func, node.value)
            if isinstance(func, WithScopeHandler):
                if not func.concise_scope or not func.def_symbol:
                    self.report_error(
                        "with scope handler " + func.signature()[0] + " is not suitable here"
                    )
                # Pattern 4
                func.enter_scope(node, self.context)
                arg_list = self.parse_arg_list(func, node.value)
                func.body = self.parse_body()
                return func.exit_scope(node, self.context, arg_list)
            elif isinstance(func, SpecialStmt):
                # Pattern 1
                func.handle(node, self.context, arg_list)
            else:
                self.report_error("Unsupported Assign stmt")
        elif isinstance(node.targets[0], ast.Subscript):
            # Pattern 2 & Pattern 3
            symbol, indexes = self.visit(node.targets[0])
            rhs = self.visit(node.value)
            if isinstance(symbol, tvm.tir.Buffer):
                # Pattern 2
                return tvm.tir.BufferStore(symbol, tvm.runtime.convert(rhs), indexes)
            else:
                if len(indexes) != 1:
                    self.report_error("Invalid Store stmt")
                # Pattern 3
                return tvm.tir.Store(
                    symbol, tvm.runtime.convert(rhs), indexes[0], tvm.runtime.convert(True)
                )
        else:
            self.report_error("Unsupported Assign stmt")

    def visit_AnnAssign(self, node):
        """AnnAssign visitor
        AST abstract grammar:
            AnnAssign(expr target, expr annotation, expr? value, int simple)

        Pattern corresponds to concise mode of with tir.let()
        """

        if isinstance(node.target, ast.Name):
            value = self.visit(node.value)
            var = tvm.te.var(node.target.id, self.parse_type(node.annotation))
            self.context.update_symbol(var.name, var)
            body = self.parse_body()
            self.context.remove_symbol(var.name)
            return tvm.tir.LetStmt(var, value, body)
        else:
            self.report_error("Unsupported AnnAssign stmt")

    def visit_Assert(self, node):
        """Assert visitor
        AST abstract grammar:
            Assert(expr test, expr? msg)

        Pattern corresponds to concise mode of with tir.Assert()
        """

        condition = self.visit(node.test)
        if node.msg is None:
            self.report_error("Message of AssertStmt can't be None")
        message = self.visit(node.msg)
        body = self.parse_body()
        return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), body)

    def visit_For(self, node):
        """For visitor
        AST abstract grammar:
            For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        By now 1 pattern of For is supported:
            1. for scope handler
                for name in tir.serial()/tir.parallel()/tir.vectorized()/tir.unroll()
        """

        if not isinstance(node.iter, ast.Call):
            self.report_error("The loop iter should be a Call")
        func = self.visit(node.iter.func)
        if not isinstance(func, ForScopeHandler):
            self.report_error("Only for scope handlers can be used in for stmt")
        # prepare for new for scope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + node.iter.lineno - 1,
            node.iter.col_offset,
        )
        self.context.new_scope(nodes=node.body)
        # for scope handler process the scope
        func.enter_scope(node, self.context)
        func.body = self.parse_body()
        arg_list = self.parse_arg_list(func, node.iter)
        res = func.exit_scope(node, self.context, arg_list)
        # exit the scope
        self.context.pop_scope()
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def visit_With(self, node):
        """With visitor
        AST abstract grammar:
            With(withitem* items, stmt* body, string? type_comment)
            withitem = (expr context_expr, expr? optional_vars)
        By now 2 patterns of With is supported:
            1. with scope handler with symbol def
                with tir.allocate() as targets:
            2. with scope handler without symbol def
                with tir.let()/tir.Assert()/tir.attr()//tir.realize()
        """

        if not len(node.items) == 1:
            self.report_error("Only one with element is supported now")
        if not isinstance(node.items[0].context_expr, ast.Call):
            self.report_error("The context expression of with should be a Call")

        func_call = node.items[0].context_expr
        func_node = func_call.func
        func = self.visit(func_node)

        if not isinstance(func, WithScopeHandler):
            self.report_error("Function not allowed in with scope")
        # prepare for new block scope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + func_call.lineno - 1,
            func_call.col_offset,
        )
        self.context.new_scope(nodes=node.body)
        # with scope handler process the scope
        func.enter_scope(node, self.context)
        func.body = self.parse_body()
        arg_list = self.parse_arg_list(func, func_call)
        res = func.exit_scope(node, self.context, arg_list)
        # exit the scope
        self.context.pop_scope()
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def visit_If(self, node):
        """If visitor
        AST abstract grammar:
            If(expr test, stmt* body, stmt* orelse)
        """

        condition = self.visit(node.test)
        # then body
        self.context.new_scope(nodes=node.body)
        then_body = self.parse_body()
        self.context.pop_scope()

        # else body
        if len(node.orelse) > 0:
            self.context.new_scope(nodes=node.orelse)
            else_body = self.parse_body()
            self.context.pop_scope()
        else:
            else_body = None

        return tvm.tir.IfThenElse(condition, then_body, else_body)

    def visit_Call(self, node):
        """Call visitor
        AST abstract grammar:
            Call(expr func, expr* args, keyword* keywords)
            keyword = (identifier? arg, expr value)

        By now 3 patterns of Call is allowed
            1. Intrin representing PrimExpr/IterVar
                1.1 tir.int/uint/float8/16/32/64/floormod/floordiv/load/cast/ramp/broadcast/max
                1.2 tir.range/reduce_axis/scan_axis/opaque_axis
            2. tir.Op(dtype, ...)
            3. other callable functions
        """

        func = self.visit(node.func)
        if isinstance(func, Intrin) and not func.stmt:
            # pattern 1
            arg_list = self.parse_arg_list(func, node)
            return func.handle(arg_list)
        else:
            args = [self.visit(arg) for arg in node.args]
            kw_args = [self.visit(keyword) for keyword in node.keywords]
            kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}
            if isinstance(func, tvm.tir.op.Op):
                # pattern 2
                return tvm.tir.Call(kw_args["dtype"], func, args)
            elif callable(func):
                # pattern 3
                return func(*args, **kw_args)

        self.report_error("Unsupported function call")

    def visit_Expr(self, node):
        """Expr visitor
        AST abstract grammar:
            Expr(expr value)

        Now only 3 types of Expr stmt is allowed:
            1. Intrin representing Stmt without body
                tir.store()/tir.evaluate()
            2. with scope handlers with concise scoping without var def
                tir.attr()/tir.assert()/tir.allocate()/tir.realize()
            3. special stmt without var def
                tir.func_attr()
        """

        if not isinstance(node.value, ast.Call):
            self.report_error("Unsupported Expr stmt")

        func = self.visit(node.value.func)
        arg_list = self.parse_arg_list(func, node.value)

        if isinstance(func, Intrin) and func.stmt:
            # pattern 1
            return func.handle(arg_list)
        elif isinstance(func, WithScopeHandler) and func.concise_scope and not func.def_symbol:
            # pattern 2
            func.enter_scope(node, self.context)
            func.body = self.parse_body()
            return func.exit_scope(node, self.context, arg_list)
        elif isinstance(func, SpecialStmt) and not func.def_symbol:
            # pattern 3
            func.handle(node, self.context, arg_list)
            return

        self.report_error("Invalid Expr stmt")

    def visit_BinOp(self, node):
        """BinOp visitor
        AST abstract grammar:
            BinOp(expr left, operator op, expr right)
        """

        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not isinstance(node.op, tuple(TVMScriptParser._binop_maker.keys())):
            self.report_error("BinOp " + str(type(node.op)) + " is not supported now")
        return TVMScriptParser._binop_maker[type(node.op)](lhs, rhs)

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
            res.append(TVMScriptParser._binop_maker[type(node.ops[i])](lhs, rhs))
        return _all(*res)

    def visit_BoolOp(self, node):
        """BoolOp visitor
        AST abstract grammar:
            BoolOp(boolop op, expr* values)
        """

        values = [self.visit(value) for value in node.values]
        return TVMScriptParser._binop_maker[type(node.op)](*values)

    def visit_UnaryOp(self, node):
        """UnaryOp visitor
        AST abstract grammar:
            UnaryOp(unaryop op, expr operand)
        """

        operand = self.visit(node.operand)
        if not isinstance(node.op, tuple(TVMScriptParser._unaryop_maker.keys())):
            self.report_error("UnaryOp " + str(type(node.op)) + " is not supported now")
        return TVMScriptParser._unaryop_maker[type(node.op)](operand)

    def visit_Subscript(self, node):
        """Subscript visitor
        AST abstract grammar:
            Subscript(expr value, slice slice, expr_context ctx)
            slice = Slice(expr? lower, expr? upper, expr? step)
                    | ExtSlice(slice* dims)
                    | Index(expr value)
        By now 2 patterns of Subscript are supported:
            1. Buffer[index, index, ...], Buffer element access(BufferLoad & BufferStore)
               Var[index] Buffer element access()
            2. meta[type_key][index], Meta info access
        """

        symbol = self.visit(node.value)
        if symbol is None:
            self.report_error(node.value.id + " is not defined")
        if isinstance(symbol, (tvm.tir.expr.Var, tvm.tir.Buffer)):
            if isinstance(node.slice, ast.Index):
                # BufferLoad & BufferStore, Buffer/Var[index, index, ...]
                indexes = self.visit(node.slice.value)
                indexes = list(indexes) if isinstance(indexes, tuple) else [indexes]
                if isinstance(node.ctx, ast.Load):
                    if isinstance(symbol, tvm.tir.expr.Var):
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
        else:
            res = symbol[self.visit(slice)]
            if res is None:
                self.report_error("Only buffer variable and meta can be subscriptable")
            return res

    def visit_Attribute(self, node):
        """Attribute visitor
        AST abstract grammar:
            Attribute(expr value, identifier attr, expr_context ctx)
        """

        if isinstance(node.value, ast.Name):
            if node.value.id == "tir":
                func_name = "tir." + node.attr
                res = Registry.lookup(func_name)
                if res is not None:
                    return res
                try:
                    return tvm.ir.op.Op.get(func_name)
                except AttributeError:
                    self.report_error("Unregistered function tir." + node.attr)
            elif node.value.id == "ty":
                if not hasattr(ty, node.attr):
                    self.report_error("invalid type annotation ty." + node.attr)
                return getattr(ty, node.attr)

        symbol = self.visit(node.value)
        if symbol is None:
            self.report_error("Unsupported Attribute expression")
        if not hasattr(symbol, node.attr):
            self.report_error("Type " + type(symbol) + " has not attr " + node.attr)
        res = getattr(symbol, node.attr)
        return res

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

    def visit_Name(self, node):
        """Name visitor
        AST abstract grammar:
            Name(identifier id, expr_context ctx)
        """

        name = node.id
        if name == "meta":
            return self.meta
        symbol = Registry.lookup(name)
        if symbol is not None:
            return symbol
        symbol = self.context.lookup_symbol(name)
        if symbol is not None:
            return symbol
        self.report_error("Unknown identifier %s" % name)

    # note that after Python3.8, ast.NameConstant, ast.Num, ast.Str are no longer used
    def visit_Constant(self, node):
        return node.value

    def visit_NameConstant(self, node):
        return node.value

    def visit_Num(self, node):
        return node.n

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
    parser = TVMScriptParser(src, func_lineno)

    try:
        return parser.visit(root)
    except TVMScriptParserError as e:
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
        raise TVMError("\n".join(inject_e)) from e
    except Exception as e:
        inject_e = parser.wrap_line_col(str(e), parser.current_lineno, parser.current_col_offset)
        raise TVMScriptParserError(inject_e) from e


def _parse(script_in):
    """Helper function to parse TVM script into TIR"""
    return from_source(inspect.getsource(script_in), inspect.getsourcelines(script_in)[1])


def create_module(functions=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    functions: Optional[dict].
        Map of GlobalVar or str to PrimFunc

    Returns
    -------
    mod : IRModule
        An IRModule containing the passed definitions
    """

    return IRModule(functions=functions)


def asscript(input_ir, show_meta=False):
    """Transform a PrimFunc or IRModule to python syntax script

    Parameters
    ----------
    input_ir : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be dumped

    show_meta : bool
        Whether show meta

    Returns
    -------
    script : str
        The Python script
    """

    return _ffi_api.AsTVMScript(input_ir, show_meta)


def tir(script_in):
    """Decorate a python function or class as tvm script.

    The tvm function or parsing support parsing to the internal TIR.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """

    if inspect.isfunction(script_in):
        result = _parse(script_in)
    elif inspect.isclass(script_in):
        result = TVMScriptClass(script_in)
    else:
        raise TypeError("Only function and class are supported")
    result.__name__ = script_in.__name__
    result.__qualname__ = script_in.__qualname__
    return result


def module(script_in):
    """Decorate a python function or class as tvm script.

    Alias for tvm.script.tir for now.

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """
    return tir(script_in)


class TVMScriptClass:
    """Helper class for decorating a class"""

    def __init__(self, script_in):
        self.script = script_in

    def __call__(self, *args, **kwargs):
        # call the parser to transform tvm script into TIR
        return _parse(self.script)
