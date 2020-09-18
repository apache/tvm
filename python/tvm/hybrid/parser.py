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
import operator
from typed_ast import ast3 as ast

import tvm._ffi
from tvm import tir
from tvm._ffi.base import TVMError
from tvm.ir import GlobalVar
from tvm.tir import all as _all
from tvm.tir import expr as _expr

from . import scope_emitter, special_stmt, scope_handler, intrin, ty
from .meta_unparser import MetaUnparser
from .registry import Registry
from . import _ffi_api


class HybridParserError(RuntimeError):
    """Hybrid Parser Runtime Error"""


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to TIR
    Notes for extension:
    1. To support new types of AST nodes. Add a function visit_xxx().
    2. To support new functions
        We divide allowed function calls in hybrid script into 3 categories,
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
        self.var_env_dict = None

        self.src = src.split("\n")
        self.base_lineno = base_lienno
        self.current_lineno = 0
        self.current_col_offset = 0
        self.meta = None

        self.functions = {}
        self.target = None

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.params = []  # parameter list
        self.buffer_map = {}  # buffer map
        self.dict_attr = {}  # dict attr
        self.scope_emitter = scope_emitter.ScopeEmitter(self)  # scope emitter
        self.var_env_dict = {}  # map from var to thread env name

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

    def get_body(self):
        body = []
        while len(self.scope_emitter.node_stack[-1]) > 0:
            res = self.visit(self.scope_emitter.node_stack[-1].pop())
            if res is not None:
                body.append(res)
        return tvm.tir.SeqStmt(body) if len(body) > 1 else body[0]

    def get_type(self, type_node):
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
            arg_var = tvm.te.var(arg.arg, self.get_type(arg.annotation))
            self.scope_emitter.update_symbol(arg.arg, arg_var)
            self.params.append(arg_var)

        # visit the body of function
        self.scope_emitter.node_stack[-1].extend(reversed(node.body))

        # fetch the body and return a tir.PrimFunc
        func = tvm.tir.PrimFunc(
            self.params,
            self.get_body(),
            ret_type=self.get_type(node.returns),
            buffer_map=self.buffer_map,
            attrs=tvm.ir.make_node("DictAttrs", **self.dict_attr),
        )
        self.functions[GlobalVar(node.name)] = func
        return func

    def visit_Assign(self, node):
        """Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)
        By now only 3 types of Assign is supported:
            1. special stmts with return value
                1.1 Buffer = tir.buffer_bind()/tir.buffer_decl()
                1.2 Var = tir.var()
                1.3 Var = tir.env_thread()
            2. (BufferStore) Buffer[PrimExpr, PrimExpr, ..., PrimExpr] = PrimExpr
            3. (Store)       Var[PrimExpr] = PrimExpr
            4. with scope handlers with concise scoping and var def
                4.1 var = tir.alloc_with_scope()
        """

        if not len(node.targets) == 1:
            self.report_error("Only one-valued assignment is supported now")
        target = node.targets[0]

        if isinstance(target, ast.Name):
            # scenario 1&4
            self.target = [target.id]
            if not isinstance(node.value, ast.Call):
                self.report_error("Unsupported assign stmt")
            func = self.visit(node.value.func)
            if Registry.is_with_scope(func):
                # scenario 4
                return self.visit(node.value)
            else:
                # scenario 1
                rhs = self.visit(node.value)
                self.scope_emitter.update_symbol(target.id, rhs)
        elif isinstance(target, ast.Subscript):
            # scenario 2&3
            symbol, indexes = self.visit(target)
            rhs = self.visit(node.value)
            if isinstance(symbol, tvm.tir.Buffer):
                # BufferStore
                return tvm.tir.BufferStore(symbol, tvm.runtime.convert(rhs), indexes)
            else:
                if len(indexes) != 1:
                    self.report_error("Invalid Store stmt")
                # Store
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
            var = tvm.te.var(node.target.id, self.get_type(node.annotation))
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
            1. for name in tir.serial/parallel/vectorized/unroll(begin, end)
        """

        # check node.iter, which is a Call
        if not isinstance(node.iter, ast.Call):
            self.report_error("The loop iter should be a Call")
        func = self.visit(node.iter.func)
        if not Registry.is_for_scope(func):
            self.report_error("Function not allowed in for scope")
        # collect arguments
        args = [self.visit(arg) for arg in node.iter.args]
        kw_args = [self.visit(keyword) for keyword in node.iter.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + node.iter.lineno - 1,
            node.iter.col_offset,
        )
        res = func(self, node, args, kw_args)
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def visit_With(self, node):
        """With visitor
        AST abstract grammar:
            With(withitem* items, stmt* body, string? type_comment)
            withitem = (expr context_expr, expr? optional_vars)
        By now 2 types of With is supported:
            1. with tir.allocate() as targets:
            2. with tir.let()/tir.Assert()/tir.attr()//tir.realize()
        """
        if not len(node.items) == 1:
            self.report_error("Only one with element is supported now")
        if not isinstance(node.items[0].context_expr, ast.Call):
            self.report_error("The context expression of with should be a Call")

        func_call = node.items[0].context_expr
        func_node = func_call.func
        func = self.visit(func_node)

        if not Registry.is_with_scope(func):
            self.report_error("Function not allowed in with scope")

        self.target = []
        if node.items[0].optional_vars is not None:
            # preprocess optional var names
            if isinstance(node.items[0].optional_vars, ast.Name):
                self.target = [node.items[0].optional_vars.id]
            elif isinstance(node.items[0].optional_vars, (ast.List, ast.Tuple)):
                for var in node.items[0].optional_vars.elts:
                    if not isinstance(var, ast.Name):
                        self.report_error("Invalid optional var definition")
                self.target = [var.id for var in node.items[0].optional_vars.elts]
            else:
                self.report_error("Invalid optional var definition")
        # parse other arguments
        args = [self.visit(arg) for arg in func_call.args]
        kw_args = [self.visit(keyword) for keyword in func_call.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno, self.current_col_offset = (
            self.base_lineno + func_call.lineno - 1,
            func_call.col_offset,
        )
        res = func(self, node, args, kw_args)
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
        All the functions used outside With and For are registered in special_stmt or intrin
        """

        func = self.visit(node.func)
        # collect arguments
        args = [self.visit(arg) for arg in node.args]
        kw_args = [self.visit(keyword) for keyword in node.keywords]
        kw_args = {kw_arg[0]: kw_arg[1] for kw_arg in kw_args}

        if callable(func):
            if Registry.is_registered(func):
                return func(self, node, args, kw_args)
            else:
                return func(*args, **kw_args)
        elif isinstance(func, tvm.tir.op.Op):
            return tvm.tir.Call(kw_args["dtype"], func, args)

        self.report_error("Unsupported function call")

    def visit_Expr(self, node):
        """Expr visitor
        AST abstract grammar:
            Expr(expr value)
        Now only 3 types of `Expr` stmt is allowed:
            1. reducer.step()/tir.store()
            2. tir.attr()/tir.assert()/tir.allocate()/tir.realize()
            3. tir.set_func_attr()
        """

        if not isinstance(node.value, ast.Call):
            self.report_error("Unsupported Expr stmt")
        res = self.visit(node.value)
        if res is None or isinstance(res, tvm.tir.Stmt):
            return res
        self.report_error("Invalid Expr stmt")

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

        symbol = self.visit(node.value)
        if symbol is None:
            self.report_error(node.value.id + " is not defined")
        if isinstance(symbol, (tvm.tir.expr.Var, tvm.tir.Buffer)):
            if isinstance(node.slice, ast.Index):
                # BufferLoad & BufferStore, Buffer/Var[index, index, ...]
                indexes = self.visit(node.slice.value)
                indexes = list(indexes) if isinstance(indexes, tuple) else [indexes]
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
                res = Registry.look_up_function(func_name)
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
        symbol = Registry.look_up_function(name)
        if symbol is not None:
            return symbol
        symbol = self.scope_emitter.lookup_symbol(name)
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


tvm._ffi._init_api("hybrid", __name__)
