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
"""TVM Script Parser For TIR

We use [synr](https://synr.readthedocs.io) to get an AST that is stable over
different python versions. Synr also provides an error handling context that we
use for error reporting.
"""
# pylint: disable=invalid-name, inconsistent-return-statements, no-else-return, broad-except
import types
import json
import operator
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from synr import ast, Transformer, to_ast

import tvm
from tvm import IRModule
from tvm._ffi.base import TVMError
from tvm.ir import GlobalVar
from tvm.ir.function import BaseFunc
from tvm.tir import buffer
from tvm.tir.function import PrimFunc
from . import _ffi_api
from . import tir

from .context_maintainer import ContextMaintainer
from .meta_unparser import MetaUnparser
from .registry import Registry
from .diagnostics import TVMDiagnosticCtx
from .utils import tvm_span_from_synr, synr_span_from_tvm, call_with_error_reporting

from .tir.intrin import Intrin
from .tir.node import Slice, BufferSlice
from .tir.scope_handler import ScopeHandler, WithScopeHandler, ForScopeHandler
from .tir.special_stmt import SpecialStmt
from .tir import ty


class CallArgumentReader(object):
    """Helper class to read required arguments from passed arguments.

    When parsing a function call, we need to match the arguments provided in
    the AST to the required arguments of the function. This class makes sure
    all the positional arguments are filled and also fill keyword arguments
    with thier default value if a different value was not provided.
    """

    def __init__(self, func_name, args, kwargs, parser, node):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser
        self.node = node

    def get_pos_only_arg(self, pos, name):
        """Get corresponding position only function argument from argument list"""
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs:
            # If no positional argument was found in the AST, we see if it was
            # defined by name instead.
            # TODO(tkonolige): this error message is not quite correct. The
            # number of required arguments is >= pos
            self.parser.report_error(
                f"{self.func_name} requires {pos} arguments, but only {len(self.args)} were given.",
                self.node.span,
            )
        else:
            arg = self.kwargs[name]

        return arg

    def get_kwarg(self, pos, name, default):
        """Get corresponding keyword function argument from argument list.

        If the user hasn't provided the argument, set it to the default value.
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


class TVMScriptParser(Transformer):
    """Synr AST visitor pass which finally lowers to TIR.

    Notes for Extension
    -------------------
    1. To support a new type of AST node, add a function transform_xxx().
    2. To support new functions, add the function to the appropriate registry:
        We divide allowed function calls in TVM script into 3 categories,
        intrin, scope_handler and special_stmt.
        1. intrin functions are low level functions like mod, load, and
           constants. They correspond to a tir `IRNode`. They must have a
           return value. The user can register intrin functions for the parser to
           use.
        2. scope_handler functions have no return value. They take two
           arguments: the parser and the AST node. scope_handler functions are
           used in with and for statements.
        3. special_stmt functions handle cases that do not have a corresponding
           tir `IRNode`. These functions take the parser and the AST node as
           arguments and may return a value.
        When visiting a Call node, we check the special_stmt registry first. If
        no registered function is found, we then check the intrin registry.
        When visiting With node, we check the with_scope registry.
        When visiting For node, we check the for_scope registry.
    """

    _binop_maker = {
        ast.BuiltinOp.Add: tvm.tir.Add,
        ast.BuiltinOp.Sub: tvm.tir.Sub,
        ast.BuiltinOp.Mul: tvm.tir.Mul,
        ast.BuiltinOp.Div: tvm.tir.Div,
        ast.BuiltinOp.FloorDiv: tvm.tir.FloorDiv,
        ast.BuiltinOp.Mod: tvm.tir.FloorMod,
        ast.BuiltinOp.BitOr: lambda lhs, rhs, span: operator.or_(lhs, rhs),
        ast.BuiltinOp.BitAnd: lambda lhs, rhs, span: operator.and_(lhs, rhs),
        ast.BuiltinOp.BitXor: lambda lhs, rhs, span: operator.xor(lhs, rhs),
        ast.BuiltinOp.GT: tvm.tir.GT,
        ast.BuiltinOp.GE: tvm.tir.GE,
        ast.BuiltinOp.LT: tvm.tir.LT,
        ast.BuiltinOp.LE: tvm.tir.LE,
        ast.BuiltinOp.Eq: tvm.tir.EQ,
        ast.BuiltinOp.NotEq: tvm.tir.NE,
        ast.BuiltinOp.And: tvm.tir.And,
        ast.BuiltinOp.Or: tvm.tir.Or,
    }

    _unaryop_maker = {
        ast.BuiltinOp.USub: lambda rhs, span: operator.neg(rhs),
        ast.BuiltinOp.Invert: lambda rhs, span: operator.invert(rhs),
        ast.BuiltinOp.Not: tvm.tir.Not,
    }

    # pylint gets confused here with synr.Transformer which doesn't have a
    # custom init, so just disable it
    def __init__(
        self, base_lineno, tir_namespace, closure_vars
    ):  # pylint: disable=super-init-not-called
        self.context = None

        self.base_lineno = base_lineno
        self.current_lineno = 0
        self.current_col_offset = 0
        self.tir_namespace = tir_namespace
        self.closure_vars = closure_vars
        self.meta = None
        self._inside_buffer_sugar = False

    def init_function_parsing_env(self):
        """Initialize function parsing environment"""
        self.context = ContextMaintainer(self.report_error, self.closure_vars)  # scope emitter

    def init_meta(self, meta_dict):
        if meta_dict is not None:
            self.meta = tvm.ir.load_json(json.dumps(meta_dict))

    def transform(self, node):
        """Generic transformation for visiting the AST. Dispatches to
        `transform_ClassName` for the appropriate ClassName."""
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset

        if hasattr(node, "lineno"):
            self.current_lineno = self.base_lineno + node.lineno - 1
        if hasattr(node, "col_offset"):
            self.current_col_offset = node.col_offset

        method = "transform_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        transform_res = visitor(node)

        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset

        return transform_res

    def match_tir_namespace(self, identifier: str) -> bool:
        """Check if the namespace is equal to tvm.script.tir"""
        return identifier in self.tir_namespace

    def report_error(self, message: str, span: Union[ast.Span, tvm.ir.Span]):
        """Report an error occuring at a location.

        This just dispatches to synr's DiagnosticContext.

        Parameters
        ----------
        message : str
            Error message
        span : Union[synr.ast.Span, tvm.ir.Span]
            Location of the error
        """
        if isinstance(span, tvm.ir.Span):
            span = synr_span_from_tvm(span)
        self.error(message, span)

    def parse_body(self, parent):
        """Parse remaining statements in this scope.

        Parameters
        ----------
        parent : synr.ast.Node
            Parent node of this scope. Errors will be reported here.
        """
        body = []
        spans = []
        stmt = parent
        while len(self.context.node_stack[-1]) > 0:
            stmt = self.context.node_stack[-1].pop()
            spans.append(stmt.span)
            res = self.transform(stmt)
            if res is not None:
                body.append(res)
        if len(body) == 0:
            self.report_error(
                "Expected another statement at the end of this block. Perhaps you "
                "used a concise statement and forgot to include a body afterwards.",
                stmt.span,
            )
        else:
            return (
                tvm.tir.SeqStmt(body, tvm_span_from_synr(ast.Span.union(spans)))
                if len(body) > 1
                else body[0]
            )

    def parse_arg_list(self, func, node_call):
        """Match the arguments of a function call in the AST to the required
        arguments of the function. This handles positional arguments,
        positional arguments specified by name, keyword arguments, and varargs.

        Parameters
        ----------
        func : Function
            The function that provides the signature

        node_call: Union[ast.Call, ast.TypeApply, ast.TypeCall]
            The AST call node that calls into the function.

        Returns
        -------
        arg_list : list
            The parsed positional argument.
        """
        assert isinstance(node_call, (ast.Call, ast.TypeApply, ast.TypeCall))
        # collect arguments
        args = [self.transform(arg) for arg in node_call.params]
        if isinstance(node_call, ast.TypeApply):
            kw_args = {}  # TypeApply (e.g. foo[bar]) doesn't have kwargs defined in synr
        else:
            kw_args = {
                self.transform(k): self.transform(v) for k, v in node_call.keyword_params.items()
            }
        # get the name and parameter list of func
        if isinstance(func, (Intrin, ScopeHandler, SpecialStmt)):
            func_name, param_list = func.signature()
        else:
            self.report_error(
                "Internal Error: function must be of type Intrin, ScopeHandler or SpecialStmt, "
                f"but it is {type(func).__name__}",
                node_call.span,
            )
        # check arguments and parameter list and get a list of arguments
        reader = CallArgumentReader(func_name, args, kw_args, self, node_call)
        pos_only, kwargs, varargs = param_list
        internal_args = list()

        for i, arg_name in enumerate(pos_only):
            internal_args.append(reader.get_pos_only_arg(i + 1, arg_name))
        for i, arg_info in enumerate(kwargs):
            arg_name, default = arg_info
            internal_args.append(reader.get_kwarg(i + 1 + len(pos_only), arg_name, default=default))
        if varargs is not None:
            internal_args.extend(reader.get_varargs(len(pos_only) + len(kwargs) + 1))
        elif len(args) + len(kw_args) > len(pos_only) + len(kwargs):
            self.report_error(
                "Arguments mismatched. "
                + f"Expected {len(pos_only) + len(kwargs)} args but got "
                + f"{len(args) + len(kw_args)}",
                node_call.span,
            )
        return internal_args

    def parse_type(self, type_node, parent):
        """Parse a type annotation.

        We require the parent object to the type so that we have a place to
        report the error message if the type does not exist.
        """
        if type_node is None:
            self.report_error("A type annotation is required", parent.span)
        res_type = self.transform(type_node)
        return tvm.ir.TupleType([]) if res_type is None else res_type.evaluate()

    def generic_visit(self, node):
        """Fallback visitor if node type is not handled. Reports an error."""

        self.report_error(type(node).__name__ + " AST node is not supported", node.span)

    def transform_Module(self, node):
        """Module visitor

        Right now, we only support two formats for TVM Script.

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

            @tvm.script.ir_module
            class MyMod():
                @T.prim_func
                def A(...):
                    ...
                @T.prim_func
                def B(...):
                    ...

                __tvm_meta__ = ...

            # returns an IRModule
            mod = MyMod
        """
        if len(node.funcs) == 1:
            return self.transform(next(iter(node.funcs.values())))
        elif len(node.funcs) == 0:
            self.report_error(
                "You must supply at least one class or function definition", node.span
            )
        else:
            self.report_error(
                "Only one-function, one-class or function-with-meta source code is allowed",
                ast.Span.union([x.span for x in list(node.funcs.values())[1:]]),
            )

    def transform_Class(self, node):
        """Class definition visitor.

        A class can have multiple function definitions and a single
        :code:`__tvm_meta__` statement. Each class corresponds to a single
        :code:`IRModule`.

        Example
        -------
        .. code-block:: python

            @tvm.script.ir_module
            class MyClass:
                __tvm_meta__ = {}
                def A():
                    T.evaluate(0)
        """
        if len(node.assignments) == 1:
            if not (
                len(node.assignments[0].lhs) == 1
                and isinstance(node.assignments[0].lhs[0], ast.Var)
                and node.assignments[0].lhs[0].id.name == "__tvm_meta__"
            ):
                self.report_error(
                    "The only top level assignments allowed are `__tvm_meta__ = ...`",
                    node.assignments[0].span,
                )
            self.init_meta(
                MetaUnparser().do_transform(node.assignments[0].rhs, self._diagnostic_context)
            )
        elif len(node.assignments) > 1:
            self.report_error(
                "Only a single top level `__tvm_meta__` is allowed",
                ast.Span.union([x.span for x in node.assignments[1:]]),
            )

        return IRModule(
            {GlobalVar(name): self.transform(func) for name, func in node.funcs.items()}
        )

    def transform_Function(self, node):
        """Function definition visitor.

        Each function definition is translated to a single :code:`PrimFunc`.

        There are a couple restrictions on TVM Script functions:
        1. Function arguments must have their types specified.
        2. The body of the function can contain :code:`func_attr` to specify
           attributes of the function (like it's name).
        3. The body of the function can also contain multiple :code:`buffer_bind`s,
           which give shape and dtype information to arguments.
        4. Return statements are implicit.

        Example
        -------
        .. code-block:: python

            @T.prim_func
            def my_function(x: T.handle):  # 1. Argument types
                T.func_attr({"global_symbol": "mmult"})  # 2. Function attributes
                X_1 = tir.buffer_bind(x, [1024, 1024])  # 3. Buffer binding
                T.evaluate(0)  # 4. This function returns 0
        """

        def check_as_torch_decorator(decorator: Union[ast.Call, ast.Var]):
            if isinstance(decorator, ast.Call):
                if len(decorator.params) != 1:
                    return False
                func_name = decorator.func_name
            else:
                func_name = decorator
            if isinstance(func_name, ast.Var):
                return func_name.id.name == "as_torch"

        def check_decorator(decorators: List[ast.Expr]) -> bool:
            """Check the decorator is `T.prim_func"""
            if len(decorators) > 2 or len(decorators) == 0:
                return False
            if len(decorators) == 2 and not check_as_torch_decorator(decorators[0]):
                return False
            d: ast.Expr = decorators[-1]
            return (
                isinstance(d, ast.Attr)
                and isinstance(d.object, ast.Var)
                and self.match_tir_namespace(d.object.id.name)
                and d.field.name == "prim_func"
            )

        self.init_function_parsing_env()
        self.context.enter_scope(nodes=node.body.stmts)

        # add parameters of function
        for arg in node.params:
            # Note that this case is for T.match_buffer syntax sugar
            if isinstance(arg.ty, (ast.TypeCall, ast.TypeApply)) and isinstance(
                self.transform(arg.ty.func_name), ty.GenericBufferType
            ):
                result = self.handle_match_buffer_type(arg.ty, arg.name)
                if not isinstance(result, buffer.Buffer):
                    self.report_error(
                        "The result type of evaluating TypeCall and TypeApply stmt"
                        f" is wrong: {type(result)}. It should be a Buffer",
                        node.span,
                    )
                arg_name_with_handle = arg.name + "_handle"
                arg_var = tvm.te.var(arg_name_with_handle, tvm.ir.PrimType("handle"))
                self.context.func_buffer_map[arg_var] = result
                self.context.update_symbol(arg.name, result, node)
            else:
                arg_var = tvm.te.var(arg.name, self.parse_type(arg.ty, arg))
                self.context.update_symbol(arg.name, arg_var, node)
            self.context.func_params.append(arg_var)

        if not check_decorator(node.decorators):
            self.report_error(
                "All functions should be decorated by `T.prim_func`",
                node.span,
            )

        # fetch the body of root block
        body = self.parse_body(node.body)

        # return a tir.PrimFunc
        dict_attr = self.context.func_dict_attr
        ret_type = self.parse_type(node.ret_type, node) if node.ret_type is not None else None
        func = tvm.tir.PrimFunc(
            self.context.func_params,
            body,
            ret_type,
            buffer_map=self.context.func_buffer_map,
            preflattened_buffer_map=self.context.func_preflattened_buffer_map,
            attrs=tvm.ir.make_node("DictAttrs", **dict_attr) if dict_attr else None,
            span=tvm_span_from_synr(node.span),
        )

        # New Scope : Implicit root block
        # Each function contains an implicit root block in TensorIR,
        # so here we need a block scope for it.
        # If the PrimFunc is not a TensorIR func (e.g. TE scheduled func or low-level func),
        # the root block will not be added. The logic to add root block is in `_ffi_api.Complete`

        # Fix the PrimFunc
        # 1. generate root block if necessary
        # 2. generate surrounding loops for blocks if necessary

        func = call_with_error_reporting(
            self.report_error,
            node.span,
            _ffi_api.Complete,
            func,
            self.context.root_alloc_buffers,
        )

        self.context.exit_scope()
        return func

    def transform_Lambda(self, node):
        """Lambda visitor

        Return an array of input parameters and the transformed lambda body.
        """

        self.context.enter_scope(nodes=[node.body])

        # add parameters of the lambda
        arg_vars = []
        for arg in node.params:
            # Use "void" for dtype here. The actual type is not yet known and will be
            # determined later. Using void type will allow IRSubstitute to do the
            # replacement without flagging a type-mismatch error.
            arg_var = tvm.te.var(arg.name, dtype="")
            arg_vars.append(arg_var)
            self.context.update_symbol(arg.name, arg_var, node)

        # the body of a lambda must be an expr
        if not isinstance(node.body, ast.Expr):
            self.report_error("The body of a lambda must be an expression", node.span)

        # transform the body of the lambda
        body = self.transform(node.body)

        self.context.exit_scope()
        return arg_vars, body

    def transform_Assign(self, node):
        """Assign visitor
        AST abstract grammar:
            Assign(expr* targets, expr value, string? type_comment)

        By now 5 patterns of Assign is supported:
            1. special stmts with return value
                1.1 Buffer = T.match_buffer()/T.buffer_decl()
                1.2 Var = T.var()
                1.3 Var = T.env_thread()
            2. (BufferStore) Buffer[PrimExpr, PrimExpr, ..., PrimExpr] = PrimExpr
            3. (Store)       Var[PrimExpr] = PrimExpr
            4. with scope handlers with concise scoping and var def
                4.1 var = T.allocate()
            5. A call to a pure python function, consuming and producing TVMScript values.
               The outputs are inlined into the following body (no variable is created).
               x, y = f(...)
        """

        if isinstance(node.rhs, ast.Call):
            # Pattern 1 & Pattern 4
            if isinstance(node.rhs.func_name, ast.Op):
                func = None
            else:
                func = self.transform(node.rhs.func_name)

            if isinstance(func, WithScopeHandler):
                if not func.concise_scope or not func.def_symbol:
                    self.report_error(
                        "with scope handler " + func.signature()[0] + " is not suitable here",
                        node.rhs.span,
                    )
                # Pattern 4
                arg_list = self.parse_arg_list(func, node.rhs)
                func.enter_scope(node, self.context, arg_list, node.rhs.func_name.span)
                func.body = self.parse_body(node)
                return func.exit_scope(node, self.context, arg_list, node.rhs.func_name.span)
            elif isinstance(func, SpecialStmt):
                # Pattern 1
                arg_list = self.parse_arg_list(func, node.rhs)
                func.handle(node, self.context, arg_list, node.rhs.func_name.span)
                return self.parse_body(node)
            elif isinstance(func, types.FunctionType):
                # Pattern 5
                args = [self.transform(arg) for arg in node.rhs.params]
                try:
                    out = func(*args)
                except Exception as e:
                    self.report_error(
                        "Error occurred when invoking the function "
                        + func.__name__
                        + ": \n"
                        + str(e),
                        node.rhs.span,
                    )

                if len(node.lhs) == 1 and not isinstance(out, list):
                    out = [out]

                assert len(out) == len(node.lhs)

                for var, value in zip(node.lhs, out):
                    self.context.update_symbol(var.id.name, value, node)

                body = self.parse_body(node)

                for var, value in zip(node.lhs, out):
                    self.context.remove_symbol(var.id.name)

                return body

        if isinstance(node.rhs, (ast.Call, ast.Constant)):
            # Pattern 4 of let binding
            value = self.transform(node.rhs)
            if len(node.lhs) == 1 and not isinstance(node.lhs[0], ast.Var):
                # This is a little confusing because it only is true when
                # we have taken this branch. We might need to clarify what
                # exectly is allowed in Assignments in tvmscript.
                self.report_error(
                    "Left hand side of assignment must be an unqualified variable",
                    node.span,
                )
            ast_var = node.lhs[0]

            if node.ty is None and hasattr(value, "dtype"):
                var_ty = value.dtype
            else:
                var_ty = self.parse_type(node.ty, ast_var)

            var = tvm.te.var(
                ast_var.id.name,
                var_ty,
                span=tvm_span_from_synr(ast_var.span),
            )
            self.context.update_symbol(var.name, var, node)
            body = self.parse_body(node)
            self.context.remove_symbol(var.name)
            return tvm.tir.LetStmt(var, value, body, span=tvm_span_from_synr(node.span))

        self.report_error(
            """Assignments should be one of:
            1. A "special statement" with return value
                1.1 Buffer = T.match_buffer()/T.buffer_decl()
                1.2 Var = T.var()
                1.3 Var = T.env_thread()
            2. A store into a buffer: Buffer[PrimExpr, PrimExpr, ..., PrimExpr] = PrimExpr
            3. A store into a variable: Var[PrimExpr] = PrimExpr
            4. A with scope handler with concise scoping and var def
                4.1 var = T.allocate()
            5. The right-hand side being a call to a pure python function, consuming and
               producing TVMScript values.
               x, y = f(...)""",
            node.span,
        )

    def transform_SubscriptAssign(self, node):
        """Visitor for statements of the form :code:`x[1] = 2`."""
        symbol = self.transform(node.params[0])
        indexes = self.transform(node.params[1])
        rhs = self.transform(node.params[2])
        rhs_span = tvm_span_from_synr(node.params[2].span)
        if isinstance(symbol, tvm.tir.Buffer):
            if len(indexes) != len(symbol.shape):
                self.report_error(
                    f"Buffer {symbol.name} is {len(symbol.shape)}-dimensional, "
                    f"cannot be indexed by {len(indexes)}-dimensional indices.",
                    node.params[1].span,
                )

            def __convert_index(x):
                if isinstance(x, Slice):
                    return x.as_index_expr(self.report_error)
                return x

            # BufferStore
            indexes = [__convert_index(x) for x in indexes]
            return tvm.tir.BufferStore(
                symbol,
                tvm.runtime.convert(rhs, span=rhs_span),
                indexes,
                span=tvm_span_from_synr(node.span),
            )
        else:
            if symbol.dtype == "handle" and len(indexes) != 1:
                self.report_error(
                    "Handles only support one-dimensional indexing. Use `T.match_buffer` to "
                    "construct a multidimensional buffer from a handle.",
                    node.params[0].span,
                )
            if len(indexes) != 1:
                self.report_error(
                    f"Store is only allowed with one index, but {len(indexes)} were provided.",
                    node.params[1].span,
                )
            self.report_error(
                "Use of tir.Store has been deprecated in favor of tir.BufferStore.", node.span
            )

    def transform_AttrAssign(self, node):
        """Visitor for statements of the form :code:`x.y = 2`."""
        obj = self.transform(node.params[0])
        field = node.params[1]
        value = self.transform(node.params[2])

        if not hasattr(obj, field.name):
            self.error(f"Field {field.name} does not exist", field.span)

        var = getattr(obj, field.name)

        if not isinstance(var, tvm.tir.Var):
            self.error(
                f"Can only assign to tir.Var attributes, not {type(var).__name__}", node.span
            )

        body = self.parse_body(node)
        return tvm.tir.LetStmt(var, value, body, span=tvm_span_from_synr(node.span))

    def transform_Assert(self, node):
        """Assert visitor

        Pattern corresponds to concise mode of :code:`with T.Assert()`.
        """

        condition = self.transform(node.condition)
        if node.msg is None:
            self.report_error("Assert statements must have an error message.", node.span)
        message = self.transform(node.msg)
        body = self.parse_body(node)
        return tvm.tir.AssertStmt(
            condition, tvm.runtime.convert(message), body, span=tvm_span_from_synr(node.span)
        )

    def transform_For(self, node):
        """For visitor
        AST abstract grammar:
            For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        By now 1 pattern of For is supported:
            1. for scope handler
                for name in T.serial()/T.parallel()/T.vectorized()/T.unroll()/range()/
                            T.grid()/T.thread_binding()
        """

        if not isinstance(node.rhs, ast.Call):
            self.report_error("The loop iterator should be a function call.", node.rhs.span)
        func = self.transform(node.rhs.func_name)
        if not isinstance(func, ForScopeHandler):
            self.report_error(
                "Only For scope handlers can be used in a for statement.", node.rhs.func_name.span
            )
        # prepare for new for scope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno = node.span.start_line
        self.current_col_offset = node.span.start_column
        self.context.enter_scope(nodes=node.body.stmts)
        # for scope handler process the scope
        arg_list = [
            tvm.runtime.convert(arg, span=tvm_span_from_synr(node.rhs.span))
            for arg in self.parse_arg_list(func, node.rhs)
        ]
        func.enter_scope(node, self.context, arg_list, node.rhs.func_name.span)
        func.body = self.parse_body(node)
        res = func.exit_scope(node, self.context, arg_list, node.rhs.func_name.span)
        # exit the scope
        self.context.exit_scope()
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def transform_While(self, node):
        """While visitor
        AST abstract grammar:
            While(expr condition, stmt* body)
        """
        condition = self.transform(node.condition)
        # body
        self.context.enter_scope(nodes=node.body.stmts)
        body = self.parse_body(node)
        self.context.exit_scope()

        return tvm.tir.While(condition, body, span=tvm_span_from_synr(node.span))

    def transform_With(self, node):
        """With visitor
        AST abstract grammar:
            With(withitem* items, stmt* body, string? type_comment)
            withitem = (expr context_expr, expr? optional_vars)
        By now 2 patterns of With is supported:
            1. with scope handler with symbol def
                with T.allocate() as targets:
            2. with scope handler without symbol def
                with T.block(*axes)/T.let()/T.Assert()/T.attr()/T.realize()
        """

        if not isinstance(node.rhs, ast.Call):
            self.report_error(
                "The context expression of a `with` statement should be a function call.",
                node.rhs.span,
            )

        func = self.transform(node.rhs.func_name)

        if not isinstance(func, WithScopeHandler):
            self.report_error(
                f"Function {func} cannot be used in a `with` statement.", node.rhs.func_name.span
            )
        # prepare for new block scope
        old_lineno, old_col_offset = self.current_lineno, self.current_col_offset
        self.current_lineno = node.body.span.start_line
        self.current_col_offset = node.body.span.start_column
        self.context.enter_block_scope(nodes=node.body.stmts)
        # with scope handler process the scope
        arg_list = self.parse_arg_list(func, node.rhs)
        func.enter_scope(node, self.context, arg_list, node.rhs.func_name.span)
        func.body = self.parse_body(node)
        res = func.exit_scope(node, self.context, arg_list, node.rhs.func_name.span)
        # exit the scope
        self.context.exit_block_scope()
        self.current_lineno, self.current_col_offset = old_lineno, old_col_offset
        return res

    def transform_If(self, node):
        """If visitor
        AST abstract grammar:
            If(expr test, stmt* body, stmt* orelse)
        """

        condition = self.transform(node.condition)
        # then body
        self.context.enter_scope(nodes=node.true.stmts)
        then_body = self.parse_body(node)
        self.context.exit_scope()

        # else body
        if len(node.false.stmts) > 0:
            self.context.enter_scope(nodes=node.false.stmts)
            else_body = self.parse_body(node)
            self.context.exit_scope()
        else:
            else_body = None

        return tvm.tir.IfThenElse(
            condition, then_body, else_body, span=tvm_span_from_synr(node.span)
        )

    def transform_Call(self, node):
        """Call visitor

        3 different Call patterns are allowed:
            1. Intrin representing a PrimExpr/IterVar
                1.1 tir.int/uint/float8/16/32/64/floormod/floordiv/load/cast/ramp/broadcast/max
                1.2 tir.range/reduce_axis/scan_axis/opaque_axis
            2. tir.Op(dtype, ...)
            3. other callable functions
        """

        if isinstance(node.func_name, ast.Op):
            if node.func_name.name == ast.BuiltinOp.Subscript:
                return self.transform_Subscript(node)
            if node.func_name.name in self._binop_maker:
                lhs = self.transform(node.params[0])
                # There is no supertype for everything that can appear in
                # an expression, so we manually add what we might get here.
                if not isinstance(lhs, (tvm.tir.PrimExpr, BufferSlice)):
                    # We would really like to report a more specific
                    # error here, but this parser contains no distinction
                    # between parsing statements and parsing expressions. All
                    # rules just call `transform`.
                    self.report_error(
                        f"Left hand side of binary op must be a PrimExpr, "
                        "but it is a {type(lhs).__name__}",
                        node.params[0].span,
                    )
                rhs = self.transform(node.params[1])
                if not isinstance(rhs, (tvm.tir.PrimExpr, BufferSlice)):
                    self.report_error(
                        f"Right hand side of binary op must be a PrimExpr, "
                        "but it is a {type(rhs).__name__}",
                        node.params[1].span,
                    )
                return call_with_error_reporting(
                    self.report_error,
                    node.span,
                    lambda node, lhs, rhs, span: self._binop_maker[node.func_name.name](
                        lhs, rhs, span=span
                    ),
                    node,
                    lhs,
                    rhs,
                    tvm_span_from_synr(node.span),
                )
            if node.func_name.name in self._unaryop_maker:
                rhs = self.transform(node.params[0])
                return self._unaryop_maker[node.func_name.name](
                    rhs, span=tvm_span_from_synr(node.span)
                )
            self.report_error(f"Unsupported operator {node.func_name.name}.", node.func_name.span)
        else:
            func = self.transform(node.func_name)
            if isinstance(func, Intrin) and not func.stmt:
                # pattern 1
                arg_list = self.parse_arg_list(func, node)
                return call_with_error_reporting(
                    self.report_error,
                    node.func_name.span,
                    func.handle,
                    arg_list,
                    node.func_name.span,
                )
            else:
                args = [self.transform(arg) for arg in node.params]
                kw_args = {
                    self.transform(k): self.transform(v) for k, v in node.keyword_params.items()
                }
                if isinstance(func, tvm.tir.op.Op):
                    if not "dtype" in kw_args.keys():
                        self.report_error(f"{func} requires a dtype keyword argument.", node.span)
                    # pattern 2
                    return tvm.tir.Call(
                        kw_args["dtype"], func, args, span=tvm_span_from_synr(node.span)
                    )
                elif callable(func):
                    # pattern 3
                    return func(*args, **kw_args)
                else:
                    self.report_error(
                        f"Function is neither callable nor a tvm.tir.op.Op (it is a {type(func)}).",
                        node.func_name.span,
                    )

    def transform_UnassignedCall(self, node):
        """Visitor for statements that are function calls.

        This handles function calls that appear on thier own line like `tir.realize`.

        Examples
        --------
        .. code-block:: python

            @T.prim_func
            def f():
                A = T.buffer_decl([10, 10])
                T.realize(A[1:2, 1:2], "")  # This is an UnassignedCall
                A[1, 1] = 2  # This is also an UnassignedCall
        """
        # Only allowed builtin operator that can be a statement is x[1] = 3 i.e. subscript assign.
        if isinstance(node.call.func_name, ast.Op):
            if node.call.func_name.name == ast.BuiltinOp.SubscriptAssign:
                return self.transform_SubscriptAssign(node.call)

            if node.call.func_name.name == ast.BuiltinOp.AttrAssign:
                return self.transform_AttrAssign(node.call)

            self.report_error(
                "Binary and unary operators are not allowed as a statement", node.span
            )

        # handle a regular function call
        func = self.transform(node.call.func_name)
        arg_list = self.parse_arg_list(func, node.call)

        if isinstance(func, tir.scope_handler.AssertHandler):
            self.report_error(
                "A standalone `T.Assert` is not allowed. Use `assert condition, message` "
                "instead.",
                node.call.func_name.span,
            )

        if isinstance(func, Intrin):
            if func.stmt:
                return call_with_error_reporting(
                    self.report_error,
                    node.call.func_name.span,
                    func.handle,
                    arg_list,
                    node.call.func_name.span,
                )
            else:
                self.report_error(f"This intrinsic cannot be used as a statement.", node.call.span)
        elif isinstance(func, WithScopeHandler) and func.concise_scope and not func.def_symbol:
            func.enter_scope(node, self.context, arg_list, node.call.func_name.span)
            func.body = self.parse_body(node)
            return func.exit_scope(node, self.context, arg_list, node.call.func_name.span)
        elif isinstance(func, SpecialStmt) and not func.def_symbol:
            func.handle(node, self.context, arg_list, node.call.func_name.span)
            return

        self.report_error(
            "Unexpected statement. Expected an assert, an intrinsic, a with statement, or a "
            f"special statement, but got {type(func).__name__}.",
            node.call.func_name.span,
        )

    def transform_Slice(self, node):
        """Index slice visitor."""
        start = self.transform(node.start)
        end = self.transform(node.end)
        if not (
            isinstance(node.step, ast.Constant)
            and isinstance(node.step.value, int)
            and node.step.value > 0
        ):
            self.report_error(
                "Only positive integer step size is supported for slices.", node.step.span
            )
        return Slice(start, end, node.step.value, tvm_span_from_synr(node.span))

    def transform_Subscript(self, node):
        """Array access visitor.

        By now only 3 types of Subscript are supported:
            1. Buffer[index, index, ...], Buffer element access(BufferLoad & BufferStore)
               Var[index] Buffer element access()
            2. Buffer[start: stop, start: stop, ...], BufferRealize(realize(buffer[...]))
            3. Array[index], Buffer element access
        """

        symbol = self.transform(node.params[0])
        if symbol is None:
            self.report_error(
                f"Variable {node.params[0].id.name} is not defined.", node.params[0].span
            )

        indexes = [self.transform(x) for x in node.params[1].values]
        if isinstance(symbol, tvm.tir.expr.Var):
            if symbol.dtype == "handle":
                self.report_error(
                    "Cannot read directly from a handle, use `T.match_buffer` "
                    "to create a buffer to read from.",
                    node.params[0].span,
                )
            if len(indexes) > 1:
                self.report_error(
                    "Only a single index can be provided when indexing into a `var`.",
                    node.params[1].span,
                )
            index = indexes[0]
            if not isinstance(index, (tvm.tir.PrimExpr, int)):
                self.report_error(
                    "Var load index should be an int or PrimExpr, but it is a" + type(index),
                    node.span,
                )

            self.report_error(
                "Use of tir.Load has been deprecated in favor of tir.BufferLoad", node.span
            )
        elif isinstance(symbol, tvm.tir.Buffer):
            return BufferSlice(
                symbol, indexes, self.report_error, span=tvm_span_from_synr(node.span)
            )
        elif isinstance(symbol, tvm.container.Array):
            if len(indexes) > 1:
                self.report_error(
                    "Array access should be one-dimension access, but the indices are "
                    + str(indexes),
                    node.span,
                )
            index = indexes[0]
            if not isinstance(index, (int, tvm.tir.expr.IntImm)):
                self.report_error(
                    "Array access index expected int or IntImm, but got " + type(index),
                    node.span,
                )
            if int(index) >= len(symbol):
                self.report_error(
                    f"Array access out of bound, size: {len(symbol)}, got index {index}.",
                    node.span,
                )
            return symbol[int(index)]
        else:
            self.report_error(
                f"Cannot subscript from a {type(symbol).__name__}. Only variables and "
                "buffers are supported.",
                node.params[0].span,
            )

    def transform_Attr(self, node):
        """Visitor for field access of the form `x.y`.

        This visitor is used to lookup function and symbol names. We have two
        cases to handle here:
        1. If we have a statement of the form `tir.something`, then we lookup
           `tir.something` in the `Registry`. If the function is not in the
           registry, then we try to find a `tvm.ir.op.Op` with the same name.
        2. All other names `tvm.something` are lookup up in this current python
           namespace.
        """

        def get_full_attr_name(node: ast.Attr) -> str:
            reverse_field_names = [node.field.name]
            while isinstance(node.object, ast.Attr):
                node = node.object
                reverse_field_names.append(node.field.name)
            if isinstance(node.object, ast.Var):
                reverse_field_names.append(node.object.id.name)
            return ".".join(reversed(reverse_field_names))

        if isinstance(node.object, (ast.Var, ast.Attr)):
            full_attr_name = get_full_attr_name(node)
            attr_object, fields = full_attr_name.split(".", maxsplit=1)
            if self.match_tir_namespace(attr_object):
                func_name = "tir." + fields
                res = Registry.lookup(func_name)
                if res is not None:
                    return res
                try:
                    return tvm.ir.op.Op.get(func_name)
                except TVMError as e:
                    # Check if we got an attribute error
                    if e.args[0].find("AttributeError"):
                        self.report_error(f"Unregistered function `tir.{fields}`.", node.span)
                    else:
                        raise e

        symbol = self.transform(node.object)
        if symbol is None:
            self.report_error("Unsupported Attribute expression.", node.object.span)
        if not hasattr(symbol, node.field.name):
            self.report_error(
                f"Type {type(symbol)} does not have a field called `{node.field.name}`.", node.span
            )
        res = getattr(symbol, node.field.name)
        return res

    def transform_TypeAttr(self, node):
        """Visitor for field access of the form `x.y` for types.

        We have two cases here:
        1. If the type is of the form `T.something`, we look up the type in
           the `tir` namespace in this module.
        2. If the type is of the form `tvm.x.something` then we look up
           `tvm.x.something` in this modules namespace.
        """
        if isinstance(node.object, ast.TypeVar):
            if self.match_tir_namespace(node.object.id.name):
                if not hasattr(tir, node.field.name):
                    self.report_error(
                        f"Invalid type annotation `tir.{node.field.name}`.", node.span
                    )
                return getattr(tir, node.field.name)

        symbol = self.transform(node.object)
        if symbol is None:
            self.report_error("Unsupported Attribute expression", node.object.span)
        if not hasattr(symbol, node.field):
            self.report_error(
                f"Type {type(symbol)} does not have a field called `{node.field}`.", node.span
            )
        res = getattr(symbol, node.field)
        return res

    def transform_DictLiteral(self, node):
        """Dictionary literal visitor.

        Handles dictionary literals of the form `{x:y, z:2}`.
        """

        keys = [self.transform(key) for key in node.keys]
        values = [self.transform(value) for value in node.values]

        return dict(zip(keys, values))

    def transform_Tuple(self, node):
        """Tuple visitor.

        Handles tuples of the form `(x, y, 2)`.
        """

        return tuple(self.transform(element) for element in node.values)

    def transform_ArrayLiteral(self, node):
        """List literal visitor.

        Handles lists of the form `[x, 2, 3]`.
        """

        return [self.transform(element) for element in node.values]

    def transform_Var(self, node):
        """Variable visitor

        Handles variables like `x` in `x = 2`.
        """

        name = node.id.name
        if name == "meta":
            return self.meta
        symbol = Registry.lookup(name)
        if symbol is not None:
            return symbol
        symbol = self.context.lookup_symbol(name)
        if symbol is not None:
            return symbol
        self.report_error(f"Unknown identifier {name}.", node.span)

    def transform_TypeVar(self, node):
        """Type variable visitor.

        Equivalent to `transform_Var` but for types.
        """
        name = node.id.name
        symbol = Registry.lookup(name) or self.context.lookup_symbol(name)
        if symbol is not None:
            return symbol
        self.report_error(f"Unknown identifier {name}.", node.span)

    def transform_Constant(self, node):
        """Constant value visitor.

        Constant values include `None`, `"strings"`, `2` (integers), `4.2`
        (floats), and `true` (booleans).
        """
        return tvm.runtime.convert(node.value, span=tvm_span_from_synr(node.span))

    def transform_TypeConstant(self, node):
        """Constant value visitor for types.

        See `transform_Constant`.
        """
        if self._inside_buffer_sugar:
            return self.transform_Constant(node)

        return node.value

    def transform_TypeTuple(self, node):
        """Tuple value visitor for types.

        Mostly used in `transform_TypeCall` and `transform_TypeApply`.
        """
        return [self.transform(value) for value in node.values]

    def transform_TypeCall(self, node):
        """TypeCall visitor

        This occurs when an expression is used inside a T.Buffer
        parameter annotation.
        """

        # ast.Call has the BuiltinOp as node.func_name.name, where
        # ast.TypeCall has the BuiltinOp as node.func_name.  So we can
        # delegate to self.transform_Call, but the error messages for
        # unsupported operations will highlight the entire expression
        # and not just the function itself.
        op = ast.Op(node.span, node.func_name)
        call = ast.Call(node.span, op, node.params, node.keyword_params)
        return self.transform_Call(call)

    def transform_TypeApply(self, node):
        """Visitor for Type[Type] expressions.

        Mostly used for ``T.Ptr`` expressions.
        """
        func = self.transform(node.func_name)

        if not isinstance(func, ty.TypeGeneric) or not hasattr(func, "__getitem__"):
            self.report_error(
                f"Use of type arguments requires a type that accepts type arguments (e.g. T.Ptr), "
                f"but found {type(func).__name__} instead.",
                node.span,
            )

        param_types = []
        for idx, param in enumerate(node.params):
            param_type = self.transform(param)
            if not isinstance(param_type, ty.TypeGeneric) and func.require_type_generic_at(idx):
                self.report_error(
                    f"Expected a type but found {type(param).__name__} "
                    f"at {idx}th type argument",
                    param.span,
                )

            param_types.append(param_type)

        if len(param_types) == 1:
            return func[param_types[0]]
        else:
            return func[param_types]

    def handle_match_buffer_type(self, node, buffer_name):
        """special function to handle syntax sugar for match buffer.

        This method is for buffer declarations in the function parameters.
        """
        func = self.transform(node.func_name)
        assert isinstance(func, SpecialStmt)

        # parse args and kwargs for TypeCall and TypeApply
        self._inside_buffer_sugar = True
        try:
            arg_list = self.parse_arg_list(func, node)
        finally:
            self._inside_buffer_sugar = False

        # Note that the third element in arg_list would always be the 'name'
        # TODO: This index is hardcoded as a workaround. Better to make it programmatic
        if arg_list[2] is None:
            arg_list[2] = buffer_name
        buf = func.handle(node, self.context, arg_list, node.func_name.span)
        return buf

    def transform_Return(self, node):
        self.report_error(
            "TVM script does not support return statements. Instead the last statement in any "
            "block is implicitly returned.",
            node.span,
        )


def get_tir_namespace(script: Union[Callable, type]) -> List[str]:
    assert inspect.isfunction(script) or inspect.isclass(script)
    env: Dict[str, Any] = script.__globals__
    return [key for key in env.keys() if env[key] == tir]


def from_source(
    input_func: Union[str, Callable], tir_prefix: Optional[List[str]] = None
) -> Union[PrimFunc, IRModule]:
    """Parse function or string into PrimFunc or IRModule.

    If possible, pass the TVM script in as a function so that line numbers and
    filename will be accurate.

    Parameters
    ----------
    input_module : Union[str, Callable]
        The python function to be parsed.

    tir_prefix : Optional[List[str]]
        The tir prefix list. Only works for str input, default by "tir" and "T".

    Returns
    -------
    output : Union[Function, Module]
        The Function or Module in IR.
    """
    if isinstance(input_func, str):
        tir_prefix = ["T", "tir"] if tir_prefix is None else tir_prefix
        return to_ast(input_func, TVMDiagnosticCtx(), TVMScriptParser(0, tir_prefix, {}))
    elif inspect.isfunction(input_func):
        _, start_line = inspect.getsourcelines(input_func)
        env: Dict[str, Any] = input_func.__globals__
        namespace = [key for key in env.keys() if env[key] is tir]
        _closure_vars = inspect.getclosurevars(input_func)
        closure_vars = {**_closure_vars.nonlocals, **_closure_vars.globals}
        parser = TVMScriptParser(start_line, namespace, closure_vars)
        result = to_ast(input_func, TVMDiagnosticCtx(), parser)
        return result
    else:
        raise TypeError("Only function definitions are supported.")


def ir_module(input_module: type) -> IRModule:
    """Decorate a python class as tvm IRModule.

    Parameters
    ----------
    input_module : type
        The python class to be parsed.

    Returns
    -------
    output : IRModule
        The result IRModule.
    """
    if inspect.isclass(input_module):
        func_dict = {
            name: f for name, f in input_module.__dict__.items() if isinstance(f, BaseFunc)
        }
        return IRModule(func_dict)
    raise TypeError("Only class definitions are supported.")
