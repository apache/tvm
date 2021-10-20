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
"""Hybrid Script Parser"""

import ast
import operator
import logging
import sys
import types
import numbers

from enum import Enum
from tvm.ir import Array, Range
import tvm.runtime
import tvm.tir
import tvm.te
import tvm.te._ffi_api
import tvm.arith

from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt
from tvm.te.tensor import Tensor, Operation
from tvm.tir import all as _all
from tvm.tir import any as _any

from .utils import _internal_assert
from . import calls
from . import utils
from .preprocessor import determine_variable_usage


def concat_list_to_block(lst):
    """Concatenate a list of Python IR nodes to HalideIR Block"""
    if not lst:
        return utils.make_nop()
    n = len(lst)
    if n == 1:
        return lst[0]
    return _stmt.SeqStmt(lst)


def visit_list_to_block(visit, lst):
    """Visit and concatenate a list of Python IR nodes to HalideIR Block"""
    lst = [visit(stmt) for stmt in lst if not utils.is_docstring(stmt)]
    lst = [stmt for stmt in lst if not tvm.ir.structural_equal(stmt, utils.make_nop())]
    if not lst:
        return utils.make_nop()
    return concat_list_to_block(lst)


class Symbol(Enum):
    """Enumerates types in the symbol table"""

    Callable = 0
    Input = 1
    OutputBuffer = 2
    GlobalBuffer = 3
    LocalBuffer = 4
    SharedBuffer = 5
    ConstVar = 6
    BufferVar = 7
    LoopVar = 8
    ConstLoopVar = 9
    ThreadBind = 10


def _floordiv(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return tvm.tir.floordiv(x, y)
    return operator.floordiv(x, y)


def _floormod(x, y):
    if isinstance(x, _expr.ExprOp) or isinstance(y, _expr.ExprOp):
        return tvm.tir.floormod(x, y)
    return operator.mod(x, y)


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to HalideIR"""

    _binop_maker = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.div if sys.version_info[0] == 2 else operator.truediv,
        ast.FloorDiv: _floordiv,
        ast.Mod: _floormod,
        ast.BitOr: operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.And: _all,
        ast.Or: _any,
    }

    _unaryop_maker = {ast.USub: operator.neg, ast.Invert: operator.invert, ast.Not: operator.not_}

    def __init__(self, args, usage, symbols, closure_vars, func_name=None):
        """
        Parameters
        ----------
        args: A list of tvm.te.placeholder or te.var
            Provided by the user, the argument list of the function to be lowered.

        usage: A dict of variables used in last in this function
            Provided by last lower pass, which collects this information

        symbols : list of str
            The symbol list of the global context of the function.

        closure_vars: dict
            A dict of external name reference captured by this function.

        Returns
        -------
        func_name: str
            The name of the function to be lowered; if not provided,
            the compiler will use the name in the AST
        """
        self.args = list(args)
        self.usage = usage.copy()

        self.symbols = {}  # Symbol table
        for k, v in symbols.items():
            if isinstance(v, types.FunctionType):
                self.add_symbol(k, Symbol.Callable, v)

        self.closure_vars = closure_vars

        self.binds = {}  # Thread binds
        self.device = 0  # Is it generating device

        self.func_name = func_name  # The name of the function to be lowered
        self.outputs = []  # Output tensors' name
        self.side_effect = set()  # Tensors with side effects
        self.parsed_body = None  # The parsed HalideIR body
        self.analyzer = tvm.arith.Analyzer()
        self.returned = False  # If this function has a valid return

    def add_symbol(self, key, ty, val):  # pylint: disable=invalid-name
        """Add value to the symbol table context"""
        if key in self.symbols.keys():
            old = str(self.symbols[key])
            new = str((ty, val))
            _internal_assert(
                False, "Name conflict in symbol table! [%s] %s -> %s" % (key, old, new)
            )

        self.symbols[key] = ty, val

        if ty == Symbol.ThreadBind:
            if val.var.name not in self.binds.keys():
                self.binds[val.var.name] = val
                return
            val_ = self.binds[val.var.name]
            _internal_assert(
                tvm.tir.analysis.expr_deep_equal(val_.dom.extent, val.dom.extent),
                "Thread extents should be uniform!",
            )
            self.symbols[key] = ty, val_

    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        to_pop = []
        for key, val in self.usage.items():
            _, level, _ = val
            if key not in self.symbols:
                # don't realize the symbols that are never visited
                continue
            if level != node:
                continue
            _internal_assert(key in self.symbols.keys(), "Unknown symbol %s!" % key)

            ty, entry = self.symbols[key]  # pylint: disable=invalid-name
            if ty in [Symbol.Input, Symbol.OutputBuffer]:
                continue
            if "Buffer" in ty.name:
                _buf = entry
                _scope = "global" if ty is Symbol.BufferVar else ty.name[:-6].lower()
                to_pop.append(key)
            else:
                continue

            if _scope == "global":
                body = self.wrap_up_binds(body)

            _domain = [Range.from_min_extent(0, i) for i in _buf.shape]
            _dtype = _buf.dtype
            _true = tvm.runtime.convert(True)
            body = tvm.tir.ProducerRealize(_buf, _domain, _true, body, tvm.runtime.convert(_scope))

        for elem in to_pop:
            self.symbols.pop(elem)

        return body

    def wrap_up_binds(self, body):
        for _, iter_var in self.binds.items():
            ext = iter_var.dom.extent
            body = tvm.tir.AttrStmt(iter_var, "thread_extent", ext, body)
        self.binds = {}
        return body

    # pylint: disable=invalid-name, missing-docstring
    def visit_Module(self, node):
        _internal_assert(
            len(node.body) == 1, "Only one-function source code will be fed to this parser!"
        )
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        _internal_assert(
            len(node.args.args) == len(self.args),
            "The number of arguments passed to the \
                         function should be the same as it is defined!",
        )
        if self.func_name is None:
            self.func_name = node.name
        for idx, arg in enumerate(node.args.args):
            _attr = "id" if sys.version_info[0] < 3 else "arg"  # To make py2 and 3 compatible
            self.add_symbol(getattr(arg, _attr), Symbol.Input, self.args[idx])
        res = visit_list_to_block(self.visit, node.body)
        res = self.wrap_up_realize(node, res)
        return self.wrap_up_binds(res)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        name = node.id
        if sys.version_info[0] == 2 and name in ["True", "False"]:
            return tvm.runtime.convert(ast.literal_eval(name))

        if name in self.closure_vars:
            return tvm.runtime.convert(self.closure_vars[name])

        ty, entry = self.symbols[name]
        _internal_assert(name in self.symbols, "Unknown symbol %s!" % name)
        if ty in [Symbol.LoopVar, Symbol.Input, Symbol.ConstLoopVar]:
            return entry
        if ty is Symbol.ThreadBind:
            return entry.var
        if ty is Symbol.ConstVar:
            return entry if isinstance(node.ctx, ast.Load) else None
        if ty is Symbol.BufferVar:
            if isinstance(node.ctx, ast.Load):
                return tvm.tir.ProducerLoad(entry, [tvm.runtime.const(0, "int32")])
            return entry, [tvm.runtime.const(0, "int32")]
        # Do I need any assertion here?
        return entry

    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            _internal_assert(
                isinstance(node.n, bool), "The data type should be one of (int, float, bool)"
            )
            dtype = "bool"
        return tvm.runtime.const(node.n, dtype)

    def visit_NameConstant(self, node):
        return tvm.runtime.convert(node.value)

    def visit_AugAssign(self, node):
        buf = self.visit(node.target)
        rhs = self.visit(node.value)
        if isinstance(buf, tuple):
            _internal_assert(len(buf) == 2, "LHS is supposed to be (buf, args)!")
            buf, args = buf
        else:
            args = [tvm.runtime.const(0, "int32")]
        _internal_assert(isinstance(buf, Tensor), "LHS is supposed to be Tensor!")

        read = tvm.tir.ProducerLoad(buf, args)
        value = HybridParser._binop_maker[type(node.op)](read, rhs)

        return tvm.tir.ProducerStore(buf, value, args)

    def visit_Assign(self, node):
        rhs = self.visit(node.value)
        if isinstance(rhs, Operation):
            rmap = {}
            _internal_assert(
                len(node.targets) == rhs.num_outputs, "Unable to detuple the outs to targets"
            )
            for i in range(rhs.num_outputs):
                _internal_assert(
                    isinstance(node.targets[i], ast.Name),
                    "You should bind a pure name to the tensors",
                )
                self.add_symbol(node.targets[i].id, Symbol.GlobalBuffer, rhs.output(i))
                rmap[rhs.outputs[i].op] = rhs.output(i)
            return utils.replace_io(rhs.body, rmap)

        _internal_assert(len(node.targets) == 1, "So far only one-valued assignment is supported!")
        lhs = node.targets[0]
        if isinstance(rhs, _expr.PrimExpr):
            rhs = self.analyzer.simplify(rhs)
        if isinstance(lhs, ast.Name):
            # TODO: support defined intermediate buffer later
            lhs_ = lhs
            lhs = lhs.id
            if lhs in self.symbols.keys():
                ty, _ = self.symbols[lhs]
                _internal_assert(ty != Symbol.LoopVar, "Loop variable cannot be overwritten!")
            decl, _, rw = self.usage[lhs]
            if decl == lhs_:
                _internal_assert(
                    lhs not in self.symbols.keys(),
                    "This value should not be defined before this point!",
                )
                if isinstance(rhs, tuple):
                    shape, dtype, scope = rhs
                    ph = tvm.te.placeholder(shape, dtype=dtype, name=lhs)
                    self.add_symbol(lhs, getattr(Symbol, scope.title() + "Buffer"), ph)
                    if scope == "output":
                        self.outputs.append(lhs)
                    return utils.make_nop()
                if isinstance(rhs, utils.halide_imm_types) and ast.Store not in rw:
                    self.add_symbol(lhs, Symbol.ConstVar, rhs)
                else:
                    _internal_assert(
                        self.device == 0,
                        "Single variable not supported in devices' side!\n"
                        + "If you are using GPU, please allocate a 'local' spad "
                        + "outside the bind body",
                    )
                    ph = tvm.te.placeholder((1,), dtype=rhs.dtype, name=lhs)
                    self.add_symbol(lhs, Symbol.BufferVar, ph)
            lhs = self.visit(lhs_)
            if lhs is not None:
                buf, args = lhs
                return tvm.tir.ProducerStore(buf, rhs, args)
            return utils.make_nop()

        lhs, args = self.visit(lhs)
        _internal_assert(
            isinstance(lhs, Tensor), "An array access's LHS is expected to be a expr.Call!"
        )
        res = tvm.tir.ProducerStore(lhs, rhs, args)
        return res

    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return self.visit(node.value)
        return [self.visit(node.value)]

    def visit_Attribute(self, node):
        buf = self.visit(node.value)
        return getattr(buf, node.attr)

    def visit_Subscript(self, node):
        args = self.visit(node.slice)
        arr = self.visit(node.value)
        if isinstance(arr, Array):
            for i in args:
                if isinstance(i, numbers.Integral):
                    arr = arr[i]
                else:
                    _internal_assert(
                        isinstance(i, (_expr.IntImm,)), "All indices are supposed to be constants"
                    )
                    arr = arr[i.value]
            return arr
        if isinstance(node.ctx, ast.Load):
            return tvm.tir.ProducerLoad(arr, args)
        return arr, args

    def visit_With(self, node):
        if sys.version_info[0] < 3:
            context = node.context_expr
            option = node.optional_vars
        else:
            _internal_assert(len(node.items) == 1, "Only one with element is supported so far!")
            context = node.items[0].context_expr
            option = node.items[0].optional_vars
        _internal_assert(isinstance(context, ast.Call), "The object must be a Python func call!")
        _internal_assert(isinstance(option, ast.Name), "The object after 'as' must be an id!")
        self.annotation[option.id] = context.func.id
        return visit_list_to_block(self.visit, node.body)

    def visit_If(self, node):
        cond = self.analyzer.simplify(self.visit(node.test))

        # Return no IfThenElse if proven
        if isinstance(cond, _expr.IntImm):
            if cond.value:
                return visit_list_to_block(self.visit, node.body)
            if node.orelse:
                return visit_list_to_block(self.visit, node.orelse)
            return utils.make_nop()

        if_body = visit_list_to_block(self.visit, node.body)

        if node.orelse:
            else_body = visit_list_to_block(self.visit, node.orelse)
        else:
            else_body = None
        return tvm.tir.IfThenElse(cond, if_body, else_body)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if_body = self.visit(node.body)
        else_body = self.visit(node.orelse)
        return tvm.tir.Select(cond, if_body, else_body)

    def visit_Compare(self, node):
        _internal_assert(len(node.ops) == len(node.comparators), "#compare ops != #comparators")
        ops = [self.visit(node.left)]
        ops += [self.visit(i) for i in node.comparators]
        res = []
        for i in range(len(node.ops)):
            lhs = ops[i]
            rhs = ops[i + 1]
            res.append(HybridParser._binop_maker[type(node.ops[i])](lhs, rhs))
        return _all(*res)

    def visit_BoolOp(self, node):
        n = len(node.values)
        if n == 1:
            _internal_assert(isinstance(node.op, ast.Not), "Unary is supposed to be not!")
            return operator.not_(self.visit(node.values[0]))
        _internal_assert(isinstance(node.op, (ast.And, ast.Or)), "Binary is supposed to be and/or!")
        values = [self.visit(i) for i in node.values]
        return HybridParser._binop_maker[type(node.op)](*values)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return HybridParser._unaryop_maker[type(node.op)](operand)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return HybridParser._binop_maker[type(node.op)](lhs, rhs)

    def visit_Call(self, node):
        # Yet, no function pointer supported
        _internal_assert(
            isinstance(node.func, ast.Name), "Only id-function function call is supported so far!"
        )

        func_id = node.func.id
        args = [self.visit(i) for i in node.args]
        # Intrinsics'
        if hasattr(calls, func_id):
            return getattr(calls, func_id)(func_id, args)
        # Contexts'
        _internal_assert(
            func_id in self.symbols.keys(),
            "The function called (%s) is not in the context either!" % func_id,
        )
        ty, entry = self.symbols[func_id]
        _internal_assert(ty is Symbol.Callable, "Are you sure what you call is a function?!")
        outs = entry(*args)
        op = outs.op if isinstance(outs, Tensor) else outs[0].op
        return op

    def visit_For(self, node):
        iter_var, low, ext, kind = self.visit(node.iter)
        _internal_assert(
            isinstance(node.target, ast.Name), "The loop iterator should be a variable!"
        )

        _name = node.target.id

        if isinstance(kind, tuple):
            low = self.analyzer.simplify(low)
            ext = self.analyzer.simplify(ext)
            _internal_assert(
                isinstance(low, _expr.ConstExpr) and isinstance(ext, _expr.ConstExpr),
                "Const range should start from a const " + "and iterate const times",
            )

            low, ext = low.value, ext.value
            if ext > 114514:
                logging.log(
                    logging.CRITICAL, "[Warning] Are you sure to unroll a large loop in Python?"
                )

            bodies = []
            for i in range(low, low + ext):
                self.add_symbol(_name, Symbol.ConstLoopVar, i)
                body = visit_list_to_block(self.visit, node.body)
                body = self.wrap_up_realize(node, body)
                bodies.append(body)
                self.symbols.pop(_name)
            return concat_list_to_block(bodies)

        if iter_var is None:
            _internal_assert(kind is not None, "The loop iterating function parse error!")
            offset = iter_var = tvm.te.var(_name)
            if not tvm.tir.analysis.expr_deep_equal(low, tvm.runtime.const(0, "int32")):
                offset = iter_var + low
            self.add_symbol(_name, Symbol.LoopVar, offset)
            _body = visit_list_to_block(self.visit, node.body)
        else:
            _internal_assert(kind is None, "The loop bind function parse error!")
            self.add_symbol(_name, Symbol.ThreadBind, iter_var)
            self.device += 1
            _body = visit_list_to_block(self.visit, node.body)
            self.device -= 1

        _body = self.wrap_up_realize(node, _body)

        if kind is None:
            res = _body
        else:
            _internal_assert(
                not isinstance(kind, tuple), "Micro expansion should be handled before!"
            )
            res = tvm.tir.For(iter_var, tvm.runtime.const(0, "int32"), ext, kind, _body)

        self.symbols.pop(_name)
        return res

    def visit_Return(self, node):
        _internal_assert(
            all(ty != Symbol.LoopVar for ty, _ in self.symbols.values()),
            "Return should not be in a loop body!",
        )
        ids = []
        if isinstance(node.value, ast.Name):
            ids = [node.value.id]
        else:
            _internal_assert(
                isinstance(node.value, ast.Tuple),
                "You should return either a single tensor or a tuple",
            )
            _internal_assert(
                all(isinstance(i, ast.Name) for i in node.value.elts), "What do you return?"
            )
            ids = [i.id for i in node.value.elts]
        _internal_assert(len(set(ids)) == len(ids), "Duplicated tensors in the return tuples")
        if len(ids) < len(self.outputs):
            logging.log(logging.CRITICAL, "[Warning] Not all the output buffers returned!")
        self.outputs = [self.symbols[i][1] for i in ids]
        self.returned = True
        return utils.make_nop()

    def visit_Tuple(self, node):
        return tuple(self.visit(i) for i in node.elts)

    def visit_Str(self, node):
        return node.s

    def visit_Assert(self, node):
        test = self.visit(node.test)
        mesg = tvm.runtime.convert(self.visit(node.msg))
        return tvm.tir.AssertStmt(test, mesg, utils.make_nop())


def parse_python(src, args, symbols, closure_vars):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    symbols : list of str
        The symbol list of the global context of the function.

    closure_vars: dict
        A dict of external name reference captured by this function.

    Returns
    -------
    root : Stmt
        The result Halide IR and the parser class instance.
    """
    root = ast.parse(src) if isinstance(src, str) else src
    _internal_assert(root, ast.AST)
    var_usage = determine_variable_usage(root, args, symbols, closure_vars)
    parser = HybridParser(args, var_usage, symbols, closure_vars)
    parser.parsed_body = parser.visit(root)
    _internal_assert(parser.returned, "No valid return found in the function body!")
    return parser


def source_to_op(src, args, symbols, closure_vars):
    """Another level of wrapper

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    symbols : list of str
        The symbol list of the global context of the function.

    closure_vars: dict
        A dict of external name reference captured by this function.

    Returns
    -------
    res : list of output tensors
        The result of output tensors of the formed OpNode.
    """
    parser = parse_python(src, args, symbols, closure_vars)

    input_tensors = []

    def get_input_tensors(arg):
        if isinstance(arg, Tensor):
            input_tensors.append(arg)
        elif isinstance(arg, Array):
            for i in arg:
                get_input_tensors(i)

    for i in args:
        get_input_tensors(i)
    op = tvm.te._ffi_api.HybridOp(
        parser.func_name, "HybridOp", None, input_tensors, parser.outputs, parser.parsed_body
    )
    res = [op.output(i) for i in range(len(parser.outputs))]
    return res[0] if len(res) == 1 else res
