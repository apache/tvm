"""Hybrid Script Parser"""

import ast
import operator
import logging
import sys
import types
import numbers

from enum import Enum

from .util import _internal_assert
from . import calls
from . import util
from .var_decl import determine_variable_usage
from ..api import all as _all
from ..api import any as _any
from ..container import Array
from ..tensor import Tensor, Operation
from .. import _api_internal as _tvm_internal
from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass as _ir_pass


def concat_list_to_block(lst):
    """Concatenate a list of Python IR nodes to HalideIR Block"""
    n = len(lst)
    if n == 1:
        return lst[0]
    body = lst[n - 1]
    for i in range(1, n):
        stmt = lst[n - 1 - i]
        if isinstance(stmt, _stmt.AssertStmt):
            body = _make.AssertStmt(stmt.condition, stmt.message, body)
        else:
            body = _make.Block(stmt, body)
    return body


def visit_list_to_block(visit, lst):
    """Visit and concatenate a list of Python IR nodes to HalideIR Block"""
    lst = [visit(stmt) for stmt in lst if not util.is_docstring(stmt)]
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, util.make_nop())]
    if not lst:
        return util.make_nop()
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


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to HalideIR"""


    _binop_maker = {
        ast.Add     : operator.add,
        ast.Sub     : operator.sub,
        ast.Mult    : operator.mul,
        ast.Div     : operator.div if sys.version_info[0] == 2 else operator.truediv,
        ast.FloorDiv: operator.div if sys.version_info[0] == 2 else operator.truediv,
        ast.Mod     : operator.mod,
        ast.BitOr   : operator.or_,
        ast.BitAnd  : operator.and_,
        ast.BitXor  : operator.xor,
        ast.Gt      : operator.gt,
        ast.GtE     : operator.ge,
        ast.Lt      : operator.lt,
        ast.LtE     : operator.le,
        ast.Eq      : operator.eq,
        ast.NotEq   : operator.ne,
        ast.And   : _all,
        ast.Or    : _any,
    }


    _unaryop_maker = {
        ast.USub   : operator.neg,
        ast.Invert : operator.invert,
        ast.Not    : operator.not_
    }


    def __init__(self, args, usage, symbols, func_name=None):
        """
        Parameters
        ----------
        args: A list of tvm.placeholder or tvm.var
            Provided by the user, the argument list of the function to be lowered.

        usage: A dict of variables used in last in this function
            Provided by last lower pass, which collects this information

        Returns
        -------
        func_name: str
            The name of the function to be lowered; if not provided,
            the compiler will use the name in the AST
        """
        self.args = list(args)
        self.usage = usage.copy()

        self.symbols = {} # Symbol table
        for k, v in symbols.items():
            if isinstance(v, types.FunctionType):
                self.symbols[k] = Symbol.Callable, v

        self.func_name = func_name # The name of the function to be lowered
        self.outputs = [] # Output tensors' name
        self.side_effect = set() # Tensors with side effects
        self.parsed_body = None # The parsed HalideIR body
        self.returned = False # If this function has a valid return



    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        to_pop = []
        for key, val in self.usage.items():
            _, level, _ = val
            if level != node:
                continue
            _internal_assert(key in self.symbols.keys(), "Unknown symbol %s!" % key)

            ty, entry = self.symbols[key] #pylint: disable=invalid-name
            if ty in [Symbol.Input, Symbol.OutputBuffer]:
                continue
            elif 'Buffer' in ty.name:
                _buf = entry
                _scope = ty.name[:-6].lower() if ty is not Symbol.BufferVar else 'global'
                to_pop.append(key)
            else:
                continue

            _domain = [_make.range_by_min_extent(0, i) for i in _buf.shape]
            _dtype = _buf.dtype
            _true = _api.convert(True)
            body = _make.Realize(_buf.op, 0, _dtype, _domain, _true, body)
            body = _make.AttrStmt(_buf.op, 'realize_scope', _api.convert(_scope), body)

        for elem in to_pop:
            self.symbols.pop(elem)

        return body


    #pylint: disable=invalid-name, missing-docstring
    def visit_Module(self, node):
        _internal_assert(len(node.body) == 1, \
                         "Only one-function source code will be fed to this parser!")
        return self.visit(node.body[0])


    def visit_FunctionDef(self, node):
        _internal_assert(len(node.args.args) == len(self.args), \
                         "The number of arguments passed to the \
                         function should be the same as it is defined!")
        if self.func_name is None:
            self.func_name = node.name
        for idx, arg in enumerate(node.args.args):
            _attr = 'id' if sys.version_info[0] < 3 else 'arg' # To make py2 and 3 compatible
            self.symbols[getattr(arg, _attr)] = (Symbol.Input, self.args[idx])
        res = visit_list_to_block(self.visit, node.body)
        res = self.wrap_up_realize(node, res)
        return res


    def visit_Expr(self, node):
        return self.visit(node.value)


    def visit_Name(self, node):
        name = node.id
        ty, entry = self.symbols[name]
        _internal_assert(name in self.symbols, "Unknown symbol %s!" % name)
        if ty in [Symbol.LoopVar, Symbol.Input, Symbol.ConstLoopVar]:
            return entry
        elif ty is Symbol.ConstVar:
            return entry if isinstance(node.ctx, ast.Load) else None
        elif ty is Symbol.BufferVar:
            if isinstance(node.ctx, ast.Load):
                return _make.Call(entry.dtype, entry.name, [_api.const(0, 'int32')], \
                                  _expr.Call.Halide, entry.op, entry.value_index)
            return entry, [_api.const(0, 'int32')]
        # Do I need any assertion here?
        return entry


    def visit_Num(self, node):
        if isinstance(node.n, numbers.Integral):
            dtype = "int32"
        elif isinstance(node.n, float):
            dtype = "float32"
        else:
            _internal_assert(isinstance(node.n, bool),
                             "The data type should be one of (int, float, bool)")
            dtype = "bool"
        return _api.const(node.n, dtype)


    def visit_AugAssign(self, node):
        buf = self.visit(node.target)
        rhs = self.visit(node.value)
        if isinstance(buf, tuple):
            _internal_assert(len(buf) == 2, "LHS is supposed to be (buf, args)!")
            buf, args = buf
        else:
            args = [_api.const(0, 'int32')]
        _internal_assert(isinstance(buf, Tensor), "LHS is supposed to be Tensor!")

        read = _make.Call(buf.dtype, buf.name, args, _expr.Call.Halide, buf.op, buf.value_index)
        value = HybridParser._binop_maker[type(node.op)](read, rhs)

        return _make.Provide(buf.op, 0, value, args)


    def visit_Assign(self, node):
        rhs = self.visit(node.value)
        if isinstance(rhs, Operation):
            rmap = {}
            _internal_assert(len(node.targets) == rhs.num_outputs, \
                             "Unable to detuple the outs to targets")
            for i in range(rhs.num_outputs):
                _internal_assert(isinstance(node.targets[i], ast.Name),
                                 "You should bind a pure name to the tensors")
                self.symbols[node.targets[i].id] = Symbol.GlobalBuffer, rhs.output(i)
                rmap[rhs.outputs[i].op] = rhs.output(i)
            return util.replace_io(rhs.body, rmap)

        _internal_assert(len(node.targets) == 1, "So far only one-valued assignment is supported!")
        lhs = node.targets[0]
        if isinstance(rhs, _expr.Expr):
            rhs = _ir_pass.Simplify(rhs)
        if isinstance(lhs, ast.Name):
            #TODO: support defined intermediate buffer later
            lhs_ = lhs
            lhs = lhs.id
            if lhs in self.symbols.keys():
                ty, _ = self.symbols[lhs]
                _internal_assert(ty != Symbol.LoopVar, \
                                 "Loop variable cannot be overwritten!")
            decl, _, rw = self.usage[lhs]
            if decl == lhs_:
                _internal_assert(lhs not in self.symbols.keys(),
                                 "This value should not be defined before this point!")
                if isinstance(rhs, tuple):
                    shape, dtype, scope = rhs
                    ph = _api.placeholder(shape, dtype=dtype, name=lhs)
                    self.symbols[lhs] = getattr(Symbol, scope.title() + "Buffer"), ph
                    if scope == 'output':
                        self.outputs.append(lhs)
                    return util.make_nop()
                if isinstance(rhs, util.halide_imm_types) and ast.Store not in rw:
                    self.symbols[lhs] = Symbol.ConstVar, rhs
                else:
                    ph = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                    self.symbols[lhs] = Symbol.BufferVar, ph
            lhs = self.visit(lhs_)
            if lhs is not None:
                buf, args = lhs
                return _make.Provide(buf.op, 0, rhs, args)
            return util.make_nop()
        else:
            lhs, args = self.visit(lhs)
            _internal_assert(isinstance(lhs, Tensor), \
                             "An array access's LHS is expected to be a expr.Call!")
            res = _make.Provide(lhs.op, lhs.value_index, rhs, args)
            return res


    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return self.visit(node.value)
        return [self.visit(node.value)]


    def visit_Attribute(self, node):
        _internal_assert(isinstance(node.value, ast.Name), \
                         "For atrribute access, only both names are supported so far!")
        buf = self.visit(node.value)
        return getattr(buf, node.attr)


    def visit_Subscript(self, node):
        args = self.visit(node.slice)
        if isinstance(node.value, ast.Name):

            buf = self.visit(node.value)
            if isinstance(buf, Array):
                for i in args:
                    if isinstance(i, numbers.Integral):
                        buf = buf[i]
                    else:
                        _internal_assert(isinstance(i, (_expr.IntImm, _expr.UIntImm)), \
                                         "All indices are supposed to be constants")
                        buf = buf[i.value]

                return buf

            if isinstance(node.ctx, ast.Load):
                return _make.Call(buf.dtype, buf.name, args, \
                                  _expr.Call.Halide, buf.op, buf.value_index)

            return buf, args

        shape = self.visit(node.value)
        _internal_assert(len(args) == 1, "For 'shape' access the argument should be only one!")
        args = args[0]
        #TODO: maybe support non-constant value later?
        _internal_assert(isinstance(args, (_expr.IntImm, _expr.UIntImm)), \
                         "So far only constant shape access supported!")
        return shape[args.value]


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
        cond = self.visit(node.test)

        # Return no IfThenElse if proven
        if isinstance(cond, _expr.UIntImm):
            if cond.value:
                return visit_list_to_block(self.visit, node.body)
            elif node.orelse:
                return visit_list_to_block(self.visit, node.orelse)
            return util.make_nop()

        if_body = visit_list_to_block(self.visit, node.body)

        if node.orelse:
            else_body = visit_list_to_block(self.visit, node.orelse)
        else:
            else_body = util.make_nop()
        return _make.IfThenElse(cond, if_body, else_body)


    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if_body = self.visit(node.body)
        else_body = self.visit(node.orelse)
        return _make.Select(cond, if_body, else_body)


    def visit_Compare(self, node):
        _internal_assert(len(node.ops) == len(node.comparators),
                         "#compare ops != #comparators")
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
            _internal_assert(isinstance(node.op, ast.Not), \
                             "Unary is supposed to be not!")
            return operator.not_(self.visit(node.values[0]))
        _internal_assert(isinstance(node.op, (ast.And, ast.Or)), \
                         "Binary is supposed to be and/or!")
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
        _internal_assert(isinstance(node.func, ast.Name), \
                         "Only id-function function call is supported so far!")

        func_id = node.func.id
        args = [self.visit(i) for i in node.args]
        try:
            return getattr(calls, func_id)(func_id, args)
        except AttributeError:
            _internal_assert(func_id in self.symbols.keys(), \
                             "The function called is not in the context either!")
            ty, entry = self.symbols[func_id]
            _internal_assert(ty is Symbol.Callable, \
                             "Are you sure what you call is a function?!")
            outs = entry(*args)
            op = outs.op if isinstance(outs, Tensor) else outs[0].op
            return op


    def visit_For(self, node):
        iter_var, low, ext, for_type = self.visit(node.iter)
        _internal_assert(isinstance(node.target, ast.Name), \
                         "The loop iterator should be a variable!")

        _name = node.target.id

        if isinstance(for_type, tuple):
            low = _ir_pass.Simplify(low)
            ext = _ir_pass.Simplify(ext)
            _internal_assert(isinstance(low, _expr.ConstExpr) and
                             isinstance(ext, _expr.ConstExpr), \
                             "Const range should start from a const" + \
                             "and iterate const times")

            low, ext = low.value, ext.value
            if ext > 114514:
                logging.log(logging.CRITICAL, \
                            '[Warning] Are you sure to unroll a large loop in Python?')

            bodies = []
            for i in range(low, low + ext):
                self.symbols[_name] = Symbol.ConstLoopVar, i
                body = visit_list_to_block(self.visit, node.body)
                body = self.wrap_up_realize(node, body)
                bodies.append(body)
            return concat_list_to_block(bodies)

        elif iter_var is None:
            _internal_assert(for_type is not None, "The loop bind function parse error!")
            offset = iter_var = _api.var(_name)
            if not _ir_pass.Equal(low, _api.const(0, 'int32')):
                offset = iter_var + low
            self.symbols[_name] = Symbol.LoopVar, offset
            _body = visit_list_to_block(self.visit, node.body)
        else:
            _internal_assert(for_type is None, "The loop iterating function parse error!")
            self.symbols[_name] = Symbol.LoopVar, iter_var.var
            _body = visit_list_to_block(self.visit, node.body)

        _body = self.wrap_up_realize(node, _body)

        if for_type is None:
            res = _make.AttrStmt(iter_var, 'thread_extent', ext, _body)
        else:
            _internal_assert(not isinstance(for_type, tuple), \
                            "Micro expansion should be handled before!")
            res = _make.For(iter_var, _api.const(0, 'int32'), ext, for_type, 0, _body)

        self.symbols.pop(_name)
        return res


    def visit_Return(self, node):
        _internal_assert(all(ty != Symbol.LoopVar for ty, _ in self.symbols.values()), \
                         "Return should not be in a loop body!")
        ids = []
        if isinstance(node.value, ast.Name):
            ids = [node.value.id]
        else:
            _internal_assert(isinstance(node.value, ast.Tuple), \
                             "You should return either a single tensor or a tuple")
            _internal_assert(all(isinstance(i, ast.Name) for i in node.value.elts), \
                             "What do you return?")
            ids = [i.id for i in node.value.elts]
        _internal_assert(len(set(ids)) == len(ids), "Duplicated tensors in the return tuples")
        if len(ids) < len(self.outputs):
            logging.log(logging.CRITICAL, '[Warning] Not all the output buffers returned!')
        self.outputs = [self.symbols[i][1] for i in ids]
        self.returned = True
        return util.make_nop()


    def visit_Tuple(self, node):
        return tuple(self.visit(i) for i in node.elts)


    def visit_Str(self, node):
        return node.s


    def visit_Assert(self, node):
        test = self.visit(node.test)
        mesg = _api.convert(self.visit(node.msg))
        return _make.AssertStmt(test, mesg, util.make_nop())


def parse_python(src, symbols, args):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    symbols : str
        The symbol list of the global context of the function.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    Returns
    -------
    root : Stmt
        The result Halide IR and the parser class instance.
    """
    root = ast.parse(src) if isinstance(src, str) else src
    _internal_assert(root, ast.AST)
    var_usage = determine_variable_usage(root, args, symbols)
    parser = HybridParser(args, var_usage, symbols)
    parser.parsed_body = parser.visit(root)
    _internal_assert(parser.returned, 'No valid return found in the function body!')
    return parser


def source_to_op(src, symbols, args):
    """Another level of wrapper

    Parameters
    ----------
    src : ast.node or str
        If an ast.node, then directly lower it.
        If a str, then parse it to ast and lower it.

    symbols : str
        The symbol list of the global context of the function.

    args : list of Tensors or Vars
        The argument lists to the function.
        It is NOT encouraged to write a function without arguments.
        It is NOT encouraged to write a function with side effect.

    Returns
    -------
    res : list of output tensors
        The result of output tensors of the formed OpNode.
    """
    parser = parse_python(src, symbols, args)

    input_tensors = []
    for i in args:
        if isinstance(i, Tensor):
            input_tensors.append(i)
    op = _tvm_internal._HybridOp(parser.func_name, "HybridOp", None, input_tensors,
                                 parser.outputs, parser.parsed_body)
    res = [op.output(i) for i in range(len(parser.outputs))]
    return res[0] if len(res) == 1 else res
