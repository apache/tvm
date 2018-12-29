"""Hybrid Script Parser"""

import ast
import operator
import logging
import sys
from numbers import Integral

from .util import _internal_assert
from . import calls
from . import util
from .var_decl import determine_variable_usage
from ..api import all as _all
from ..api import any as _any
from ..tensor import Tensor, Operation
from .. import expr as _expr
from .. import make as _make
from .. import api  as _api
from .. import ir_pass as _ir_pass

def list_to_block(visit, lst):
    """Convert a list of Python IR nodes to HalideIR Block"""
    lst = [visit(stmt) for stmt in lst if not util.is_docstring(stmt)]
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, util.make_nop())]
    if not lst:
        return util.make_nop()
    if len(lst) == 1:
        return lst[0]
    body = lst[0]
    for i in lst[1:]:
        body = _make.Block(body, i)
    return body


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
        self._args = {} # Dict maps arg name to actual arg instance (either a var or a buffer)
        self.alloc_buffers = {} # Buffers formed by explicit allocate instructions
        self.loops_above = {} # State variable that indicates loop levels above the current node
        self.variables = {} # The status of defined variables
        self.func_name = func_name # The name of the function to be lowered
        self.outputs = [] # Output tensors' name
        self.side_effect = set() # Tensors with side effects
        self.parsed_body = None # The parsed HalideIR body
        self.returned = False # If this function has a valid return
        self.symbols = symbols # The global context


    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        pop_buf = []
        pop_var = []
        for key, val in self.usage.items():
            _, level, _ = val
            if level != node:
                continue
            if key in self._args.keys():
                continue
            if key in self.alloc_buffers.keys():
                _buf, _scope = self.alloc_buffers[key]
                if _scope == 'output':
                    continue
                pop_buf.append(key)
            else:
                _internal_assert(key in self.variables.keys(),
                                 "Key should be either in one of args, buffers, and vars")
                if not isinstance(self.variables[key], tuple):
                    continue
                _buf, _scope = self.variables[key]
                pop_var.append(key)
            _domain = [_make.range_by_min_extent(0, i) for i in _buf.shape]
            _dtype = _buf.dtype
            _true = _api.convert(True)
            body = _make.Realize(_buf.op, 0, _dtype, _domain, _true, body)
            body = _make.AttrStmt(_buf.op, 'realize_scope', _api.convert(_scope), body)

        for elem in pop_buf:
            self.alloc_buffers.pop(elem)
        for elem in pop_var:
            self.variables.pop(elem)
        return body


    def _get_buffer_from_id(self, s, for_provide=False):
        _internal_assert((s in self._args.keys()) + (s in self.alloc_buffers.keys()) == 1,
                         "This %s is expected to be in either \
                          argument list or allocated buffer!" % s)
        if s in self._args.keys():
            if for_provide:
                self.side_effect.add(self._args[s])
            return self._args[s]
        return self.alloc_buffers[s][0]

    def _const(self, value, dtype=None):
        if dtype is None:
            if isinstance(value, bool):
                dtype = "bool"
            elif isinstance(value, Integral):
                dtype = "int32"
            else:
                dtype = "float32"
        return _api.const(value, dtype)

    #pylint: disable=invalid-name, missing-docstring
    def visit_Module(self, node):
        _internal_assert(len(node.body) == 1, \
                         "Only one-function source code can be fed to this parser!")
        return self.visit(node.body[0])


    def visit_FunctionDef(self, node):
        _internal_assert(len(node.args.args) == len(self.args), \
                         "The number of arguments passed to the \
                         function should be the same as it is defined!")
        if self.func_name is None:
            self.func_name = node.name
        for idx, arg in enumerate(node.args.args):
            _attr = 'id' if sys.version_info[0] < 3 else 'arg' # To make py2 and 3 compatible
            self._args[getattr(arg, _attr)] = self.args[idx]
        res = list_to_block(self.visit, node.body)
        res = self.wrap_up_realize(node, res)
        return res


    def visit_Expr(self, node):
        return self.visit(node.value)


    def visit_Name(self, node):
        name = node.id
        if name in self.loops_above.keys():
            return self.loops_above[name]
        elif name in self.variables.keys():
            res = self.variables[name]
            if isinstance(res, tuple):
                buf = res[0]
                if isinstance(node.ctx, ast.Load):
                    return _make.Call(buf.dtype, buf.name, [self._const(0)], \
                                      _expr.Call.Halide, buf.op, buf.value_index)
                return buf, [self._const(0)]
            if isinstance(node.ctx, ast.Load):
                return res
            return None
        buf = self._get_buffer_from_id(name)
        return buf


    def visit_Num(self, node):
        return self._const(node.n)


    def visit_AugAssign(self, node):
        buf = self.visit(node.target)
        rhs = self.visit(node.value)
        if isinstance(buf, tuple):
            _internal_assert(len(buf) == 2, "LHS is supposed to be (buf, args)!")
            buf, args = buf
        else:
            args = [self._const(0)]
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
                self.alloc_buffers[node.targets[i].id] = (rhs.output(i), 'global')
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
            _internal_assert(lhs not in self.loops_above.keys(), \
                             "Loop variable cannot be overwritten!")
            decl, _, rw = self.usage[lhs]
            if decl == lhs_:
                _internal_assert(lhs not in self.variables.keys() and
                                 lhs not in self.alloc_buffers.keys(), \
                                 "This value should not be defined before this point!")
                if isinstance(rhs, tuple):
                    shape, dtype, scope = rhs
                    ph = _api.placeholder(shape, dtype=dtype, name=lhs)
                    self.alloc_buffers[lhs] = (ph, scope)
                    if scope == 'output':
                        self.outputs.append(lhs)
                    return util.make_nop()
                if isinstance(rhs, util.halide_imm_types) and ast.Store not in rw:
                    self.variables[lhs] = rhs
                else:
                    ph = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                    self.variables[lhs] = (ph, 'global')
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
        buf = self._get_buffer_from_id(node.value.id)
        return getattr(buf, node.attr)


    def visit_Subscript(self, node):
        args = self.visit(node.slice)
        if isinstance(node.value, ast.Name):
            buf = self.visit(node.value)
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
        return list_to_block(self.visit, node.body)


    def visit_If(self, node):
        cond = self.visit(node.test)
        if_body = list_to_block(self.visit, node.body)
        if node.orelse:
            else_body = list_to_block(self.visit, node.orelse)
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
            outs = self.symbols[func_id](*args)
            op = outs.op if isinstance(outs, Tensor) else outs[0].op
            return op


    def visit_For(self, node):
        iter_var, low, ext, for_type = self.visit(node.iter)
        _internal_assert(isinstance(node.target, ast.Name), \
                         "The loop iterator should be a variable!")
        _name = node.target.id
        if iter_var is None:
            _internal_assert(for_type is not None, "The loop bind function parse error!")
            offset = iter_var = _api.var(_name)
            if not _ir_pass.Equal(low, self._const(0)):
                offset = iter_var + low
            self.loops_above[_name] = offset
        else:
            _internal_assert(for_type is None, "The loop iterating function parse error!")
            self.loops_above[_name] = iter_var.var
        _body = list_to_block(self.visit, node.body)
        _body = self.wrap_up_realize(node, _body)
        if for_type is None:
            res = _make.AttrStmt(iter_var, 'thread_extent', ext, _body)
        else:
            res = _make.For(iter_var, self._const(0), ext, for_type, 0, _body)
        self.loops_above.pop(_name)
        return res


    def visit_Return(self, node):
        _internal_assert(not self.loops_above, "Return should not be in a loop body!")
        ids = []
        if isinstance(node.value, ast.Name):
            ids.append(node.value.id)
        else:
            _internal_assert(isinstance(node.value, ast.Tuple), \
                             "You should return either a single tensor or a tuple")
            for i in node.value.elts:
                _internal_assert(isinstance(i, ast.Name), "What do you return?")
                ids.append(i.id)
        _internal_assert(len(set(ids)) == len(ids), "Duplicated tensors in the return tuples")
        if len(ids) < len(self.outputs):
            logging.log(logging.CRITICAL, '[Warning] Not all the output buffers returned!')
        self.outputs = [self.alloc_buffers[i][0] for i in ids]
        self.returned = True
        return util.make_nop()


    def visit_Tuple(self, node):
        return tuple(self.visit(i) for i in node.elts)


    def visit_Str(self, node):
        return node.s


def parse_python(src, symbols, args):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : str
        The source code of the function to be parsed.

    src : str
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
    root = ast.parse(src)
    var_usage = determine_variable_usage(root, args, symbols)
    parser = HybridParser(args, var_usage, symbols)
    parser.parsed_body = parser.visit(root)
    _internal_assert(parser.returned, 'No valid return found in the function body!')
    return parser
