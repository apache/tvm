"""Hybrid Script Parser"""

import ast
import operator
import logging
import sys
from .util import make_nop, halide_imm_types, is_docstring, _internal_assert
from .intrin import LOOP_INTRIN, MATH_INTRIN
from .var_decl import determine_variable_usage
from ..api import thread_axis
from ..api import all as _all
from ..api import any as _any
from .. import expr as _expr
from .. import make as _make
from .. import intrin
from .. import api  as _api
from .. import ir_pass as _ir_pass

def list_to_block(visit, lst):
    """Convert a list of Python IR nodes to HalideIR Block"""
    lst = [visit(stmt) for stmt in lst if not is_docstring(stmt)]
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, make_nop())]
    if not lst:
        return make_nop()
    if len(lst) == 1:
        return lst[0]
    body = lst[0]
    for i in lst[1:]:
        body = _make.Block(body, i)
    return body


class HybridParser(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to HalideIR"""


    _binop_maker = {
        ast.Add   : operator.add,
        ast.Sub   : operator.sub,
        ast.Mult  : operator.mul,
        ast.Div   : operator.div if sys.version_info[0] == 2 else operator.truediv,
        ast.Mod   : operator.mod,
        ast.BitOr : operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor,
        ast.Gt    : operator.gt,
        ast.GtE   : operator.ge,
        ast.Lt    : operator.lt,
        ast.LtE   : operator.le,
        ast.Eq    : operator.eq,
        ast.NotEq : operator.ne,
        ast.And   : _all,
        ast.Or    : _any,
    }


    _unaryop_maker = {
        ast.USub   : operator.neg,
        ast.Invert : operator.invert,
        ast.Not    : operator.not_
    }


    def __init__(self, args, usage, func_name=None):
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
        self.alloc_buffers = {} # Buffers formed by allocate instructions
        self.loops_above = {} # State variable that indicates loop levels above the current node
        self.var_consts = {} # Variables that are determined as readonly in previous stage
        self.func_name = func_name # The name of the function to be lowered
        self.outputs = [] # Output tensors' name
        self.side_effect = set() # Tensors with side effects
        self.parsed_body = None # The parsed HalideIR body
        self.returned = False


    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        for key, val in self.usage.items():
            if key in self.var_consts.keys():
                continue
            _, level, _ = val
            if level == node:
                if key in self._args.keys():
                    continue
                else:
                    _buf, _scope = self.alloc_buffers[key]
                _domain = [_make.range_by_min_extent(0, i) for i in _buf.shape]
                _dtype = _buf.dtype
                _true = _api.convert(True)
                body = _make.Realize(_buf.op, 0, _dtype, _domain, _true, body)
                body = _make.AttrStmt(_buf.op, 'realize_scope', _api.convert(_scope), body)
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



    #pylint: disable=invalid-name, missing-docstring
    def visit_Module(self, node):
        _internal_assert(len(node.body) == 1, \
                         "Only one-function source code can be fed to this parser!")
        return self.visit(node.body[0])


    def visit_FunctionDef(self, node):
        _internal_assert(len(node.args.args) == len(self.args), \
                         "The number of arguments passed to the \
                         function should be the same as it is defined!")
        for idx, arg in enumerate(node.args.args):
            _attr = 'id' if sys.version_info[0] < 3 else 'arg' # To make py2 and 3 compatible
            self._args[getattr(arg, _attr)] = self.args[idx]
        res = list_to_block(self.visit, node.body)
        res = self.wrap_up_realize(node, res)
        if self.func_name is None:
            self.func_name = node.name
        return res


    def visit_Expr(self, node):
        return self.visit(node.value)


    def visit_Name(self, node):
        _id = node.id
        if _id in self._args.keys() and isinstance(self._args[_id], (_expr.Var, _expr.ConstExpr)):
            return self._args[_id]
        elif _id in self.loops_above.keys():
            return self.loops_above[_id]
        _internal_assert(_id not in self._args.keys(), \
                         "This id %s should be handled in visit_Subscript!" % _id)
        _internal_assert(_id in self.usage.keys(), \
                         "This id %s is expected to be a defined variable!" % _id)
        # Buffer
        if _id in self.alloc_buffers.keys():
            _buf, _ = self.alloc_buffers[_id]
            return _make.Call(_buf.dtype, _id, [_api.const(0)], _expr.Call.Halide, _buf.op, 0)
        # Compilation time constant
        _internal_assert(_id in self.var_consts.keys(),
                         "This id %s is expected to a compilation time constant!" % _id)
        return self.var_consts[_id]


    def visit_Num(self, node):
        return _api.const(node.n)


    def visit_AugAssign(self, node):
        lhs = self.visit(node.target)
        rhs = self.visit(node.value)
        rhs = HybridParser._binop_maker[type(node.op)](lhs, rhs)
        _internal_assert(isinstance(lhs, _expr.Call), \
                         "The LHS of an AugAssign is supposed to be a call!")
        return _make.Provide(lhs.func, 0, rhs, lhs.args)


    def visit_Assign(self, node):
        _internal_assert(len(node.targets) == 1, "So far only one-valued assignment is supported!")
        lhs = node.targets[0]
        rhs = self.visit(node.value)
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
                _internal_assert(lhs not in self.var_consts.keys(), \
                                 "A constant cannot be overwritten!")
                _internal_assert(lhs not in self.alloc_buffers.keys(), \
                                 "This value should not be defined before this point!")
                if isinstance(rhs, tuple):
                    shape, dtype, scope = rhs
                    ph = _api.placeholder(shape, dtype=dtype, name=lhs)
                    if scope != 'output':
                        self.alloc_buffers[lhs] = (ph, scope)
                    else:
                        self._args[lhs] = ph
                        self.outputs.append(lhs)
                    return make_nop()
                if isinstance(rhs, halide_imm_types) and ast.Store not in rw:
                    self.var_consts[lhs] = rhs
                else:
                    ph = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                    self.alloc_buffers[lhs] = (ph, 'global')
            if lhs in self.var_consts.keys():
                return make_nop()
            _internal_assert(lhs in self.alloc_buffers.keys(), \
                             "This variable should be defined before!")
            tgt, _ = self.alloc_buffers[lhs]
            return _make.Provide(tgt.op, 0, rhs, [_api.const(0, dtype=rhs.dtype)])
        else:
            lhs = self.visit(lhs)
            _internal_assert(isinstance(lhs, _expr.Call), \
                             "An array access's LHS is expected to be a expr.Call!")
            #TODO: support slice later
            buf = self._get_buffer_from_id(lhs.name, for_provide=True)
            return _make.Provide(buf.op, 0, rhs, lhs.args)


    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return [self.visit(i) for i in node.value.elts]
        return [self.visit(node.value)]


    def visit_Subscript(self, node):
        args = self.visit(node.slice)
        if isinstance(node.value, ast.Name):
            array = node.value.id
            _buf = self._get_buffer_from_id(array)
            return _make.Call(_buf.dtype, array, args, _expr.Call.Halide, _buf.op, 0)

        _internal_assert(isinstance(node.value, ast.Attribute), \
                         "Only variable and attribute's subscript supported so far")
        _internal_assert(isinstance(node.value.value, ast.Name), \
                         "The root of array access is expect to be a id!")
        _internal_assert(node.value.attr == "shape", \
                         "Attribute access so far only 'shape' is supported!")
        _internal_assert(len(args) == 1, "For 'shape' access the argument should be only one!")
        args = args[0]
        #TODO: maybe support non-constant value later?
        _internal_assert(isinstance(args, (_expr.IntImm, _expr.UIntImm)), \
                         "So far only constant shape access supported!")
        buf = self._get_buffer_from_id(node.value.value.id)
        return buf.shape[args.value]


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
            else_body = make_nop()
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
        elif n == 2:
            _internal_assert(isinstance(node.op, (ast.And, ast.Or)), \
                             "Binary is supposed to be and/or!")
            values = [self.visit(i) for i in node.values]
            return HybridParser._binop_maker[type(node.op)](*values)
        else:
            raise ValueError("This Bool Op is not supported yet!")


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
        n = len(node.args)
        if func_id in LOOP_INTRIN.keys() and func_id != 'bind':
            if n == 1:
                low, ext = _api.const(0, dtype='int32'), self.visit(node.args[0])
            else:
                _internal_assert(n == 2, "A loop intrinsic should only have 1 or 2 arguments!")
                low, ext = self.visit(node.args[0]), self.visit(node.args[1])
            if not _ir_pass.Equal(low, _api.const(0, dtype='int32')):
                ext = ext - low
            for_type = LOOP_INTRIN[func_id]
            iter_var = None
            return iter_var, low, ext, for_type
        elif func_id == 'bind':
            _internal_assert(n == 2, "A loop bind should only have 2 arguments!")
            _internal_assert(isinstance(node.args[0], ast.Str), \
                             "A loop bind's first argument should be a string!")
            _vn = node.args[0].s
            iter_var = thread_axis(node.args[0].s)
            low, ext = _api.const(0, dtype='int32'), self.visit(node.args[1])
            for_type = None
            return iter_var, low, ext, for_type
        elif func_id in MATH_INTRIN:
            return getattr(intrin, func_id)(*[self.visit(arg) for arg in node.args])
        elif func_id in ['allocate', 'output_tensor']:
            _internal_assert(isinstance(node.args[0], ast.Tuple), \
                             "allocate's first argument should be a tuple of shape!")
            shape = tuple(self.visit(i) for i in node.args[0].elts)
            if func_id == 'output_tensor':
                _internal_assert(not self.loops_above, \
                                 "Are you sure to allocate a output buffer multiple times?")
            for i in shape:
                _internal_assert(isinstance(i, _expr.Expr), "The shape should be an expression")
            if n > 1:
                if isinstance(node.args[1], ast.Str):
                    dtype = node.args[1].s
                else:
                    _internal_assert(isinstance(node.args[1], ast.Attribute), \
                                     "Unable to evaluate to get data type")
                    to_eval = node.args[1]
                    _internal_assert(isinstance(to_eval.value, ast.Name), \
                                     "Unable to evaluate the attribute to get data type")
                    _internal_assert(to_eval.attr == 'dtype', \
                                     "Only dtype attribute is supported so far")
                    dtype = self._get_buffer_from_id(to_eval.value.id).dtype
            else:
                dtype = 'float32'
            if n > 2:
                _internal_assert(isinstance(node.args[2], ast.Str), \
                                 "The data scope should be an string")
                _internal_assert(func_id != 'output_tensor', "Output tensor cannot specify scope")
                scope = node.args[2].s
            else:
                scope = 'global' if func_id != 'output_tensor' else 'output'
            return (shape, dtype, scope)
        elif func_id == 'max' or func_id == 'min':
            _internal_assert(n == 2, "Max/Min function should have 2 elements")
            a, b = self.visit(node.args[0]), self.visit(node.args[1])
            return getattr(_make, func_id.title())(a, b)
        else:
            raise ValueError("Function call not supported yet!")


    def visit_For(self, node):
        iter_var, low, ext, for_type = self.visit(node.iter)
        _internal_assert(isinstance(node.target, ast.Name), \
                         "The loop iterator should be a variable!")
        _name = node.target.id
        if iter_var is None:
            _internal_assert(for_type is not None, "The loop bind function parse error!")
            offset = iter_var = _api.var(_name)
            if not _ir_pass.Equal(low, _api.const(0, dtype='int32')):
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
            res = _make.For(iter_var, _api.const(0, dtype='int32'), ext, for_type, 0, _body)
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
        if len(ids) != len(self.outputs):
            logging.log(logging.CRITICAL, '[Warning] Not all the output buffers returned!')
        self.outputs = [self._args[i] for i in ids]
        self.returned = True
        return make_nop()


def parse_python(src, args):
    """The helper function of calling the AST visitor

    Parameters
    ----------
    src : str
        The source code of the function to be parsed.

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
    var_usage = determine_variable_usage(root, args)
    parser = HybridParser(args, var_usage)
    parser.parsed_body = parser.visit(root)
    _internal_assert(parser.returned, 'No valid return found in the function body!')
    return parser
