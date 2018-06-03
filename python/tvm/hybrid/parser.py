"""Compiling a subset of Python to HalideIR"""
#pylint: disable=no-else-return
import ast
import operator
import sys
from ._internal import NOP, TRUE, RANGE_ONE, HALIDE_IMM, ZERO
from ._intrin import LOOP_INTRIN
from .var_decl import determine_variable_usage
from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass as _ir_pass

def list_to_block(visit, lst):
    """Convert a list of Python IR nodes to HalideIR Block"""
    lst = list(map(visit, lst))
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, NOP)]
    if not lst:
        return NOP
    if len(lst) == 1:
        return lst[0]
    body = lst[0]
    for i in lst[1:]:
        body = _make.Block(body, i)
    return body

class PyAST2HalideIR(ast.NodeVisitor):
    """Python AST visitor pass which finally lowers it to HalideIR"""

    _binop_maker = {
        ast.Add   : operator.add,
        ast.Sub   : operator.sub,
        ast.Mult  : operator.mul,
        ast.Div   : _make.Div,
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
    }

    _unaryop_maker = {
        ast.USub   : operator.neg,
        ast.Invert : operator.invert,
        ast.Not    : operator.not_
    }

    def __init__(self, args, usage, func_name=None):
        """
        args: A list of tvm.placeholder or tvm.var
        ----------------------------------------
        Provided by the user, the argument list of the function to be lowered.

        usage: A dict of variables used in last in this function
        --------------------------------------------------------
        Provided by last lower pass, which collects this information

        func_name: str
        --------------
        The name of the function to be lowered; if not provided,
        the compiler will use the name in the AST
        """
        self.args = args[:]
        self.usage = usage.copy()
        self._args = {} # Dict maps arg name to actual arg instance (either a var or a buffer)
        self.buffers = {}
        self.loops_above = {} # State variable that indicates loop levels above the current node
        self.var_consts = {} # Variables that are determined as readonly in previous stage
        self.func_name = func_name # The name of the function to be lowered
        self.iter_axis = []

    #pylint: disable=missing-docstring, invalid-name
    #pylint: consider-merging-isinstance, no-else-return
    #pylint: disable=inconsistent-return-statements

    def wrap_up_realize(self, node, body):
        """Wrap up all the variables which will no longer be used"""
        for key, val in self.usage.items():
            if key in self.var_consts.keys():
                continue
            _, scope, _ = val
            if scope == node:
                _buf = self.buffers[key]
                body = _make.Realize(_buf.op, 0, _buf.dtype, [RANGE_ONE], TRUE, body)
        return body

    def visit_Module(self, node):
        assert len(node.body) == 1
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        assert len(node.args.args) == len(self.args)
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
        if _id in self._args.keys() and isinstance(self._args[_id], _expr.Var):
            return self._args[_id]
        elif _id in self.loops_above.keys():
            return self.loops_above[_id]
        # This id cannot be a buffer; buffer will be handled in subscript
        assert _id not in self._args.keys()
        assert _id in self.usage.keys()
        # Buffer
        if _id in self.buffers.keys():
            _buf = self.buffers[_id]
            return _make.Call(_buf.dtype, _id, [_api.const(0)], _expr.Call.Halide, _buf.op, 0)
        # Compilation time constant
        assert _id in self.var_consts.keys()
        return self.var_consts[_id]

    def visit_Num(self, node):
        return _api.const(node.n)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        lhs = node.targets[0]
        rhs = _ir_pass.Simplify(self.visit(node.value))
        if isinstance(lhs, ast.Name):
            #TODO: support defined intermediate buffer later
            lhs_ = lhs
            lhs = lhs.id
            assert lhs not in self.loops_above.keys()
            decl, _, rw = self.usage[lhs]
            if decl == lhs_:
                assert lhs not in self.var_consts.keys()
                assert lhs not in self.buffers.keys()
                if isinstance(rhs, HALIDE_IMM) and ast.Store not in rw:
                    self.var_consts[lhs] = rhs
                else:
                    self.buffers[lhs] = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
            if lhs in self.var_consts.keys():
                return NOP
            else:
                assert lhs in self.buffers.keys()
                return _make.Provide(self.buffers[lhs].op, 0, rhs, [ZERO])
        else:
            lhs = self.visit(lhs)
            assert isinstance(lhs, _expr.Call)
            #TODO: support slice later
            assert lhs.name in self._args.keys()
            return _make.Provide(self._args[lhs.name].op, 0, rhs, lhs.args)

    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return [self.visit(i) for i in node.value.elts]
        return [self.visit(node.value)]

    def visit_Subscript(self, node):
        #assert isinstance(node.value, ast.Name) or isinstance(node.value, ast.Attribute)
        args = self.visit(node.slice)
        if isinstance(node.value, ast.Name):
            array = node.value.id
            assert array in self._args.keys()
            _buf = self._args[array]
            return _make.Call(_buf.dtype, array, args, _expr.Call.Halide, _buf.op, 0)
        elif isinstance(node.value, ast.Attribute):
            assert isinstance(node.value.value, ast.Name)
            assert node.value.attr == "shape"
            assert len(args) == 1
            args = args[0]
            #TODO: maybe support non-constant value later?
            assert isinstance(args, (_expr.IntImm, _expr.UIntImm))
            assert node.value.value.id in self._args.keys()
            return self._args[node.value.value.id].shape[args.value]
        else:
            assert False

    def visit_With(self, node):
        if sys.version_info[0] < 3:
            context = node.context_expr
            option = node.optional_vars
        else:
            assert len(node.items) == 1
            context = node.items[0].context_expr
            option = node.items[0].optional_vars
        assert isinstance(context, ast.Call)
        assert isinstance(option, ast.Name)
        self.annotation[option.id] = context.func.id
        return list_to_block(self.visit, node.body)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if_body = list_to_block(self.visit, node.body)
        if node.orelse:
            else_body = list_to_block(self.visit, node.orelse)
        else:
            else_body = NOP
        return _make.IfThenElse(cond, if_body, else_body)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if_body = self.visit(node.body)
        else_body = self.visit(node.orelse)
        return _make.Select(cond, if_body, else_body)

    def visit_Compare(self, node):
        lhs = self.visit(node.left)
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        rhs = self.visit(node.comparators[0])
        return PyAST2HalideIR._binop_maker[type(node.ops[0])](lhs, rhs)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        return PyAST2HalideIR._unaryop_maker[type(node.op)](operand)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return PyAST2HalideIR._binop_maker[type(node.op)](lhs, rhs)

    def visit_For(self, node):
        assert isinstance(node.target, ast.Name)
        iter_var = _api.var(node.target.id)
        self.loops_above[iter_var.name] = iter_var
        self.iter_axis.append(iter_var.name)
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id in LOOP_INTRIN
        _for_type = node.iter.func.id
        _body = list_to_block(self.visit, node.body)
        _body = self.wrap_up_realize(node, _body)
        if _for_type != 'bind':
            if len(node.iter.args) == 1:
                low, ext = ZERO, self.visit(node.iter.args[0])
            else:
                assert len(node.iter.args) == 2
                low, ext = self.visit(node.iter.args[0]), self.visit(node.iter.args[1])
            _for_type = getattr(_stmt.For, node.iter.func.id.title())
            assert len(node.iter.args) == 1 or len(node.iter.args) == 2
            res = _make.For(iter_var, low, ext, _for_type, 0, _body)
        else:
            assert len(node.iter.args) == 2
            low, ext = ZERO, self.visit(node.iter.args[0])
            pass
        self.loops_above.pop(iter_var.name)
        return res

def parse_python(src, args):
    """ The helper function of calling the AST visitor"""
    root = ast.parse(src)
    var_usage = determine_variable_usage(root, args)
    parser = PyAST2HalideIR(args, var_usage)
    halide_ir = parser.visit(root)
    return (halide_ir, parser)
