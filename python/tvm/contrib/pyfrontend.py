"""Compiling a subset of Python to HalideIR"""
import ast
import inspect
import types
import operator
import sys
#import numpy
from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass  as _ir_pass
from .. import build_module as builder

#pylint: disable=missing-docstring, invalid-name, consider-merging-isinstance, no-else-return
#pylint: disable=inconsistent-return-statements, eval-used

NOP = _make.Evaluate(_api.const(0))

class LoopAnnotation(object):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        return self

#Loop Annotation Classes
class Serial(LoopAnnotation): pass     #pylint: disable=multiple-statements
class Unrolled(LoopAnnotation): pass   #pylint: disable=multiple-statements
class Vectorized(LoopAnnotation): pass #pylint: disable=multiple-statements
class Parallel(LoopAnnotation): pass   #pylint: disable=multiple-statements


ANNOTATION_GLOBALS = {
    'Serial'    : Serial,
    'Unrolled'  : Unrolled,
    'Vectorized': Vectorized,
    'Parallel'  : Parallel
}

#TODO: uncomment these lines later to support intermediate buffer
#def allocate(shape, dtype=None):
#    return numpy.zeros(shape).tolist()

def py_frontend(func):
    func.__globals__.update(ANNOTATION_GLOBALS)
    return func

def _list_to_block(lst):
    lst = [stmt for stmt in lst if not _ir_pass.Equal(stmt, NOP)]
    if not lst:
        return NOP
    if len(lst) == 1:
        return lst[0]
    body = lst[0]
    for i in lst[1:]:
        body = _make.Block(body, i)
    return body

class PyVariableUsage(ast.NodeVisitor):

    def __init__(self):
        self.declared_vars = {}
        self.loops_above = []
        self.in_loop_decl = False

    def visit_For(self, node):
        self.in_loop_decl = True
        self.visit(node.target)
        self.in_loop_decl = False

        for i in node.body:
            self.visit(i)

        self.loops_above.pop()

    def visit_Name(self, node):
        if self.in_loop_decl:
            self.loops_above.append(node.id)
            return

        # The loop variable cannot be overwritten when iteration
        if isinstance(node.ctx, ast.Store):
            assert node.id not in self.loops_above

        if node.id not in self.declared_vars:
            if isinstance(node.ctx, ast.Store):
                self.declared_vars[node.id] = set()
        else:
            self.declared_vars[node.id].add(type(node.ctx))

def _find_all_variable_decl(ir):
    visitor = PyVariableUsage()
    visitor.visit(ir)
    return visitor.declared_vars

class PyAST2HalideIR(ast.NodeVisitor):

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

    def __init__(self, args, usage):
        self.input_param = {}
        self.vars_buffer = {}
        self.loop_levels = {}
        self.vars_const = {}
        self.rw_status = usage
        self.func_name = ""
        self.args = args
        self.annotation = {}
        self.iter_axis = []

    def visit_Module(self, node):
        assert len(node.body) == 1
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        assert len(node.args.args) == len(self.args)
        for idx, arg in enumerate(node.args.args): #pylint: disable=unused-variable
            # I do not know how to fix it. The only thing I can do is to detect the runtime version.
            _attr = 'id' if sys.version_info[0] < 3 else 'arg'
            self.input_param[eval('arg.%s' % _attr)] = self.args[idx]
        res = [self.visit(i) for i in node.body]
        res = _list_to_block(res)
        one = _make.range_by_min_extent(0, 1)
        for _, v in self.vars_buffer.items():
            res = _make.Realize(v.op, 0, v.dtype, [one], _api.convert(True), res)
        self.func_name = node.name
        return res

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        _id = node.id
        if _id in self.input_param.keys() and isinstance(self.input_param[_id], _expr.Var):
            return self.input_param[_id]
        elif _id in self.loop_levels.keys():
            return self.loop_levels[_id]
        assert _id not in self.input_param.keys()
        assert _id in self.rw_status.keys()
        assert _id in self.vars_buffer.keys() or _id in self.vars_const.keys()
        if _id in self.vars_buffer.keys():
            _buf = self.vars_buffer[_id]
            return _make.Call(_buf.dtype, _id, [_api.const(0)], _expr.Call.Halide, _buf.op, 0)
        else:
            return self.vars_const[_id]

    def visit_Num(self, node):
        return _api.const(node.n)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        rhs = self.visit(node.value)
        rhs = _ir_pass.Simplify(rhs)
        if isinstance(node.targets[0], ast.Name):
            lhs = node.targets[0].id
            if lhs not in self.rw_status.keys():
                # This means variable %s is not used after declaration!
                # Thus, we discard this variable.
                return NOP
            elif ast.Store in self.rw_status[lhs]:
                if lhs not in self.vars_buffer.keys():
                    self.vars_buffer[lhs] = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                return _make.Provide(self.vars_buffer[lhs].op, 0, rhs, [_api.const(0)])
            elif isinstance(rhs, _expr.FloatImm) or isinstance(rhs, _expr.IntImm):
                self.vars_const[lhs] = rhs
                return NOP
            else:
                self.vars_buffer[lhs] = _api.placeholder((1, ), dtype=rhs.dtype, name=lhs)
                return _make.Provide(self.vars_buffer[lhs].op, 0, rhs, [_api.const(0)])
        else:
            lhs = self.visit(node.targets[0])
            assert isinstance(lhs, _expr.Call)
            #TODO: support tuple later(?)
            #TODO: support defined intermediate buffer later
            assert lhs.name in self.input_param.keys()
            return _make.Provide(self.input_param[lhs.name].op, 0, rhs, lhs.args)

    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return [self.visit(i) for i in node.value.elts]
        return [self.visit(node.value)]

    def visit_Subscript(self, node):
        #assert isinstance(node.value, ast.Name) or isinstance(node.value, ast.Attribute)
        args = self.visit(node.slice)
        if isinstance(node.value, ast.Name):
            array = node.value.id
            assert array in self.input_param.keys()
            _buf = self.input_param[array]
            return _make.Call(_buf.dtype, array, args, _expr.Call.Halide, _buf.op, 0)
        elif isinstance(node.value, ast.Attribute):
            assert isinstance(node.value.value, ast.Name)
            assert node.value.attr == "shape"
            assert len(args) == 1
            args = args[0]
            #TODO: maybe support non-constant value later?
            assert isinstance(args, _expr.IntImm) or isinstance(args, _expr.UIntImm)
            assert node.value.value.id in self.input_param.keys()
            return self.input_param[node.value.value.id].shape[args.value]
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
        return _list_to_block([self.visit(i) for i in node.body])

    def visit_If(self, node):
        cond = self.visit(node.test)
        if_body = _list_to_block([self.visit(i) for i in node.body])
        if node.orelse:
            else_body = _list_to_block([self.visit(i) for i in node.orelse])
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
        var = node.target.id
        self.loop_levels[var] = _api.var(var)
        self.iter_axis.append(var)
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id == 'range'
        assert len(node.iter.args) == 1 or len(node.iter.args) == 2
        if len(node.iter.args) == 1:
            low, high = _api.const(0), self.visit(node.iter.args[0])
        elif len(node.iter.args) == 2:
            low, high = self.visit(node.iter.args[0]), self.visit(node.iter.args[1])
        ext = high if isinstance(low, _expr.IntImm) and low.value == 0 else high - low
        _body = [self.visit(stmt) for stmt in node.body]
        _body = _list_to_block(_body)
        _for_type = _stmt.For.Serial
        if var in self.annotation.keys():
            _for_type = eval('_stmt.For.%s' % self.annotation[var])
        res = _make.For(self.loop_levels[var], low, ext, _for_type, 0, _body)
        self.loop_levels.pop(var)
        return res

def parse(func, args):
    """Parse a subset of Python to HalideIR

    Parameters
    ----------
    func : str or types.FunctionType
        If it is a string, parse the source code.
        If it is a function, parse the function.

    args : list of Buffer or Tensor or Var
        The argument lists to the function.
        Leave it None if no buffer is related to the function to be parsed

    Returns
    -------
    (halide_ir, parser) : (Stmt, PyAST2HalideIR)
        The result Halide IR and the parser class instance.
        TODO: The parser class isinstance will later provide some interface for hybrid
              programming model.
    """
    if isinstance(func, str):
        ir = ast.parse(func)
    else:
        assert isinstance(func, types.FunctionType)
        ir = ast.parse(inspect.getsource(func))
    var_usage = _find_all_variable_decl(ir)
    parser = PyAST2HalideIR(args, var_usage)
    halide_ir = parser.visit(ir)
    return (halide_ir, parser)


def lower(func, args, binds=None, simple_mode=False):
    """Lower a subset of Python to LoweredFunc

    Parameters
    ----------
    func : str or types.FunctionType
        If it is a string, parse the source code.
        If it is a function, parse the function.

    args : list of Buffer or Tensor or Var
        The argument lists to the function.
        Leave it None if no buffer is related to the function to be parsed

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    Returns
    -------
    ir : LoweredFunc
        The result lowered function
    """
    stmt, parser = parse(func, args)
    if simple_mode:
        return stmt
    binds_args, args = builder.get_binds(parser.args)
    binds_vars, _ = builder.get_binds(parser.vars_buffer)
    _binds = binds_args.copy()
    _binds.update(binds_vars)
    _binds.update(binds if binds is not None else {})
    stmt = _ir_pass.StorageFlatten(stmt, _binds, 64)
    _config = builder.current_build_config().restricted_func
    return _ir_pass.MakeAPI(stmt, parser.func_name, args, 0, _config)
