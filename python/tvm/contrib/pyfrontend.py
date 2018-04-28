import ast, inspect, types, operator, numpy

from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass  as _ir_pass
from .. import build_module as builder

_noop = _make.Evaluate(_api.const(0))

def _list_to_block(lst):
    lst = list(filter(lambda stmt: not _ir_pass.Equal(stmt, _noop), lst))
    if not lst:
        return _noop
    if len(lst) == 1:
        return lst[0]
    body = lst[0]
    for i in lst[1:]:
        body = _make.Block(body, i)
    return body


# The AST visitor that detects the status of each variable, which is either
# read only, or read and write.
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
        ast.Add   : _make.Add,
        ast.Sub   : _make.Sub,
        ast.Mult  : _make.Mul,
        ast.Div   : _make.Div,
        ast.Mod   : _make.Mod,
        ast.BitOr : operator.or_,
        ast.BitAnd: operator.and_,
        ast.BitXor: operator.xor
    }

    def __init__(self, args, usage):
        self.input_param = {}
        self.rw_status   = usage
        self.vars_buffer = {}
        self.vars_const  = {}
        self.args        = args
        self.loop_levels = {}
        self.func_name   = ""

    def visit_Module(self, node):
        assert len(node.body) == 1
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        assert len(node.args.args) == len(self.args)
        for idx, arg in enumerate(node.args.args):
            self.input_param[arg.arg] = self.args[idx]
        res = [self.visit(i) for i in node.body]
        res = _list_to_block(res)
        one = _make.range_by_min_extent(0, 1)
        for k, v in self.vars_buffer.items():
            res = _make.Realize(v.op, 0, v.dtype, [one], _api.convert(True), res)
        self.func_name = node.name
        return res

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        #print(node.id)
        if node.id in self.input_param.keys() and isinstance(self.input_param[node.id], _expr.Var):
            return self.input_param[node.id]
        elif node.id in self.loop_levels.keys():
            return self.loop_levels[node.id]
        else:
            assert node.id not in self.input_param.keys()
            assert node.id in self.rw_status.keys()
            assert node.id in self.vars_buffer.keys() or node.id in self.vars_const.keys()
            if node.id in self.vars_buffer.keys():
                _buf = self.vars_buffer[node.id]
                return _make.Call(_buf.dtype, node.id, [_api.const(0)],
                    _expr.Call.Halide, _buf.op, 0)
            else:
                return self.vars_const[node.id]

    def visit_Num(self, node):
        return _api.const(node.n)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        rhs = self.visit(node.value)
        rhs = _ir_pass.Simplify(rhs)
        if isinstance(node.targets[0], ast.Name):
            lhs = node.targets[0].id
            #print(ast.Store, self.rw_status[lhs])
            if lhs not in self.rw_status.keys():
                print('Warning: Variable %s is not used after declaration! Discard stmt!' % lhs)
                return _noop
            elif ast.Store in self.rw_status[lhs]:
                #print('read write %s' % lhs)
                if lhs not in self.vars_buffer.keys():
                    self.vars_buffer[lhs] = _api.placeholder((1, ), dtype = rhs.dtype, name = lhs)
                return _make.Provide(self.vars_buffer[lhs].op, 0, rhs, [_api.const(0)])
            else:
                if isinstance(rhs, _expr.FloatImm) or isinstance(rhs, _expr.IntImm):
                    #print('read only const %s' % lhs)
                    self.vars_const[lhs] = rhs
                    return _noop
                else:
                    #print('read only non-const %s' % lhs)
                    self.vars_buffer[lhs] = _api.placeholder((1, ), dtype = rhs.dtype, name = lhs)
                    return _make.Provide(self.vars_buffer[lhs].op, 0, rhs, [_api.const(0)])
        else:
            lhs = self.visit(node.targets[0])
            assert isinstance(lhs, _expr.Call)
            #TODO: support tuple later(?)
            #TODO: support defined intermediate buffer later
            assert lhs.name in self.input_param.keys()
            return _make.Provide(self.input_param[lhs.name].op, 0, rhs, lhs.args)

    def visit_Index(self, node):
        #print(ast.dump(node))
        if isinstance(node.value, ast.Tuple):
            return [self.visit(i) for i in node.value.elts]
        else:
            return [self.visit(node.value)]

    def visit_Subscript(self, node):
        #print(node.value)
        #assert isinstance(node.value, ast.Name) or isinstance(node.value, ast.Attribute)
        args = self.visit(node.slice)
        #print(args)
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

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        #print(PyAST2HalideIR._binop_maker[type(node.op)](lhs, rhs))
        return PyAST2HalideIR._binop_maker[type(node.op)](lhs, rhs)

    def visit_For(self, node):
        assert isinstance(node.target, ast.Name)
        var = node.target.id
        self.loop_levels[var] = _api.var(var)
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id == 'range'
        if len(node.iter.args) == 1:
            low, high = _api.const(0), self.visit(node.iter.args[0])
        elif len(node.iter.args) == 2:
            low, high = self.visit(node.iter.args[0]), self.visit(node.iter.args[1])
        ext  = high if isinstance(low, _expr.IntImm) and low.value == 0 else high - low
        _body = [self.visit(stmt) for stmt in node.body]
        _body = _list_to_block(_body)
        res = _make.For(self.loop_levels[var], low, ext, _stmt.For.Serial, 0, _body)
        self.loop_levels.pop(var)
        return res

def parse(func, args, dump = False):
    """Parse a subset of Python to HalideIR

    Parameters
    ----------
    func : str or types.FunctionType
        If it is a string, parse the source code.
        If it is a function, parse the function.

    args : list of Buffer or Tensor or Var
        The argument lists to the function.
        Leave it None if no buffer is related to the function to be parsed

    dump : bool, optional
        If it is true, Python ast will be dumped.
        A debug parameter.

    Returns
    -------
    ir : Stmt
        The result Halide IR
    """
    if isinstance(func, str):
        ir = ast.parse(func)
    else:
        assert isinstance(func, types.FunctionType)
        ir = ast.parse(inspect.getsource(func))
    var_usage = _find_all_variable_decl(ir)
    #print(var_usage)
    return PyAST2HalideIR(args, var_usage).visit(ir)

def lower(func, args, binds = None):
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
    if isinstance(func, str):
        ir = ast.parse(func)
    else:
        assert isinstance(func, types.FunctionType)
        ir = ast.parse(inspect.getsource(func))
    if binds is None:
        binds = {}
    var_usage = _find_all_variable_decl(ir)
    parser = PyAST2HalideIR(args, var_usage)
    stmt = parser.visit(ir)
    binds_args, args = builder.get_binds(parser.args)
    binds_vars, _    = builder.get_binds(parser.vars_buffer)
    _binds = binds_args.copy()
    _binds.update(binds_vars)
    _binds.update(binds_vars)
    _binds.update(binds)
    #print(_binds)
    stmt = _ir_pass.StorageFlatten(stmt, _binds, 64)
    #print(stmt)
    return _ir_pass.MakeAPI(stmt, parser.func_name, args, 0,
                    builder.current_build_config().restricted_func)

def allocate(shape, dtype = None):
    return numpy.zeros(shape).tolist()

