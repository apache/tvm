import ast, inspect, types, operator, numpy

from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api
from .. import ir_pass  as _ir_pass

def _list_to_block(lst):
    if len(lst) == 1:
        return lst[0]
    else:
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

    def __init__(self, inputs, usage):
        self.input_param = {}
        self.rw_status   = usage
        self.vars_buffer = {}
        self.vars_const  = {}
        self.inputs      = inputs
        self.loop_levels = []

    def visit_Module(self, node):
        res = [self.visit(i) for i in node.body]
        return _list_to_block(res)

    def visit_FunctionDef(self, node):
        assert len(node.args.args) == len(self.inputs)
        for idx, arg in enumerate(node.args.args):
            self.input_param[arg.arg] = self.inputs[idx]
        res = [self.visit(i) for i in node.body]
        res = _list_to_block(res)
        one = _make.range_by_min_extent(0, 1)
        for k, v in self.vars_buffer.items():
            res = _make.Realize(v.op, 0, v.dtype, [one], _api.convert(True), res)
        return res

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        #print(node.id)
        if node.id in self.input_param.keys() and isinstance(self.input_param[node.id], _expr.Var):
            return self.input_param[node.id]
        elif node.id in self.loop_levels:
            return _api.var(node.id)
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
                return _make.Evaluate(_api.const(0))
            elif ast.Store in self.rw_status[lhs]:
                #print('read write %s' % lhs)
                if lhs not in self.vars_buffer.keys():
                    self.vars_buffer[lhs] = _api.placeholder((1, ), dtype = rhs.dtype, name = lhs)
                return _make.Provide(self.vars_buffer[lhs].op, 0, rhs, [_api.const(0)])
            else:
                if isinstance(rhs, _expr.FloatImm) or isinstance(rhs, _expr.IntImm):
                    #print('read only const %s' % lhs)
                    self.vars_const[lhs] = rhs
                    return _make.Evaluate(_api.const(0))
                else:
                    self.vars_buffer[lhs] = _api.placeholder((1, ), dtype = rhs.dtype, name = lhs)
                    #print('read only non-const %s' % lhs)
                    return _make.Provide(self.vars_lut[lhs].op, 0, rhs, [_api.const(0)])
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
            #TODO: detect the type
            assert array in self.input_param.keys()
            _buf = self.input_param[array]
            return _make.Call(_buf.dtype, array, args, _expr.Call.Halide, _buf, 0)
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
        self.loop_levels.append(var)
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id == 'range'
        if len(node.iter.args) == 1:
            low, high = _api.const(0), self.visit(node.iter.args[0])
        elif len(node.iter.args) == 2:
            low, high = self.visit(node.iter.args[0]), self.visit(node.iter.args[1])
        ext  = high if isinstance(low, _expr.IntImm) and low.value == 0 else high - low
        _body = [self.visit(stmt) for stmt in node.body]
        _body = _list_to_block(_body)
        self.loop_levels.pop()
        return _make.For(_api.var(var), low, ext, _stmt.For.Serial, 0, _body)


def parse(func, **kwargs):
    """Parse a subset of Python to HalideIR
    
    Parameters
    ----------
    name : str or types.FunctionType
        If it is a string, open the corresponding file and parse it.
        If it is a function, parse the function.

    Returns
    -------
    ir : Module
        The result Halide IR
    """
    if isinstance(func, str):
        ir = ast.parse(func)
    elif isinstance(func, types.FunctionType):
        ir = ast.parse(inspect.getsource(func))
    else:
        assert False
    # TODO: To be finished...
    if kwargs.get('dump') is not None and kwargs.get('dump'):
        print(ast.dump(ir))
    if kwargs.get('inputs') is not None:
        assert isinstance(kwargs['inputs'], list)
        inputs = kwargs['inputs']
    else:
        inputs = []
    var_usage = _find_all_variable_decl(ir)
    #print(var_usage)
    return PyAST2HalideIR(inputs, var_usage).visit(ir)

def allocate(shape, dtype = None):
    return numpy.zeros(shape).tolist()

