import ast, inspect, types, operator, numpy

from .. import expr as _expr
from .. import stmt as _stmt
from .. import make as _make
from .. import api  as _api

class PyAST2HalideIR(ast.NodeVisitor):

    _binop_maker = {
        ast.Add   :_make.Add,
        ast.Sub   :_make.Sub,
        ast.Mult  :_make.Mul,
        ast.Div   :_make.Div,
        ast.Mod   :_make.Mod,
        ast.BitOr :operator.or_,
        ast.BitAnd:operator.and_,
        ast.BitXor:operator.xor
    }

    def __init__(self, inputs):
        self.converted     = []
        self.scope         = {}
        self.variables     = set([])
        self.inputs        = inputs

    def visit_Module(self, node):
        for i in node.body:
            halide_stmt = self.visit(i)
            self.converted.append(halide_stmt)
        return self.converted

    def visit_FunctionDef(self, node):
        assert len(node.args.args) == len(self.inputs)
        for idx, arg in enumerate(node.args.args):
            self.scope[arg.arg] = self.inputs[idx]
        res = [self.visit(i) for i in node.body]
        return res

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        #print(node.id)
        if node.id not in self.scope.keys():
            return _api.var(node.id)
        elif isinstance(self.scope[node.id], _expr.Var):
            return self.scope[node.id]
        else:
            _buf = self.scope[node.id]
            return _make.Call(_buf.dtype, _buf.name, [_api.const(0)], _expr.Call.PureExtern, None, 0)

    def visit_Num(self, node):
        return _api.const(node.n)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)
        if isinstance(lhs, _expr.Var):
            #TODO: Check [1] or LetStmt var
            _buf = self.scope[lhs.name] = _api.placeholder((1,), rhs.dtype, name = lhs.name)
            _body = _make.Provide(_buf.op, 0, rhs, [_api.const(0)])
            _rng = [_make.range_by_min_extent(0, 1)]
            return _make.Realize(_buf.op, 0, _buf.dtype, _rng, _api.const(0) == _api.const(0), _body)
        elif isinstance(lhs, _expr.Call):
            # Provide Stmt
            #TODO: support tuple later(?)
            assert lhs.name in self.scope.keys()
            return _make.Provide(self.scope[lhs.name].op, 0, rhs, lhs.args)
        else:
            assert False

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
            assert array in self.scope.keys()
            return _make.Call(self.scope[array].dtype, array, args, _expr.Call.PureExtern, None, 0)
        elif isinstance(node.value, ast.Attribute):
            assert isinstance(node.value.value, ast.Name)
            assert node.value.attr == "shape"
            assert len(args) == 1
            args = args[0]
            #TODO: maybe support non-constant value later?
            assert isinstance(args, _expr.IntImm) or isinstance(args, _expr.UIntImm)
            assert node.value.value.id in self.scope.keys()
            return self.scope[node.value.value.id].shape[args.value]
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
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id == 'range'
        if len(node.iter.args) == 1:
            low, high = _api.const(0), self.visit(node.iter.args[0])
        elif len(node.iter.args) == 2:
            low, high = self.visit(node.iter.args[0]), self.visit(node.iter.args[1])
        ext  = high - low
        _body = [self.visit(stmt) for stmt in node.body]
        if len(_body) == 1:
            return _make.For(_api.var(var), low, ext, _stmt.For.Serial, 0, _body[0])
        else:
            body = _body[0]
            for i in _body[1:]:
                body = _make.Block(body, i)
            return _make.For(_api.var(var), low, ext, _stmt.For.Serial, 0, body)
        #body = _make.Block(body)

            

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
    return PyAST2HalideIR(inputs).visit(ir)

def allocate(shape, dtype = None):
    return numpy.zeros(shape).tolist()

