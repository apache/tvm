
class LetList:
    VAR_COUNTER = 0

    def __init__(self):
        self.bindings = []

    def push(self, expr, *, ty=None, bind_var=None):
        if bind_var is None:
            bind_var = relay.Var(f'fresh{LetList.VAR_COUNTER}', ty)
            LetList.VAR_COUNTER += 1
        self.bindings.append((bind_var, expr))
        return bind_var

    def get(self, expr):
        ret = expr
        for (var, rhs) in reversed(self.bindings):
            ret = relay.Let(var, rhs, ret)
        return ret

    def with_ll(func):
        ll = LetList()
        return ll.get(func(ll))


def get_shape(expr):
    return [int(x) for x in expr.checked_type.shape]


class Config:
    def __init__(self, n=1, i=16):
        self.n = n
        self.i = i
        self.o = i


class Layout:
    def key(self):
        raise NotImplementedError(type(self))

    def keyname(self):
        return (self.key(), type(self).__name__)

    def __eq__(self, other):
        return self.keyname() == other.keyname()

    def __hash__(self):
        return hash(self.keyname())

    def to_layout(self, expr, config, original_shape):
        raise NotImplementedError(type(self))

    def from_layout(self, expr, config, original_shape):
        raise NotImplementedError(type(self))


def full_div(l, r):
    assert l % r == 0
    return l // r


class NIniLayout(Layout):
    def key(self):
        return ()

    def to_layout(self, expr, config, original_shape):
        # NI
        assert len(original_shape) == 2
        N, I = original_shape
        n, i = config.n, config.i
        # NnIi
        expr = relay.op.reshape(expr,
                                newshape=(full_div(N, n), n,
                                          full_div(I, i), i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

    def from_layout(self, expr, config, original_shape):
        assert len(original_shape) == 2
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        expr = relay.op.reshape(expr, newshape=original_shape)
        return expr

class OIoiLayout(Layout):
    def __init__(self, I):
        self.I = I

    def key(self):
        return self.I

    def to_layout(self, expr, config, original_shape):
        # probably packing incorrectly. but operation cost is the same so no worry.
        # NI
        assert len(original_shape) == 2
        I = self.I
        O = full_div(original_shape[0] * original_shape[1], I)
        i, o = config.i, config.o
        # NnIi
        expr = relay.op.reshape(expr,
                                newshape=(full_div(O, o), o,
                                          full_div(I, i), i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

class BiasNIniLayout(Layout):
    def __init__(self, N):
        assert N == 1
        self.N = N

    def key(self):
        return self.N

    def to_layout(self, expr, config, original_shape):
        assert len(original_shape) == 1
        I = original_shape[0]
        i = config.i
        assert I % i == 0
        # NnIi
        expr = relay.op.reshape(expr, newshape=(1, 1, I // i, i))
        # NIni
        expr = relay.op.transpose(expr, axes=(0, 2, 1, 3))
        return expr

class IdentityLayout(Layout):
    def key(self):
        return ()

    def to_layout(self, expr, config, original_shape):
        return expr

    def from_layout(self, expr, config, original_shape):
        return expr

def safe_zip(l, r):
    assert len(l) == len(r)
    return zip(l, r)

# NIni input/output, OIoi weight
def layout_vta(expr):
    assert isinstance(expr, relay.Function)

    expr_layouts = dict()
    ops_result_layouts = dict()
    rhs_let = set()
    progress = True

    def add(expr, layout):
        nonlocal progress
        if expr not in expr_layouts:
            expr_layouts[expr] = set()
        if layout not in expr_layouts[expr]:
            expr_layouts[expr].add(layout)
            progress = True

    def add_ops(expr, in_layouts, out_layout):
        ops_result_layouts[expr] = (in_layouts, out_layout)
        add(expr, out_layout)
        for arg, layout in safe_zip(expr.args, in_layouts):
            add(arg, layout)

    class VtaLayout(ExprVisitor):
        def visit_call(self, expr):
            if isinstance(expr.op, relay.Op):
                if expr.op.name == 'nn.dense':
                    x, y = expr.args
                    x_layout = NIniLayout()
                    y_layout = OIoiLayout(get_shape(x)[1])
                    add_ops(expr, (x_layout, y_layout),  NIniLayout())
                elif expr.op.name == 'add' and all([arg in expr_layouts for arg in expr.args]):
                    x, y = expr.args
                    assert len(get_shape(x)) == 2
                    x_layout = NIniLayout()
                    if len(get_shape(y)) == 1:
                        y_layout = BiasNIniLayout(get_shape(x)[0])
                    else:
                        assert len(get_shape(y)) == 2
                        y_layout = NIniLayout()
                    add_ops(expr, (x_layout, y_layout), NIniLayout())

            super().visit_call(expr)

        def visit_let(self, expr):
            if expr.value in expr_layouts:
                for layout in expr_layouts[expr.value]:
                    add(expr.var, layout)
            rhs_let.add(expr.value)
            super().visit_let(expr)

    for param in expr.params:
        if isinstance(param.checked_type, relay.TensorType):
            add(param, IdentityLayout())

    while progress:
        progress = False
        VtaLayout().visit(expr)

    # either variable or a call with variable args
    for expr in expr_layouts:
        if isinstance(expr, relay.Var):
            pass
        elif isinstance(expr, relay.Call):
            assert(expr in rhs_let)
            assert(all([isinstance(arg, relay.Var) and arg in expr_layouts for arg in expr.args]))
        else:
            assert False
    return expr_layouts, ops_result_layouts


def rewrite_vta(expr, expr_layouts, ops_result_layouts, config=None):
    if config is None:
        config = Config()
    class VtaRewrite(ExprMutator):
        def __init__(self):
            self.vta_map = {}
            super().__init__()

        def transform_var(self, var, ll):
            assert isinstance(var, relay.Var)
            if var in expr_layouts:
                for layout in expr_layouts[var]:
                    if (var, layout) not in self.vta_map:
                        self.vta_map[(var, layout)] = ll.push(layout.to_layout(var, config, get_shape(var)))
            return var

        def visit_let(self, expr):
            if isinstance(expr.value, relay.Call) and expr.value in expr_layouts:
                assert isinstance(expr.value.op, relay.Op)
                def _with_func(ll):
                    assert expr.value in ops_result_layouts
                    _, layout = ops_result_layouts[expr.value]
                    vta_var = ll.push(self.transform(expr.value))
                    self.vta_map[(expr.value, layout)] = vta_var
                    self.vta_map[(expr.var, layout)] = vta_var
                    ll.push(layout.from_layout(vta_var, config, get_shape(expr.value)), bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            elif isinstance(expr.value, relay.Var) and expr.value in expr_layouts:
                def _with_func(ll):
                    for layout in expr_layouts[expr.value]:
                        self.vta_map[(expr.var, layout)] = self.vta_map[(expr.value, layout)]
                    ll.push(expr.value, bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            elif expr.var in expr_layouts:
                def _with_func(ll):
                    ll.push(self.visit(expr.value), bind_var=expr.var)
                    self.transform_var(expr.var, ll)
                    return self.visit(expr.body)
                return LetList.with_ll(_with_func)
            else:
                return super().visit_let(expr)

        def transform_clause(self, clause):
            def _with_func(ll):
                for var in relay.analysis.bound_vars(clause.lhs):
                    self.transform_var(var, ll)
                return self.visit(clause.rhs)
            return relay.Clause(clause.lhs, LetList.with_ll(_with_func))

        def visit_match(self, expr):
            clauses = [self.transform_clause(clauses) for clauses in expr.clauses]
            return relay.Match(self.visit(expr.data), clauses, expr.complete)

        def visit_function(self, expr):
            def _with_func(ll):
                for var in expr.params:
                    self.transform_var(var, ll)
                return self.visit(expr.body)
            return relay.Function(
                    expr.params,
                    LetList.with_ll(_with_func),
                    expr.ret_type,
                    expr.type_params,
                    expr.attrs)

        def get_vta_map(self, expr, layout):
            if (expr, layout) not in self.vta_map:
                print("not find! vta_map:")
                print(self.vta_map)
                print("not find! layout:")
                print((expr, layout))
                for x, y in self.vta_map.keys():
                    if x == expr:
                        print("expr in!")
                        print(y)
                        print(layout)
                        print(y == layout)
            assert (expr, layout) in self.vta_map
            return self.vta_map[(expr, layout)]

        def transform(self, expr):
            assert expr in ops_result_layouts
            in_layouts, _ = ops_result_layouts[expr]
            remapped_args = [self.get_vta_map(arg, layout) for arg, layout in safe_zip(expr.args, in_layouts)]
            new_call = relay.Call(expr.op, remapped_args, expr.attrs, expr.type_args)
            ty = expr.checked_type
            assert isinstance(ty, relay.TensorType)
            shape = [int(x) for x in ty.shape]

            if expr.op.name == 'add':
                return self.transform_add(new_call, shape)
            elif expr.op.name == 'nn.dense':
                return self.transform_dense(new_call, shape)
            else:
                raise

        def transform_add(self, expr, old_shape):
            return expr

        def transform_dense(self, expr, old_shape):
            expr = relay.op.nn.NCncdense(expr.args[0], expr.args[1], out_dtype="int32")
            expr = relay.op.cast(expr, dtype="int8")
            return expr

    return VtaRewrite().visit(expr)
