from tvm.ir import Op
from tvm import relay

from tvm.relay import ExprVisitor
from tvm.relay.function import Function
from tvm.relay.expr import Call, Let, Var, GlobalVar
from tvm.relay.expr import If, Tuple, TupleGetItem, Constant
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause

# TODO Make local_count some more generic name and separate this into a base class that
# allows stuff to be passed around and a specific implementation for counting depth.
# Also make one for exprmutator. Good to have both.
# Add to relay.whatever

class DepthCounter(ExprVisitor):
    """Determine how many operations of the specified type are in the graph."""
    def __init__(self, valid_ops):
        self.depth_count = 0
        self.valid_ops = [relay.op.get(op) for op in valid_ops]
        super().__init__()

    # pylint: disable=no-else-return
    def visit(self, expr, local_count=0):
        """Apply the visitor to an expression."""
        if expr in self.memo_map:
            return self.memo_map[expr]

        if isinstance(expr, Function):
            res = self.visit_function(expr, local_count)
        elif isinstance(expr, Call):
            res = self.visit_call(expr, local_count)
        elif isinstance(expr, Let):
            res = self.visit_let(expr, local_count)
        elif isinstance(expr, Var):
            res = self.visit_var(expr, local_count)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr, local_count)
        elif isinstance(expr, If):
            res = self.visit_if(expr, local_count)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr, local_count)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr, local_count)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr, local_count)
        elif isinstance(expr, Op):
            res = self.visit_op(expr, local_count)
        elif isinstance(expr, RefCreate):
            res = self.visit_ref_create(expr, local_count)
        elif isinstance(expr, RefRead):
            res = self.visit_ref_read(expr, local_count)
        elif isinstance(expr, RefWrite):
            res = self.visit_ref_write(expr, local_count)
        elif isinstance(expr, Constructor):
            res = self.visit_constructor(expr, local_count)
        elif isinstance(expr, Match):
            res = self.visit_match(expr, local_count)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        self.memo_map[expr] = res

        return res
        
    def visit_call(self, call, local_count):
        if call.op in self.valid_ops:
            local_count = local_count + 1
            self.depth_count = max(self.depth_count, local_count)
        for arg in call.args:
            self.visit(arg, local_count)
        
    def visit_tuple(self, tup, local_count):
        for x in tup.fields:
            self.visit(x, local_count)

    def visit_var(self, var, local_count):
        pass

    def visit_let(self, let, local_count):
        self.visit(let.var, local_count)
        self.visit(let.value, local_count)
        self.visit(let.body, local_count)

    def visit_function(self, f, local_count):
        self.visit(f.body, local_count)

    def visit_if(self, i, local_count):
        self.visit(i.cond, local_count)
        self.visit(i.true_branch, local_count)
        self.visit(i.false_branch, local_count)

    def visit_global_var(self, gv, local_count):
        pass

    def visit_constructor(self, c, local_count):
        pass

    def visit_op(self, op, local_count):
        pass

    def visit_constant(self, const, local_count):
        pass

    def visit_ref_create(self, r, local_count):
        self.visit(r.value, local_count)

    def visit_ref_read(self, r, local_count):
        self.visit(r.ref, local_count)

    def visit_ref_write(self, r, local_count):
        self.visit(r.ref, local_count)
        self.visit(r.value, local_count)

    def visit_tuple_getitem(self, t, local_count):
        self.visit(t.tuple_value, local_count)

    def visit_match(self, m, local_count):
        self.visit(m.data, local_count)
        for c in m.clauses:
            self.visit(c.rhs, local_count)