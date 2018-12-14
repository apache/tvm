from tvm.relay.parser import fromtext
from tvm.relay.expr import ExprFunctor
from tvm.relay.op import Op
from tvm import relay

class SpanChecker(ExprFunctor):
    def visit(self, expr):
        if isinstance(expr, Op):
            return self.visit_op(expr)
        else:
            return super().visit(expr)

    def visit_var(self, var):
        assert var.span

    def visit_op(self, op):
        pass

    def visit_function(self, func):
        for param in func.params:
            self.visit(param)

        self.visit(func.body)

        assert func.span

def check_spans(expr):
    sp_ck = SpanChecker()
    sp_ck.visit(expr)

def annotate_spans(expr):
    # sn = SourceName("my_expr.relay")
    return fromtext(expr.astext())

def test_var():
    x = relay.var('x')
    func = relay.Function([x], x)
    func = annotate_spans(func)
    check_spans(func)
