from tvm.relay.parser import fromtext
from tvm.relay.expr_functor import ExprFunctor
from tvm.relay.op import Op
from tvm import relay

class SpanChecker(ExprFunctor):
    def visit_var(self, var):
        assert var.span

    def visit_op(self, op):
        pass

    def visit_function(self, func):
        for param in func.params:
            self.visit(param)

        self.visit(func.body)

        assert func.span

    def visit_call(self, call):
        assert call.span

def check_spans(expr):
    sp_ck = SpanChecker()
    sp_ck.visit(expr)

def annotate_spans(expr):
    # sn = SourceName("my_expr.relay")
    return fromtext(expr.astext())

def test_var_span():
    x = relay.var('x')
    func = relay.Function([x], x)
    func = annotate_spans(func)
    check_spans(func)

def test_type_check_call():
    x = relay.var('x', shape=(10, 10))
    func = relay.Function([x], x)
    y = relay.var('x', shape=(10, 11))
    call = relay.Call(func, [y])
    func2 = relay.Function([y], call)
    print(func2.astext())
    call = annotate_spans(func2)
    check_spans(call)
    import pdb; pdb.set_trace()
    relay.ir_pass.infer_type(call)

if __name__ == "__main__":
    test_var_span()
    test_type_check_call()
