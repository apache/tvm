from tvm.relay.parser import fromtext
from tvm import relay

def annotate_spans(expr):
    # sn = SourceName("my_expr.relay")
    return fromtext(expr.astext())

def test_var():
    x = relay.var('x')
    func = relay.Function([x], x)
    func = annotate_spans(func)
    import pdb; pdb.set_trace()
