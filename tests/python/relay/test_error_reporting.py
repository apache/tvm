from tvm.relay.parser import fromtext

def annotate_spans(expr):
    return fromtext(expr.astext())
