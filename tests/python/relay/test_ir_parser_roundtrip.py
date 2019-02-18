import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
from tvm.relay._expr import pretty_print
import numpy as np

from hypothesis import given, reject, settings
from hypothesis.strategies import text, lists, integers, composite, recursive, deferred

exprs = deferred(lambda: constants()
                    #    | projections(exprs)
                       | tuples(exprs))

@composite
def constants(draw):
    # python_tensor = draw(recursive(integers(), lists))
    # python_tensor = draw(lists(integers(min_value=-1000, max_value=1000)))
    python_tensor = draw(integers(min_value=-1000, max_value=1000))
    # TODO: generate higher dimensional and 0D tensors. must be box shaped
    return relay.Constant(tvm.nd.array(np.array(python_tensor).astype("int32")))

@composite
def tuples(draw, field_type):
    return relay.Tuple(draw(lists(field_type, max_size=5)))

@composite
def projections(draw, field_type):
    return relay.TupleGetItem(draw(field_type), draw(integers(min_value=-1000, max_value=1000)))

""" @given(tuples())
def test_roundtrip(e):
    print(e.astext(inline_meta_data=True))
    alpha_equal(relay.fromtext(e.astext(inline_meta_data=True)), e)
    # e.astext() """

@settings(deadline=500)
@given(exprs)
def test_roundtrip_pp(e):
    alpha_equal(relay.fromtext(pretty_print(e)), e)

# def test_roundtrip_pp_simple():
#     print(pretty_print(relay.const(1)))

if __name__ == "__main__":
    for _ in range(10):
        # print(constants().example().astext())
        # print(tuples().example().astext(inline_meta_data=True))
        print(pretty_print(exprs.example()))
