import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
import numpy as np

from hypothesis import given, reject
from hypothesis.strategies import text, lists, integers, composite, recursive

@composite
def constants(draw):
    # python_tensor = draw(recursive(integers(), lists))
    # python_tensor = draw(lists(integers(min_value=-1000, max_value=1000)))
    python_tensor = draw(integers(min_value=-1000, max_value=1000))
    # TODO: generate higher dimensional and 0D tensors. must be box shaped
    return relay.Constant(tvm.nd.array(np.array(python_tensor).astype("int32")))

@composite
def tuples(draw):
    # TODO: replace constants with exprs
    return relay.Tuple(draw(lists(constants())))

@given(tuples())
def test_roundtrip(e):
    print(e.astext(inline_meta_data=True))
    alpha_equal(relay.fromtext(e.astext(inline_meta_data=True)), e)
    # e.astext()

if __name__ == "__main__":
    for _ in range(10):
        # print(constants().example().astext())
        print(tuples().example().astext(inline_meta_data=True))
