import tvm
from tvm import relay
from hypothesis import given, reject
from hypothesis.strategies import text, lists, integers, composite, recursive

@composite
def constants(draw):
    # python_tensor = draw(recursive(integers(), lists))
    python_tensor = draw(lists(integers()))
    # TODO: generate higher dimensional and 0D tensors. must be box shaped
    return relay.Constant(tvm.nd.array(python_tensor))

@given(constants())
def test_roundtrip(e):
    relay.fromtext(e.astext())

# @given(text())
# def test_fuzz(s):
#     try:
#         relay.fromtext(s)
#     except tvm._ffi.base.TVMError:
#         reject()

if __name__ == "__main__":
    for _ in range(10):
        print(constants().example())
