import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
from tvm.relay._expr import anf_print, gnf_print
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

# TODO: figure out a way to not have to derandomize all the time
@settings(deadline=500, derandomize=True)
@given(exprs)
def test_roundtrip_pp(e):
    alpha_equal(relay.fromtext(anf_print(e)), e)

def test_gnf():
    assert gnf_print(relay.const(1)) == "v0.0.1\n1\n"
    assert gnf_print(relay.Tuple([relay.const(1), relay.const(1)])) == "v0.0.1\n%0 = 1\n%1 = 1\n(%0, %1)\n"
    one = relay.const(1)
    assert gnf_print(relay.Tuple([one, one])) == "v0.0.1\n%0 = 1\n(%0, %0)\n"
        
    assert gnf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))) == "v0.0.1\n%0 = True\nif (%0) {\n  %1 = 1\n  %2 = (%1, %1)\n  %2.0\n} else {\n  %1 = 1\n  %2 = 1\n  %3 = (%1, %1, %2)\n  %3.0\n}\n"

if __name__ == "__main__":
    # for _ in range(10):
    #     print(anf_print(exprs.example()))
    one = relay.const(1)
    tup = relay.Tuple([relay.const(1), relay.const(1)])
    print(gnf_print(relay.TupleGetItem(relay.Tuple([one, one]), 0)))
    print()
    print(gnf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    print()
    print(anf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    print()
    SEMVER = "v0.0.1"
    print(gnf_print(relay.fromtext(SEMVER+"let %x = 1; 5")))
    print(relay.fromtext(SEMVER+"let %x = 1; %x").astext())
    print(relay.fromtext(SEMVER+"let %x = (1, 1); %x").astext())
    print(relay.TupleGetItem(relay.Tuple([one, one]), 0).astext())
    print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; %x").astext())
    print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5)).astext())
    print(gnf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; 3")))
    print(gnf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; 3")))
