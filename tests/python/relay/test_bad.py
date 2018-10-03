import tvm
from tvm import relay
from tvm.relay.ir_pass import bad

def test_bad():
    ft = relay.TensorType([], 'float32')
    bad(relay.TupleType([ft, ft, ft, ft]))
