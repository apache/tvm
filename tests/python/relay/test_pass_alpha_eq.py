import tvm
from tvm import relay

def test_type_alpha_eq():
    t1 = relay.ty.TensorType((3, 4), "float32")
    t2 = relay.ty.TensorType((3, 4), "float32")
    t3 = relay.ty.TensorType((3, 4, 5), "float32")
    assert t1 == t2
    assert t1 != t3

    t1 = relay.ty.TensorType((), "float32")
    t2 = relay.ty.TensorType((), "float32")
    assert t1 == t2


if __name__ == "__main__":
    test_type_alpha_eq()
