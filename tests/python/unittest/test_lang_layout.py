import tvm
from topi.util import get_const_tuple

def test_bilayout_convertible():
    # not convertible
    assert tvm.bijective_layout("NCHW", "ABCD") is None
    # convertible
    assert tvm.bijective_layout("NCHW", "NCHW16c") is not None

def test_bilayout_shape():
    bilayout = tvm.bijective_layout("NCHW", "NCHW16c")

    assert isinstance(bilayout, tvm.schedule.BijectiveLayout)

    dst_shape = bilayout.forward_shape((1, 32, 7, 7))
    assert get_const_tuple(dst_shape) == (1, 2, 7, 7, 16)

    src_shape = bilayout.backward_shape(dst_shape)
    assert get_const_tuple(src_shape) == (1, 32, 7, 7)

if __name__ == "__main__":
    test_bilayout_convertible()
    test_bilayout_shape()
