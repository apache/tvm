import tvm
from topi.util import get_const_tuple

def test_layout():
    layout = tvm.layout("NCHW16c")
    assert layout is not None
    assert isinstance(layout, tvm.schedule.Layout)

    assert layout.factor_of("c") == 16
    assert layout.factor_of("C") == 16
    assert layout.factor_of("N") == -1

    assert layout.index_of("N") == 0
    assert layout.index_of("C") == 1
    assert layout.index_of("H") == 2
    assert layout.index_of("W") == 3
    assert layout.index_of("c") == 4
    assert layout.index_of("O") == -1

    assert "N" in layout
    assert "C" in layout
    assert "H" in layout
    assert "W" in layout
    assert "c" in layout
    assert "O" not in layout

    assert layout[0] == "N"
    assert layout[1] == "C"
    assert layout[2] == "H"
    assert layout[3] == "W"
    assert layout[4] == "c"
    assert layout[-1] == "c"

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

def test_bilayout_index():
    bilayout = tvm.bijective_layout("NCHW", "NCHW16c")
    dst_index = bilayout.forward_index([tvm.var("N"), tvm.var("C"), tvm.var("H"), tvm.var("W")])
    print(dst_index)
    assert dst_index == [tvm.var("N"), tvm.var("C")/16, tvm.var("H"), tvm.var("W"), tvm.var("C")%16]

if __name__ == "__main__":
    test_layout()
    test_bilayout_convertible()
    test_bilayout_shape()
    test_bilayout_index()
