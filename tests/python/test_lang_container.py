import tvm

def test_array():
    a = tvm.convert([1,2,3])
    assert len(a) == 3

def test_map():
    a = tvm.Var('a')
    b = tvm.Var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    assert a in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert str(dd) == str(amap)
    assert a + 1 not in amap

if __name__ == "__main__":
    test_array()
    test_map()
