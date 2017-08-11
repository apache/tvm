import tvm

def test_array():
    a = tvm.convert([1,2,3])
    assert len(a) == 3

def test_array_save_load_json():
    a = tvm.convert([1,2,3])
    json_str = tvm.save_json(a)
    a_loaded = tvm.load_json(json_str)
    assert(a[1].value == 2)

def test_map():
    a = tvm.var('a')
    b = tvm.var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    assert a in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert a in dd
    assert b in dd
    assert a + 1 not in amap

def test_map_save_load_json():
    a = tvm.var('a')
    b = tvm.var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    json_str = tvm.save_json(amap)
    amap = tvm.load_json(json_str)
    assert len(amap) == 2
    dd = {kv[0].name : kv[1].value for kv in amap.items()}
    assert(dd == {"a": 2, "b": 3})


if __name__ == "__main__":
    test_array()
    test_map()
    test_array_save_load_json()
    test_map_save_load_json()
