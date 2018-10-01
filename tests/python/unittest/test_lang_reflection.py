import tvm

def test_const_saveload_json():
    # save load json
    x = tvm.const(1)
    y = tvm.const(10)
    z = x + y
    z = z + z
    json_str = tvm.save_json(z)
    zz = tvm.load_json(json_str)
    assert tvm.save_json(zz) == tvm.save_json(z)


def test_make_smap():
    # save load json
    x = tvm.const(1)
    y = tvm.const(10)
    z = tvm.expr.Add(x, y)
    smap = tvm.convert({"z": z, "x": x})
    json_str = tvm.save_json(tvm.convert([smap]))
    arr = tvm.load_json(json_str)
    assert len(arr) == 1
    assert arr[0]["z"].a == arr[0]["x"]


def test_make_node():
    x = tvm.make.node("IntImm", dtype="int32", value=10)
    assert isinstance(x, tvm.expr.IntImm)
    assert x.value == 10
    A = tvm.placeholder((10, ), name='A')
    AA = tvm.make.node("Tensor",
                       shape=A.shape,
                       dtype=A.dtype,
                       op=A.op,
                       value_index=A.value_index)
    assert AA.op == A.op
    assert AA.value_index == A.value_index


def test_make_attrs():
    try:
        x = tvm.make.node("attrs.TestAttrs", unknown_key=1, name="xx")
        assert False
    except tvm.TVMError as e:
        assert str(e).find("unknown_key") != -1

    try:
        x = tvm.make.node("attrs.TestAttrs", axis=100, name="xx")
        assert False
    except tvm.TVMError as e:
        assert str(e).find("upper bound") != -1

    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4))
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10


    dattr = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert dattr.x.value == 1
    datrr = tvm.load_json(tvm.save_json(dattr))
    assert dattr.name.value == "xyz"



def test_make_sum():
    A = tvm.placeholder((2, 10), name='A')
    k = tvm.reduce_axis((0,10), "k")
    B = tvm.compute((2,), lambda i: tvm.sum(A[i, k], axis=k), name="B")
    json_str = tvm.save_json(B)
    BB = tvm.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None


def test_env_func():
    @tvm.register_func("test.env_func")
    def test(x):
        return x + 1

    f = tvm.get_global_func("test.env_func")
    x = tvm.get_env_func("test.env_func")
    assert x.name == "test.env_func"
    json_str = tvm.save_json([x])
    y = tvm.load_json(json_str)[0]
    assert y.name == x.name
    assert y(1) == 2
    assert y.func(1) == 2

    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4), func=y)
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10
    x = tvm.load_json(tvm.save_json(x))
    assert isinstance(x.func, tvm.container.EnvFunc)
    assert x.func(10) == 11


if __name__ == "__main__":
    test_env_func()
    test_make_attrs()
    test_make_node()
    test_make_smap()
    test_const_saveload_json()
    test_make_sum()
