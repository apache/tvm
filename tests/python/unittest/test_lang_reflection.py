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


if __name__ == "__main__":
    test_make_node()
    test_const_saveload_json()
