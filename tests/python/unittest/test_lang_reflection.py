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

def test_make_sum():
    A = tvm.placeholder((2, 10), name='A')
    k = tvm.reduce_axis((0,10), "k")
    B = tvm.compute((2,), lambda i: tvm.sum(A[i, k], axis=k), name="B")
    json_str = tvm.save_json(B)
    BB = tvm.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None

if __name__ == "__main__":
    test_make_node()
    test_const_saveload_json()
    test_make_sum()
