import tvm

def test_range_infer():
    x = tvm.Var('x')
    y = tvm.Var('y')
    t = tvm.Var('t')
    z = x + y + t
    zr = tvm.infer_range(z, {x: tvm.Range(10, 20), y : tvm.Range(10, 11)})
    assert str(zr) == "((t0 + 20), (t0 + 30))"

def test_tensor_dom_infer():
    A = tvm.Tensor(2, name='A')
    B = tvm.Tensor(2, name='B')
    rd = tvm.RDom(tvm.Range(A.shape[1]))
    T = tvm.Tensor(2, lambda i, j:
                   tvm.reduce_sum(A(i, rd.index[0]) * B(j, rd.index[0]), rdom=rd),
                   shape=(A.shape[0], B.shape[0]))
    C = tvm.Tensor(2, lambda i, j: T(i,j),
                   shape=(A.shape[0], B.shape[0]))

    cdom = [tvm.Range(0, 10), tvm.Range(1, 11)]
    tdom = C.infer_input_domains(cdom, inputs=[T])[T]
    assert T.is_rtensor
    assert str(tdom[0]) == "(0, 10)"


if __name__ == "__main__":
    test_range_infer()
    test_tensor_dom_infer()
