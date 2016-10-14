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
    T = tvm.Tensor(3, lambda i, j, k: A(i, k) * B(j, k),
                   shape=(A.shape[0], B.shape[0], A.shape[1]))
    rd = tvm.RDom(tvm.Range(A.shape[1]))
    C = tvm.Tensor(2, lambda i, j: tvm.reduce_sum(T(i, j, rd.index[0]), rdom=rd),
                   shape=(A.shape[0], B.shape[0]))

    cdom = [tvm.Range(0, 10), tvm.Range(1, 11)]
    tdom = C.infer_input_domains(cdom, inputs=[T])[T]
    assert str(tdom[0]) == "(0, 10)"


if __name__ == "__main__":
    test_range_infer()
    test_tensor_dom_infer()
