import tvm

def test_tensor():
    A = tvm.Tensor(2, name='A')
    B = tvm.Tensor(2, name='B')
    T = tvm.Tensor(3, lambda i, j, k: A(i, k) * B(j, k),
                   shape=(A.shape[0], B.shape[0], A.shape[1]))
    print(tvm.format_str(T.expr))

def test_tensor_inputs():
    A = tvm.Tensor(2, name='A')
    B = tvm.Tensor(2, name='B')
    T = tvm.Tensor(3, lambda i, j, k: A(i, k) * B(j, k),
                   shape=(A.shape[0], B.shape[0], A.shape[1]))
    assert(T.input_tensors() == [A, B])

def test_tensor_reduce():
    A = tvm.Tensor(2, name='A')
    B = tvm.Tensor(2, name='B')
    T = tvm.Tensor(3, lambda i, j, k: A(i, k) * B(j, k),
                   shape=(A.shape[0], B.shape[0], A.shape[1]))
    rd = tvm.RDom(tvm.Range(A.shape[1]))
    C = tvm.Tensor(2, lambda i, j: tvm.reduce_sum(T(i, j, rd.index[0]), rdom=rd),
                   shape=(A.shape[0], B.shape[0]))
    print(tvm.format_str(C.expr))

if __name__ == "__main__":
    test_tensor_inputs()
    test_tensor_reduce()
