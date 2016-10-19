import tvm

def test_schedule():
    A = tvm.Tensor(2, name='A')
    B = tvm.Tensor(2, name='B')
    rd = tvm.RDom(tvm.Range(A.shape[1]))
    T = tvm.Tensor(2, lambda i, j:
                   tvm.reduce_sum(A(i, rd.index[0]) * B(j, rd.index[0]), rdom=rd),
                   shape=(A.shape[0], B.shape[0]), name="T")
    C = tvm.Tensor(2, lambda i, j: T(i,j),
                   shape=(A.shape[0], B.shape[0]), name="C")

    bufA = tvm.Buffer(tvm.Scope.Thread, name='A')
    bufB = tvm.Buffer(tvm.Scope.Thread, name='B')
    bufT = tvm.Buffer(tvm.Scope.Thread, name='T')

    schA = tvm.Schedule(A, buffer=bufA)
    schB = tvm.Schedule(B, buffer=bufB)
    schT = tvm.Schedule(T, buffer=bufT)
    schC = tvm.Schedule(C)
    Cx0 = tvm.Split(dim=0, factor=64)
    Cy0 = tvm.Split(dim=1, factor=64)
    Cx1 = tvm.Split(dim=0, factor=8)
    Cy1 = tvm.Split(dim=1, factor=8)
    Tk = tvm.Split(dim=0, factor=8, rdom=True)

    schC.add_split(Cx0)
    schC.add_split(Cy0)
    schC.add_split(Cx1)
    schC.add_split(Cy1)
    schT.add_split(Tk)
    schC.attach(Cy1, schT)
    schT.attach(Tk, schA)
    schT.attach(Tk, schB)

    body = schC.realize()
    print('\n'.join(body))


if __name__ == "__main__":
    test_schedule()
