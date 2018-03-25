import tvm


def test_operator_type():

    k = 1
    n = tvm.var('n')
    A = tvm.placeholder((), name='A')
    B = tvm.placeholder((10, 5), name='B')
    B1 = B[0]
    B2 = B[0,0]

    try:
        B1 + n
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + A
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B1
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B2
        assert False
    except (ValueError, tvm.TVMError):
        pass



if __name__ == "__main__":
    test_operator_type()
