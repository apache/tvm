import tvm

@tvm.tag_scope(tag="conv")
def compute_conv(data, weight):
    N, IC, H, W = data.shape
    OC, IC, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    ic = tvm.reduce_axis((0, IC), name='ic')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    return tvm.compute((N, OC, OH, OW), lambda i, oc, h, w: \
        tvm.sum(data[i, ic, h+dh, w+dw] * weight[oc, ic, dh, dw],
                axis=[ic, dh, dw]))

def test_with():
    n = tvm.var('n')
    m = tvm.var('m')
    l = tvm.var('l')

    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    with tvm.tag_scope(tag="gemm"):
        k = tvm.reduce_axis((0, l), name='k')
        C = tvm.compute((n, m), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    assert C.op.tag == 'gemm'

def test_decorator():
    n = tvm.var('n')
    c = tvm.var('c')
    h = tvm.var('h')
    w = tvm.var('w')
    kh = tvm.var('kh')
    kw = tvm.var('kw')

    A = tvm.placeholder((n, c, h, w), name='A')
    B = tvm.placeholder((c, c, kh, kw), name='B')
    C = compute_conv(A, B)
    assert C.op.tag == 'conv'

def test_nested():
    n = tvm.var('n')
    c = tvm.var('c')
    h = tvm.var('h')
    w = tvm.var('w')
    kh = tvm.var('kh')
    kw = tvm.var('kw')

    A = tvm.placeholder((n, c, h, w), name='A')
    B = tvm.placeholder((c, c, kh, kw), name='B')
    try:
        with tvm.tag_scope(tag='conv'):
            C = compute_conv(A, B)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    import nose
    nose.runmodule()
