import tvm

def test_split_dom_infer():
    A = tvm.Tensor(2, name='A')
    rd = tvm.RDom(tvm.Range(A.shape[1]))
    split1 = tvm.Split(0, 64)
    split2 = tvm.Split(1, 64)
    split3 = tvm.Split(0, 8)
    dom = [tvm.Range(A.shape[0]), tvm.Range(A.shape[1])]
    dom1 = split1.infer_inner_domain(dom)
    dom2 = split2.infer_inner_domain(dom1)
    dom3 = split3.infer_inner_domain(dom2)
    dom4 = split3.infer_inner_domain(rd)
    i1 = split1.loop_index.name
    i2 = split2.loop_index.name
    i3 = split3.loop_index.name
    assert str(dom1) == "[((%s * 64), ((%s * 64) + 64)), (0, A_shape_1_0)]" % (i1, i1)
    assert str(dom2) == "[((%s * 64), ((%s * 64) + 64)), ((%s * 64), ((%s * 64) + 64))]" % (i1, i1, i2, i2)
    assert str(dom3) == "[(((%s * 64) + (%s * 8)), (((%s * 64) + (%s * 8)) + 8)), ((%s * 64), ((%s * 64) + 64))]" % (i1, i3, i1, i3, i2, i2)
    assert str(dom4) == "[((%s * 8), ((%s * 8) + 8))]" % (i3, i3)


if __name__ == "__main__":
    test_split_dom_infer()
