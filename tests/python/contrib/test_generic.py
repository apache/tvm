import tvm
import numpy as np

def test_sort():
    n = 1
    l = 5
    m = 3
    data = tvm.placeholder((n, l, m), name='data')
    num_element = tvm.placeholder((n,), name="num_elem", dtype='int32')
    sorted_value_index = 1
    is_descend = True
    out = tvm.extern((n, l), [data, num_element],
                     lambda ins, outs: tvm.call_packed(
                         "tvm.contrib.generic.utils.stable_sort", ins[0], ins[1],
                         outs[0], n, sorted_value_index, is_descend),
                     dtype='int32', name="sort_tensor")
    input = [[[1, 2, 3], [2, 4.5, 3], [1, 0.5, 1], [3.2, -5, 0], [1, 0, 0]]]
    sorted_index = [[1, 0, 2, 4, 3]]

    ctx = tvm.cpu(0)
    target = "llvm"
    s = tvm.create_schedule(out.op)
    f = tvm.build(s, [data, num_element, out], target)
    a = tvm.nd.array(np.array(input).astype(data.dtype), ctx)
    b = tvm.nd.array(np.array([5]).astype(num_element.dtype), ctx)
    c = tvm.nd.array(np.zeros((n, l), dtype=out.dtype), ctx)
    f(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), np.array(sorted_index).astype(out.dtype), rtol=1e-5)

if __name__ == "__main__":
    test_sort()
