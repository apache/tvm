import tvm
import numpy as np

def test_sort():
    n = 2
    l = 5
    m = 3
    data = tvm.placeholder((n, l, m), name='data')
    sort_num = tvm.placeholder((n, m), name="sort_num", dtype="int32")
    axis = 1
    is_descend = True
    out = tvm.extern(data.shape, [data, sort_num],
                     lambda ins, outs: tvm.call_packed(
                         "tvm.contrib.sort.argsort", ins[0],
                         ins[1], outs[0], axis, is_descend),
                     dtype='int32', name="sort_tensor")
    input = [[[1, 2, 3], [2, 4.5, 3.5], [1.1, 0.5, 1], [3.2, -5, 0.5], [1.5, 0, 0]],
             [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]]
    sort_num_input = [[1, 2, 3], [4, 5, 5]]
    sorted_index = [[[0, 1, 1], [1, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                    [[3, 4, 4], [2, 3, 3], [1, 2, 2], [0, 1, 1], [4, 0, 0]]]

    ctx = tvm.cpu(0)
    target = "llvm"
    s = tvm.create_schedule(out.op)
    f = tvm.build(s, [data, sort_num, out], target)
    a = tvm.nd.array(np.array(input).astype(data.dtype), ctx)
    b = tvm.nd.array(np.array(sort_num_input).astype(sort_num.dtype), ctx)
    c = tvm.nd.array(np.zeros(a.shape, dtype=out.dtype), ctx)
    f(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), np.array(sorted_index).astype(out.dtype), rtol=1e-5)

if __name__ == "__main__":
    test_sort()
