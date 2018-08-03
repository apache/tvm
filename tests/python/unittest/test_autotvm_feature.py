"""Test feature extraction"""

import numpy as np

import tvm
from tvm.autotvm import feature

def test_iter_feature_gemm():
    N = 128

    k = tvm.reduce_axis((0, N), 'k')
    A = tvm.placeholder((N, N), name='A')
    B = tvm.placeholder((N, N), name='B')
    C = tvm.compute(
        A.shape,
        lambda y, x: tvm.sum(A[y, k] * B[k, x], axis=k),
        name='C')

    s = tvm.create_schedule(C.op)

    feas = feature.get_itervar_feature(s, [A, B, C], take_log=False)

    expected = [
        {
            '_attr_': [128, 1, 128, 2097152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'A_0': [128, -1, 16384, 128, 0, 0], 'B_0': [0, -1, 16384, 128, 0, 0],
            'C_0': [128, -1, 16384, 128, 0, 0], 'C_1': [128, -1, 16384, 128, 0, 0],
        },
        {
            '_attr_': [128, 2, 16384, 16384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'A_0': [0, -1, 128, 128, 0, 0], 'B_0': [1, -1, 16384, 1, 0, 0],
            'C_0': [1, -1, 128, 128, 0, 0], 'C_1': [1, -1, 128, 128, 0, 0],
        },
        {
            '_attr_': [128, 3, 2097152, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'A_0': [1, -1, 128, 1, 0, 0], 'B_0': [128, -1, 128, 1, 0, 0],
            'C_1': [0, -1, 1, 128, 0, 0], 'C_2':  [0, -1, 1, 128, 0, 0],
        }
    ]

    for ans, row in zip(expected, feas):
        for pair in row:
            if pair[0] not in ans:
                continue
            assert ans[pair[0]] == pair[1:], "%s: %s vs %s" % (pair[0], ans[pair[0]], pair[1:])


def test_feature_shape():
    """test the dimensions of flatten feature are the same"""

    N = 1024
    n_sample = 100

    def get_gemm_feature(target):
        k = tvm.reduce_axis((0, N), 'k')
        A = tvm.placeholder((N, N), name='A')
        B = tvm.placeholder((N, N), name='B')
        C = tvm.compute(A.shape, lambda y, x: tvm.sum(A[y, k] * B[k, x], axis=k),
                        name='C')

        s = tvm.create_schedule(C.op)

        y, x = s[C].op.axis
        axes = list(s[C].tile(y, x, 8, 8)) + [k]
        perm = np.random.permutation(5)
        axes = [axes[x] for x in perm]
        s[C].reorder(*axes)

        if "gpu" in target.keys:
            pick = []
            # filter out reduction axis
            for i in range(len(perm)):
                if perm[i] != 4:
                    pick.append(axes[i])
            s[C].bind(pick[0], tvm.thread_axis("blockIdx.x"))
            s[C].bind(pick[1], tvm.thread_axis("vthread"))
            s[C].bind(pick[2], tvm.thread_axis("threadIdx.y"))

        with target:
            feas = feature.get_itervar_feature(s, [A, B, C])
            feas = feature.flatten_itervar_feature(feas)
        return feas

    targets = [
        tvm.target.cuda(),
        tvm.target.mali(),
        tvm.target.arm_cpu(),
    ]

    for target in targets:
        dim = len(get_gemm_feature(target))
        for i in range(n_sample):
            assert dim == len(get_gemm_feature(target)), "dimensions of feature do not match" \
                                                   " for different configurations"


if __name__ == "__main__":
    test_iter_feature_gemm()
    test_feature_shape()
