import tvm
import numpy as np
from tvm.sparse import *

tgt = 'llvm'

Dense = SparseFormat.Dense
Sparse = SparseFormat.Sparse

sfmt = tvm.sformat([Dense, Sparse])
print(sfmt)
# dense = dense_format(3)
# print(dense)
# spmv
n = tvm.var("n")
m = tvm.var("m")
l = tvm.var("l")
A = tvm.placeholder((m, n), sformat=tvm.sformat([Dense, Sparse]), name='A')
B = tvm.placeholder((n), sformat=tvm.sformat([Dense]), name='B')
k = tvm.reduce_axis((0, 8), 'k')
C = tvm.compute(A.shape, lambda i: tvm.sum(A[i, k] * B[k], axis=[k]), sformat=tvm.sformat([Dense]), name="C")

"""
# CSR
# row offset of A: A_rptr
# column index of A: A_cidx
# values of A: A_val
# A1_pos = [0, 2, 5, 6]
# A1_coord = [1, 2, 1, 2, 3, 3]
# A_val = [A, B, C, D, E, F]
# TACO
# for every axis:
# Ai_pos: axis index offset
# Ai_coord: axis index of A
# A_val: n-th values of A

# SPMV
# C(i) = A(i, k) * B(k)
# [0, A, B, 0,
#  0, C, D, E,
#  0, 0, 0, F,
# ]

# [Dense, Sparse]
# A1_pos = [0, 2, 5, 6]
# A1_crd = [1, 2, 1, 2, 3, 3]
# A_val = [A, B, C, D, E, F]

produce C {
  for (i, 0, 3) {
    C[i] = 0.0f
    for (idx, A1_pos[i], A1_pos[i+1]) {
       crd = A1_crd[idx]
       C[i] += A_val[idx] * B[crd]
    }
  }
}

# [Sparse, Sparse]
# [5, 1, 0, 0, 0,
#  7, 3, 0, 0, 0,
#  0, 0, 0, 0, 0,
#  8, 0, 0, 4, 9,
# ]

# A_val = [A, B, C, D, E, F]
# A_val = [5, 1, 7, 3, 8, 4, 9]
# A0_pos = [0, 3]
# A0_crd = [0, 1, 3]

# A1_pos = [0, 2, 4, 4, 7]
# A1_crd = [0, 1, 0, 1, 0, 3, 4]


# A(i_1, ..., i_k, :, ..., :)
"""

s = tvm.create_schedule(C.op)
ir = tvm.lower(s, [A, B, C], simple_mode=True)
print(ir)

raise ValueError
# run

fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() * b.asnumpy())

# print(fadd.get_source())
