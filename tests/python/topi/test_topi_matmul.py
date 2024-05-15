# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple
from tvm.topi.arm_cpu.matmul import compute_matmul_sme


def with_tvm(lam, *args):
    """Take numpy arrays as args, convert them to TVM tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    dev = tvm.cpu(0)
    pls = []  # placeholders
    vals_nd = []  # initial values
    for i, arg in enumerate(args):
        pls.append(te.placeholder(arg.shape, name="pl" + str(i)))
        vals_nd.append(tvm.nd.array(arg, dev))

    out = lam(*pls)
    out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), dev)
    s = te.create_schedule([out.op])
    m = tvm.build(s, pls + [out], "llvm")
    m(*(vals_nd + [out_nd]))
    return out_nd.numpy()


def verify_nn_matmul(sa, sb, transp_a, transp_b, bias=False):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    if bias:
        bias_shape = sb[-2] if transp_b else sb[-1]
        bias_np = np.random.uniform(low=-1.0, high=1.0, size=(bias_shape,)).astype(np.float32)

    a_np = a
    if transp_a:
        axes = list(range(len(sa)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        a_np = np.transpose(a_np, axes)
    b_np = b
    if transp_b:
        axes = list(range(len(sb)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        b_np = np.transpose(b_np, axes)

    if bias:
        c1 = np.matmul(a_np, b_np) + bias_np
        c2 = with_tvm(
            lambda A, B, bias: topi.nn.matmul(
                A, B, transpose_a=transp_a, transpose_b=transp_b, bias=bias
            ),
            a,
            b,
            bias_np,
        )
    else:
        c1 = np.matmul(a_np, b_np)
        c2 = with_tvm(
            lambda A, B: topi.nn.matmul(A, B, transpose_a=transp_a, transpose_b=transp_b), a, b
        )

    tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_nn_matmul():
    verify_nn_matmul((1, 1), (1, 1), False, False)
    verify_nn_matmul((1, 1), (1, 1), True, True)
    verify_nn_matmul((2, 2), (2, 2), False, False)
    verify_nn_matmul((2, 2), (2, 2), True, True)
    verify_nn_matmul((2, 3), (3, 5), False, False)
    verify_nn_matmul((5, 3), (3, 2), False, False)
    verify_nn_matmul((3, 5), (2, 3), True, True)
    verify_nn_matmul((3, 5), (3, 2), True, False)
    verify_nn_matmul((5, 3), (2, 3), False, True)
    # matmul with bias
    verify_nn_matmul((5, 3), (3, 2), False, False, True)
    verify_nn_matmul((3, 5), (2, 3), True, True, True)
    verify_nn_matmul((3, 5), (3, 2), True, False, True)
    verify_nn_matmul((5, 3), (2, 3), False, True, True)
    # batched matmul
    verify_nn_matmul((4, 5, 3), (4, 3, 2), False, False)
    verify_nn_matmul((4, 3, 5), (4, 2, 3), True, True)
    verify_nn_matmul((4, 3, 5), (4, 3, 2), True, False)
    verify_nn_matmul((4, 5, 3), (4, 2, 3), False, True)
    # batched matmul with broadcast
    verify_nn_matmul((4, 5, 3), (1, 2, 3), False, True)
    verify_nn_matmul((1, 5, 3), (4, 2, 3), False, True)
    verify_nn_matmul((5, 3), (4, 2, 3), False, True)
    verify_nn_matmul((4, 5, 3), (2, 3), False, True)
    verify_nn_matmul((2, 4, 5, 3), (1, 2, 3), False, True)
    # batched matmul with bias
    verify_nn_matmul((4, 5, 3), (4, 3, 2), False, False, True)
    verify_nn_matmul((4, 3, 5), (4, 2, 3), True, True, True)
    verify_nn_matmul((4, 3, 5), (4, 3, 2), True, False, True)
    verify_nn_matmul((4, 5, 3), (4, 2, 3), False, True, True)


def verify_matmul(sa, sb, transp_a, transp_b):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    c1 = np.matmul(np.transpose(a) if transp_a else a, np.transpose(b) if transp_b else b)
    c2 = with_tvm(lambda A, B: topi.matmul(A, B, transp_a, transp_b), a, b)
    tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_matmul():
    verify_matmul((1, 1), (1, 1), False, False)
    verify_matmul((1, 1), (1, 1), True, True)
    verify_matmul((2, 2), (2, 2), False, False)
    verify_matmul((2, 2), (2, 2), True, True)
    verify_matmul((2, 3), (3, 5), False, False)
    verify_matmul((5, 3), (3, 2), False, False)
    verify_matmul((3, 5), (3, 2), True, False)
    verify_matmul((3, 5), (2, 3), True, True)


def verify_tensordot(sa, sb, axes):
    a = np.random.uniform(low=-1.0, high=1.0, size=sa).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=sb).astype(np.float32)
    c1 = np.tensordot(a, b, axes)
    c2 = with_tvm(lambda A, B: topi.tensordot(A, B, axes), a, b)
    tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_tensordot():
    verify_tensordot((3), (3), 0)
    verify_tensordot((2, 3), (3, 5), 1)
    verify_tensordot((2, 2, 3), (2, 3, 5), 2)
    verify_tensordot((2, 2, 3, 4), (2, 3, 4, 5), 3)
    verify_tensordot((3, 2, 2), (2, 3, 5), (1, 0))
    verify_tensordot((3, 2, 2), (2, 3, 5), ((1, 0), (0, 1)))
    verify_tensordot((4, 3, 2, 2), (2, 4, 3, 5), ((1, 2, 0), (2, 0, 1)))


@pytest.mark.parametrize("in_dtype", ["float32", "float16"])
def test_unsupported_sme_matmul_compute_transpose_a(in_dtype):
    err_msg = "Transposed lhs not currently supported."
    with pytest.raises(AssertionError, match=err_msg):
        compute_matmul_sme(
            te.placeholder((32, 32), dtype=in_dtype),
            te.placeholder((32, 32), dtype=in_dtype),
            None,
            None,
            True,
            False,
        )


def test_unsupported_sme_matmul_compute_transpose_b():
    err_msg = "Rhs must be transposed when dtype is float16."
    with pytest.raises(AssertionError, match=err_msg):
        compute_matmul_sme(
            te.placeholder((32, 32), dtype="float16"),
            te.placeholder((32, 32), dtype="float16"),
            None,
            None,
            False,
            False,
        )


if __name__ == "__main__":
    tvm.testing.main()
