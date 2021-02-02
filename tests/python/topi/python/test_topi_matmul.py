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
import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple


def with_tvm(lam, *args):
    """Take numpy arrays as args, convert them to TVM tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    ctx = tvm.cpu(0)
    pls = []  # placeholders
    vals_nd = []  # initial values
    for i, arg in enumerate(args):
        pls.append(te.placeholder(arg.shape, name="pl" + str(i)))
        vals_nd.append(tvm.nd.array(arg, ctx))

    out = lam(*pls)
    out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), ctx)
    s = te.create_schedule([out.op])
    m = tvm.build(s, pls + [out], "llvm")
    m(*(vals_nd + [out_nd]))
    return out_nd.asnumpy()


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


if __name__ == "__main__":
    test_matmul()
    test_tensordot()
