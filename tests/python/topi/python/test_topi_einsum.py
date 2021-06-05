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


def verify_einsum(subscripts, shapes):
    ops = []
    for shape in shapes:
        tmp = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
        ops.append(tmp)

    c1 = np.einsum(subscripts, *ops)

    if len(ops) == 1:
        c2 = with_tvm(lambda A: topi.einsum(subscripts, A), *ops)
    elif len(ops) == 2:
        c2 = with_tvm(lambda A, B: topi.einsum(subscripts, A, B), *ops)
    elif len(ops) == 3:
        c2 = with_tvm(lambda A, B, C: topi.einsum(subscripts, A, B, C), *ops)

    tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_einsum():
    verify_einsum("ii", [(5, 5)])
    verify_einsum("ii->i", [(5, 5)])
    verify_einsum("ij->i", [(5, 5)])
    verify_einsum("...j->...", [(5, 5)])
    verify_einsum("...j, j", [(5, 5), (5,)])
    verify_einsum("..., ...", [(), (2, 3)])
    verify_einsum("ijk, jil->kl", [(3, 4, 5), (4, 3, 2)])
    verify_einsum("ij, ij -> i", [(1, 4), (2, 4)])
    verify_einsum("...ij, ...jk -> ...ik", [(1, 4), (4, 2)])
    verify_einsum("...ij, ...ik -> ...jk", [(1, 1, 1, 4), (1, 1, 1, 3)])
    verify_einsum("ij,jk->ik", [(2, 3), (3, 4)])
    verify_einsum("ij,jk,km->im", [(2, 3), (3, 4), (4, 5)])


if __name__ == "__main__":
    test_einsum()
