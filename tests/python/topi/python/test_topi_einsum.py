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
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple


def with_tvm(lam, shapes, ops, out_shape):
    """Take numpy arrays as args, convert them to TVM tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    dev = tvm.cpu(0)
    pls = []  # placeholders
    vals_nd = []  # initial values
    for i, (shape, arg) in enumerate(zip(shapes, ops)):
        pls.append(te.placeholder(shape, name="pl" + str(i)))
        vals_nd.append(tvm.nd.array(arg, dev))

    out = lam(*pls)
    out_nd = tvm.nd.array(np.zeros(out_shape).astype(out.dtype), device=dev)
    s = te.create_schedule([out.op])
    m = tvm.build(s, pls + [out], "llvm")
    m(*(vals_nd + [out_nd]))
    return out_nd.numpy()


def verify_einsum(subscripts, shapes, shape_dict={}):
    ops = []  # ndarrays to be used as inputs
    symbolic_shapes = []  # shapes to declare the placeholders
    name_to_var = {}

    def get_concrete_shape(shape):
        return [shape_dict[s] if isinstance(s, str) else s for s in shape]

    def get_symblic_shape_var(name, dtype="int32"):
        if name not in name_to_var:
            name_to_var[name] = te.var(name, dtype=dtype)
        return name_to_var[name]

    def get_symbolic_shape(shape):
        return [get_symblic_shape_var(s) if isinstance(s, str) else s for s in shape]

    for shape in shapes:
        concrete_shape = get_concrete_shape(shape)
        tmp = np.random.uniform(low=-1.0, high=1.0, size=concrete_shape).astype(np.float32)
        ops.append(tmp)
        symbolic_shape = get_symbolic_shape(shape)
        symbolic_shapes.append(symbolic_shape)

    c1 = np.einsum(subscripts, *ops)
    out_shape = c1.shape

    if len(ops) == 1:
        c2 = with_tvm(lambda A: topi.einsum(subscripts, A), symbolic_shapes, ops, out_shape)
    elif len(ops) == 2:
        c2 = with_tvm(lambda A, B: topi.einsum(subscripts, A, B), symbolic_shapes, ops, out_shape)
    elif len(ops) == 3:
        c2 = with_tvm(
            lambda A, B, C: topi.einsum(subscripts, A, B, C), symbolic_shapes, ops, out_shape
        )

    tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "equation,inputs",
    [
        ("ii", [(5, 5)]),
        ("ii->i", [(5, 5)]),
        ("ij->i", [(5, 5)]),
        ("...j->...", [(5, 5)]),
        ("...j, j", [(5, 5), (5,)]),
        ("..., ...", [(), (2, 3)]),
        ("ijk, jil->kl", [(3, 4, 5), (4, 3, 2)]),
        ("ij, ij -> i", [(1, 4), (2, 4)]),
        ("...ij, ...jk -> ...ik", [(1, 4), (4, 2)]),
        ("...ij, ...ik -> ...jk", [(1, 1, 1, 4), (1, 1, 1, 3)]),
        ("...ik, ...jk, ...hk -> i...jh", [(3, 4, 4), (1, 5, 3, 8, 4), (2, 5, 3, 6, 4)]),
        ("ij,jk->ik", [(2, 3), (3, 4)]),
        ("ij,jk,km->im", [(2, 3), (3, 4), (4, 5)]),
    ],
)
def test_einsum(equation, inputs):
    verify_einsum(equation, inputs)


@pytest.mark.parametrize(
    "equation,inputs,shape_dict",
    [
        ("ij,jk->ik", [(2, "K"), (1, "N")], {"K": 3, "N": 4}),
        ("ij,jk->ik", [(2, "K"), ("K2", "N")], {"K": 3, "N": 4, "K2": 3}),
        ("ij,jk->ik", [(2, "K"), ("K2", "N")], {"K": 3, "N": 4, "K2": 1}),
    ],
)
def test_einsum_symblic_shape(equation, inputs, shape_dict):
    verify_einsum(equation, inputs, shape_dict)


if __name__ == "__main__":
    tvm.testing.main()
