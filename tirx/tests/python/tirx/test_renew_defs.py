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

from __future__ import annotations

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.ir import Var
from tvm_tirx.script import tirx as T


def _check_tensor_decl(lhs: Var, rhs: Var):
    assert lhs != rhs
    assert lhs.data != rhs.data


def test_undefined_tensor():
    @T.function
    def access_alloc():
        # Var A should be remapped
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float16")])
        T.evaluate(A.data)
        for i in range(128):
            A[i] = A[i] + T.float16(1.0)

    f1 = access_alloc
    f2 = tvm_tirx.tirx.renew_def(f1)
    tvm_tirx.ir.assert_structural_equal(f1, f2)

    # AllocTensor is now a flat statement in SeqStmt
    assert f1.body.seq[0].var.data != f2.body.seq[0].var.data

    def _get_tensor_store_tensor(f):
        # SeqStmt: [AllocTensor, Evaluate, For]; For body has the TensorStore
        return f.body.seq[2].body[0].dest

    _check_tensor_decl(_get_tensor_store_tensor(f1), _get_tensor_store_tensor(f2))


def test_symbolic_func():
    m = T.dynamic("m", "int32")

    @T.function
    def symbolic_func(
        A: T.Tensor((n, m), "float32"),  # noqa: F821
        B: T.Tensor((n, m * 2), "float32"),  # noqa: F821
        n: T.int32,
    ):
        for i, j in T.grid(n, m):
            B[i, j * 2] = A[i, j]
            B[i, j * 2 + 1] = A[i, j]

    f1 = symbolic_func
    f2 = tvm_tirx.tirx.renew_def(f1)
    tvm_tirx.ir.assert_structural_equal(f1, f2)


def test_compound_tensor_param_shape_var():
    n = tvm_tirx.tirx.Var("n", "int32")
    A = tvm_tirx.tirx.Var(
        "tensor", tvm_tirx.tirx.TensorType((tvm_tirx.tirx.max(n, 1),), "float32", layout=None)
    )
    f1 = tvm_tirx.tirx.Function([A], tvm_tirx.ir.Evaluate(n))
    f2 = tvm_tirx.tirx.renew_def(f1)

    tvm_tirx.ir.assert_structural_equal(f1, f2)
    assert not f1.body[0].value.same_as(f2.body[0].value)
    assert f2.params[0].shape[0].a.same_as(f2.body[0].value)


if __name__ == "__main__":
    tvm_tirx.testing.main()
