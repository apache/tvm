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
import tvm
import tvm.testing
from tvm import te
import numpy as np


@tvm.testing.requires_package("torch")
def test_from_dlpack_shape_one():
    # A test case for the issue https://github.com/pytorch/pytorch/issues/99803
    import torch
    from torch.utils.dlpack import to_dlpack

    tgt = tvm.target.Target(target="llvm", host="llvm")

    rows = 1
    a = tvm.runtime.ndarray.from_dlpack(to_dlpack(torch.randn(rows, 16)))

    A = te.placeholder((rows, 16), name="A")
    B = te.placeholder((rows, 16), name="B")
    C = te.compute(A.shape, lambda i, j: A[i, j] + B[i, j], name="C")

    s = te.create_schedule(C.op)

    fadd = tvm.build(s, [A, B, C], tgt)

    dev = tvm.device(tgt.kind.name, 0)

    b = tvm.nd.array(np.random.uniform(size=(rows, 16)).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros((rows, 16), dtype=C.dtype), dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


@tvm.testing.requires_package("torch")
def test_from_dlpack_strided():
    import torch
    from torch.utils.dlpack import to_dlpack

    rows = 1
    inp = torch.randn(rows, 16)
    a = tvm.runtime.ndarray.from_dlpack(to_dlpack(inp))
    view = a._create_view((2, 8))

    np.testing.assert_equal(inp.numpy().reshape(2, 8), view.numpy())


if __name__ == "__main__":
    tvm.testing.main()
