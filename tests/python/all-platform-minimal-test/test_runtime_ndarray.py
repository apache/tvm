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
"""Basic runtime enablement test."""

import tvm
from tvm import te
import numpy as np
import tvm.testing


@tvm.testing.uses_gpu
def test_nd_create():
    for target, dev in tvm.testing.enabled_targets():
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"]:
            x = np.random.randint(0, 10, size=(3, 4))
            x = np.array(x, dtype=dtype)
            y = tvm.nd.array(x, device=dev)
            z = y.copyto(dev)
            assert y.dtype == x.dtype
            assert y.shape == x.shape
            assert isinstance(y, tvm.nd.NDArray)
            np.testing.assert_equal(x, y.numpy())
            np.testing.assert_equal(x, z.numpy())
        # no need here, just to test usablity
        dev.sync()


def test_fp16_conversion():
    n = 100

    for (src, dst) in [("float32", "float16"), ("float16", "float32")]:
        A = te.placeholder((n,), dtype=src)
        B = te.compute((n,), lambda i: A[i].astype(dst))

        s = te.create_schedule([B.op])
        func = tvm.build(s, [A, B], "llvm")

        x_tvm = tvm.nd.array(100 * np.random.randn(n).astype(src) - 50)
        y_tvm = tvm.nd.array(100 * np.random.randn(n).astype(dst) - 50)

        func(x_tvm, y_tvm)

        expected = x_tvm.numpy().astype(dst)
        real = y_tvm.numpy()

        tvm.testing.assert_allclose(expected, real)


def test_dtype():
    dtype = tvm.DataType("handle")
    assert dtype.type_code == tvm.DataTypeCode.HANDLE


if __name__ == "__main__":
    test_nd_create()
    test_fp16_conversion()
    test_dtype()
