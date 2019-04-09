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
import numpy as np

def enabled_ctx_list():
    ctx_list = [('cpu', tvm.cpu(0)),
                ('gpu', tvm.gpu(0)),
                ('cl', tvm.opencl(0)),
                ('metal', tvm.metal(0)),
                ('rocm', tvm.rocm(0)),
                ('vulkan', tvm.vulkan(0)),
                ('vpi', tvm.vpi(0))]
    for k, v  in ctx_list:
        assert tvm.context(k, 0) == v
    ctx_list = [x[1] for x in ctx_list if x[1].exist]
    return ctx_list

ENABLED_CTX_LIST = enabled_ctx_list()
print("Testing using contexts:", ENABLED_CTX_LIST)


def test_nd_create():
    for ctx in ENABLED_CTX_LIST:
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32",
                      "float32"]:
            x = np.random.randint(0, 10, size=(3, 4))
            x = np.array(x, dtype=dtype)
            y = tvm.nd.array(x, ctx=ctx)
            z = y.copyto(ctx)
            assert y.dtype == x.dtype
            assert y.shape == x.shape
            assert isinstance(y, tvm.nd.NDArray)
            np.testing.assert_equal(x, y.asnumpy())
            np.testing.assert_equal(x, z.asnumpy())
        # no need here, just to test usablity
        ctx.sync()


def test_fp16_conversion():
    n = 100

    for (src, dst) in [('float32', 'float16'), ('float16', 'float32')]:
        A = tvm.placeholder((n,), dtype=src)
        B = tvm.compute((n,), lambda i: A[i].astype(dst))

        s = tvm.create_schedule([B.op])
        func = tvm.build(s, [A, B], 'llvm')

        x_tvm = tvm.nd.array(100 * np.random.randn(n).astype(src) - 50)
        y_tvm = tvm.nd.array(100 * np.random.randn(n).astype(dst) - 50)

        func(x_tvm, y_tvm)

        expected = x_tvm.asnumpy().astype(dst)
        real = y_tvm.asnumpy()

        tvm.testing.assert_allclose(expected, real)

if __name__ == "__main__":
    test_nd_create()
    test_fp16_conversion()
