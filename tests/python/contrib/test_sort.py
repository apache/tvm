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


def test_sort():
    n = 2
    l = 5
    m = 3
    data = te.placeholder((n, l, m), name="data")
    sort_num = te.placeholder((n, m), name="sort_num", dtype="int32")
    axis = 1
    is_ascend = False
    out = te.extern(
        data.shape,
        [data, sort_num],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
        ),
        dtype="int32",
        name="sort_tensor",
    )
    input = [
        [[1, 2, 3], [2, 4.5, 3.5], [1.1, 0.5, 1], [3.2, -5, 0.5], [1.5, 0, 0]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
    ]
    sort_num_input = [[1, 2, 3], [4, 5, 5]]
    sorted_index = [
        [[0, 1, 1], [1, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[3, 4, 4], [2, 3, 3], [1, 2, 2], [0, 1, 1], [4, 0, 0]],
    ]

    ctx = tvm.cpu(0)
    target = "llvm"
    s = te.create_schedule(out.op)
    f = tvm.build(s, [data, sort_num, out], target)
    a = tvm.nd.array(np.array(input).astype(data.dtype), ctx)
    b = tvm.nd.array(np.array(sort_num_input).astype(sort_num.dtype), ctx)
    c = tvm.nd.array(np.zeros(a.shape, dtype=out.dtype), ctx)
    f(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.array(sorted_index).astype(out.dtype), rtol=1e-5)


def test_sort_np():
    dshape = (1, 2, 3, 4, 5, 6)
    axis = 4
    reduced_shape = (1, 2, 3, 4, 6)
    is_ascend = True
    data = te.placeholder(dshape, name="data")
    sort_num = te.placeholder(reduced_shape, name="sort_num", dtype="int32")
    out = te.extern(
        data.shape,
        [data, sort_num],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
        ),
        dtype="int32",
        name="sort_tensor",
    )

    ctx = tvm.cpu(0)
    target = "llvm"
    s = te.create_schedule(out.op)
    f = tvm.build(s, [data, sort_num, out], target)

    np_data = np.random.uniform(size=dshape)
    np_out = np.argsort(np_data, axis=axis)
    sort_num_input = np.full(reduced_shape, dshape[axis])
    a = tvm.nd.array(np.array(np_data).astype(data.dtype), ctx)
    b = tvm.nd.array(np.array(sort_num_input).astype(sort_num.dtype), ctx)
    c = tvm.nd.array(np.zeros(a.shape, dtype=out.dtype), ctx)
    f(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np_out, rtol=1e-5)


if __name__ == "__main__":
    test_sort()
    test_sort_np()
