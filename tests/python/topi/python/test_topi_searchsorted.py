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
import tvm.topi.testing
from tvm import te, topi

topi_funcs = {"generic": topi.searchsorted, "cuda": topi.cuda.searchsorted}


def get_implementations():
    topi_func_generic = topi_funcs["generic"]
    topi_func_cuda = topi_funcs["cuda"]

    return {
        "generic": (
            lambda x, y: topi_func_generic(x, y),
            topi.generic.schedule_extern,
        ),
        "vulkan": (
            lambda x, y: topi_func_cuda(x, y),
            topi.cuda.schedule_extern,
        ),
    }


def searchsorted_ref(sorted_sequence, values):
    sorted_sequence_2d = np.reshape(sorted_sequence, (-1, sorted_sequence.shape[-1]))
    values_2d = np.reshape(values, (-1, values.shape[-1]))
    indices = np.zeros(values_2d.shape)

    for i in range(indices.shape[0]):
        indices[i] = np.searchsorted(sorted_sequence_2d[i], values_2d[i])

    return np.reshape(indices, values.shape)


@tvm.testing.parametrize_targets
def test_searchsorted(dev, target):
    sequence_len = 1024
    num_search = 1000
    outer_axes = (10, 5, 3)
    sorted_sequence_shape = outer_axes + (sequence_len,)
    values_shape = outer_axes + (num_search,)
    A = te.placeholder(sorted_sequence_shape, name="A", dtype="float32")
    B = te.placeholder(values_shape, name="B", dtype="float32")

    implementations = get_implementations()
    fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)

    C = fcompute(A, B)
    s = fschedule([C])

    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B, C], target=target)

    dev = tvm.device(target, 0)
    a_np = np.random.randn(*sorted_sequence_shape).astype(A.dtype)
    b_np = np.random.randn(*values_shape).astype(B.dtype)
    a_np = np.sort(a_np, axis=-1)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros(values_shape, dtype=C.dtype), dev)
    func(a, b, c)
    ref = searchsorted_ref(a_np, b_np)
    np.testing.assert_equal(c.numpy(), ref)
    print("ok")


if __name__ == "__main__":
    target = "vulkan -from_device=0"
    test_searchsorted(tvm.device(target, 0), target)
