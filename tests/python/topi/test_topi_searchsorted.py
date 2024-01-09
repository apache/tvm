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
from tvm.topi.testing import searchsorted_ref
from tvm import te, topi

topi_funcs = {"generic": topi.searchsorted, "cuda": topi.cuda.searchsorted}


def get_implementations():
    topi_func_generic = topi_funcs["generic"]
    topi_func_cuda = topi_funcs["cuda"]

    return {
        "generic": (
            lambda x, y, side, out_dtype: topi_func_generic(x, y, side, out_dtype),
            topi.generic.schedule_extern,
        ),
        "cuda": (
            lambda x, y, side, out_dtype: topi_func_cuda(x, y, side, out_dtype),
            topi.cuda.schedule_extern,
        ),
        "vulkan": (
            lambda x, y, side, out_dtype: topi_func_cuda(x, y, side, out_dtype),
            topi.cuda.schedule_extern,
        ),
    }


@tvm.testing.parametrize_targets
def test_searchsorted(dev, target):
    def verify_with_input(sorted_sequence_np, values_np, right):
        sorted_sequence = te.placeholder(sorted_sequence_np.shape, dtype="float32")
        values = te.placeholder(values_np.shape, dtype="float32")
        out_dtype = "int32"
        implementations = get_implementations()
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)

        with tvm.target.Target(target):
            indices = fcompute(sorted_sequence, values, right, out_dtype)
            s = fschedule([indices])

        func = tvm.build(s, [sorted_sequence, values, indices], target=target)
        dev = tvm.device(target, 0)

        a = tvm.nd.array(sorted_sequence_np, dev)
        b = tvm.nd.array(values_np, dev)
        c = tvm.nd.array(np.zeros(values_np.shape, dtype=indices.dtype), dev)
        func(a, b, c)
        ref = searchsorted_ref(sorted_sequence_np, values_np, right, out_dtype)
        np.testing.assert_equal(c.numpy(), ref)

    def verify(sequence_len, num_search, outer_axes, right, sorted_sequence_1d=False):
        if sorted_sequence_1d:
            sorted_sequence_shape = (sequence_len,)
        else:
            sorted_sequence_shape = outer_axes + (sequence_len,)
        values_shape = outer_axes + (num_search,)

        verify_with_input(
            np.sort(np.random.randn(*sorted_sequence_shape).astype("float32"), axis=-1),
            np.random.randn(*values_shape).astype("float32"),
            right,
        )

    verify(1024, 1000, (10, 5, 3), False)
    verify(999, 2000, (10, 5, 3), True)
    verify(1000, 1000, (), False)
    verify(2001, 100, (500,), True)
    verify(2001, 100, (500,), False, sorted_sequence_1d=True)

    # Check edge cases
    for right in [True, False]:
        sorted_sequence = np.array([1, 2, 3, 4, 5], dtype="float32")
        verify_with_input(sorted_sequence, np.array([6], dtype="float32"), right)
        verify_with_input(sorted_sequence, np.array([0], dtype="float32"), right)
