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
    def verify(sequence_len, num_search, outer_axes, side):
        sorted_sequence_shape = outer_axes + (sequence_len,)
        values_shape = outer_axes + (num_search,)
        sorted_sequence = te.placeholder(sorted_sequence_shape, dtype="float32")
        values = te.placeholder(values_shape, dtype="float32")
        out_dtype = "int32"
        implementations = get_implementations()
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)

        with tvm.target.Target(target):
            indices = fcompute(sorted_sequence, values, side, out_dtype)
            s = fschedule([indices])

        func = tvm.build(s, [sorted_sequence, values, indices], target=target)
        dev = tvm.device(target, 0)

        a_np = np.random.randn(*sorted_sequence_shape).astype(sorted_sequence.dtype)
        b_np = np.random.randn(*values_shape).astype(values.dtype)
        a_np = np.sort(a_np, axis=-1)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(values_shape, dtype=indices.dtype), dev)
        func(a, b, c)
        ref = searchsorted_ref(a_np, b_np, side, out_dtype)
        np.testing.assert_equal(c.numpy(), ref)

    # The first argument is the range of binary search
    verify(1024, 1000, (10, 5, 3), "left")
    verify(999, 2000, (10, 5, 3), "right")
    verify(1000, 1000, (), "left")
    verify(2001, 100, (500), "right")
