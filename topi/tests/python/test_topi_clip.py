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
"""Test code for clip operator"""
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import get_all_backend

def verify_clip(N, a_min, a_max, dtype):
    A = tvm.placeholder((N, N), dtype=dtype, name='A')
    B = topi.clip(A, a_min, a_max)
    s = tvm.create_schedule([B.op])

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_clip")
    def get_ref_data():
        a_np = np.random.uniform(a_min*2, a_max*2, size=(N, N)).astype(dtype)
        b_np = np.clip(a_np, a_min, a_max)
        return a_np, b_np
    a_np, b_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device, name="clip")
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_clip():
    verify_clip(1024, -127, 127, 'float32')
    verify_clip(1024, -127, 127, 'int16')
    verify_clip(1024, -127, 127, 'int8')


if __name__ == "__main__":
    test_clip()
