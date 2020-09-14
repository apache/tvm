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
# under the License
"""test of correlation operator in NCHW layout"""
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.util import get_const_tuple

_correlation_implement = {
    "generic": (topi.nn.correlation_nchw, topi.generic.schedule_correlation_nchw),
    "cuda": (topi.cuda.correlation_nchw, topi.cuda.schedule_correlation_nchw),
}


def verify_correlation_nchw(
    data_shape, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply
):
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)"
        % (
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3],
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            pad_size,
            is_multiply,
        )
    )

    A = te.placeholder(data_shape, name="data1")
    B = te.placeholder(data_shape, name="data2")
    dtype = A.dtype

    @memoize("topi.tests.test_topi_correlation_nchw.verify_correlation_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=data_shape).astype(dtype)
        b_np = np.random.uniform(size=data_shape).astype(dtype)
        c_np = tvm.topi.testing.correlation_nchw_python(
            a_np, b_np, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply
        )
        return a_np, b_np, c_np

    a_np, b_np, c_np = get_ref_data()

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _correlation_implement)
        with tvm.target.Target(device):
            C = fcompute(
                A, B, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply
            )
            s = fschedule([C])

            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            c = tvm.nd.empty(c_np.shape, dtype=dtype, ctx=ctx)

            func = tvm.build(s, [A, B, C], device)
            func(a, b, c)
            tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_correlation_nchw():
    verify_correlation_nchw(
        (1, 3, 10, 10),
        kernel_size=1,
        max_displacement=4,
        stride1=1,
        stride2=1,
        pad_size=4,
        is_multiply=True,
    )
    verify_correlation_nchw(
        (1, 3, 10, 10),
        kernel_size=1,
        max_displacement=5,
        stride1=1,
        stride2=1,
        pad_size=5,
        is_multiply=True,
    )
    verify_correlation_nchw(
        (5, 1, 4, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=1,
        pad_size=2,
        is_multiply=True,
    )
    verify_correlation_nchw(
        (5, 1, 6, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=2,
        pad_size=2,
        is_multiply=False,
    )
    verify_correlation_nchw(
        (5, 1, 11, 11),
        kernel_size=5,
        max_displacement=1,
        stride1=1,
        stride2=1,
        pad_size=2,
        is_multiply=False,
    )


if __name__ == "__main__":
    test_correlation_nchw()
