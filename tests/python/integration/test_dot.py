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
"""Test scheduling and running a dot product."""
import numpy as np

import tvm
import tvm.testing
from tvm import te


@tvm.testing.requires_llvm
def test_dot():
    """Test dot product."""
    arr_length = 12
    arr_length_tvm = tvm.runtime.convert(arr_length)
    placeholder_a = te.placeholder((arr_length_tvm,), name="A")
    placeholder_b = te.placeholder((arr_length_tvm,), name="B")
    reduce_axis_k = te.reduce_axis((0, arr_length_tvm), "k")
    result_c = te.compute(
        (),
        lambda: te.sum(
            placeholder_a[reduce_axis_k] * placeholder_b[reduce_axis_k], axis=reduce_axis_k
        ),
        name="C",
    )
    schedule = te.create_schedule(result_c.op)

    def verify(target):
        f = tvm.driver.build(schedule, [placeholder_a, placeholder_b, result_c], target)
        # verify
        dev = tvm.cpu(0)
        buff_a = tvm.nd.array(
            np.random.uniform(size=(arr_length,)).astype(placeholder_a.dtype), dev
        )
        buff_b = tvm.nd.array(
            np.random.uniform(size=(arr_length,)).astype(placeholder_b.dtype), dev
        )
        buff_c = tvm.nd.array(np.zeros((), dtype=result_c.dtype), dev)
        f(buff_a, buff_b, buff_c)
        tvm.testing.assert_allclose(
            buff_c.numpy(), np.dot(buff_a.numpy(), buff_b.numpy()), rtol=1e-4
        )

    verify("llvm")


if __name__ == "__main__":
    test_dot()
