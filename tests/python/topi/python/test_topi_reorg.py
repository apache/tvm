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
"""Example code to do reorg."""
import numpy as np
from tvm import topi
from tvm.topi.utils import get_const_tuple
import tvm
from tvm import te
import tvm.topi.testing
import tvm.testing

_reorg_schedule = {
    "generic": topi.generic.schedule_reorg,
    "gpu": topi.cuda.schedule_reorg,
}


def verify_reorg(batch, in_size, in_channel, stride):
    """Verify reorg operator by comparing outputs from tvm and numpy implementation"""
    in_height = in_width = in_size

    A = te.placeholder((batch, in_channel, in_height, in_width), name="A")
    B = topi.vision.reorg(A, stride)

    a_shape = get_const_tuple(A.shape)
    dtype = A.dtype

    def get_ref_data_reorg():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_np = tvm.topi.testing.reorg_python(a_np, stride)
        return a_np, b_np

    a_np, b_np = get_ref_data_reorg()

    def check_device(device):
        """Cheching devices is enabled or not"""
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            s_func = tvm.topi.testing.dispatch(device, _reorg_schedule)
            s = s_func([B])
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        func = tvm.build(s, [A, B], device)
        func(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)

    for device in ["llvm", "cuda"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_reorg():
    verify_reorg(1, 20, 8, 2)


if __name__ == "__main__":
    test_reorg()
