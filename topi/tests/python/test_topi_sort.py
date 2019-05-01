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
"""Test code for vision package"""
from __future__ import print_function
import math
import numpy as np
import tvm
import topi
import topi.testing

from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi import argsort

def test_argsort():
    dshape = (1, 8)
    valid_count_shape = (2,)
    data = tvm.placeholder(dshape, name="data", dtype="float32")
    valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
    np_data = np.random.rand(dshape[0], dshape[1]).astype(data.dtype)
    np_valid_count = np.array([4]).astype(valid_count.dtype)
    np_result = np.argsort(-np_data)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = argsort(data, valid_count, axis = -1, is_ascend = False, flag=False)
            s = topi.generic.schedule_argsort(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype="float32"), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result.astype("float32"), rtol=1e0)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)


if __name__ == "__main__":
    test_argsort()
