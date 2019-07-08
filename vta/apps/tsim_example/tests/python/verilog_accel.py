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
import tsim

def test_accel():
    rmax = 64
    n = np.random.randint(1, rmax)
    c = np.random.randint(0, rmax)
    ctx = tvm.cpu(0)
    a = tvm.nd.array(np.random.randint(rmax, size=n).astype("uint64"), ctx)
    b = tvm.nd.array(np.zeros(n).astype("uint64"), ctx)
    f = tsim.load_module()
    cycles = f(a, b, c)
    msg = "cycles:{0:4} n:{1:2} c:{2:2}".format(cycles, n, c)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + c, err_msg = "[FAIL] " + msg)
    print("[PASS] " + msg)

if __name__ == "__main__":
    tsim.init("verilog")
    for i in range(10):
        test_accel()
