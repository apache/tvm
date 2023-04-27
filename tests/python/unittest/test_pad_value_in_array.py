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
from tvm import relay

# This used to crash


def test_pad_value_in_array():
    A = relay.var("A", shape=(32, 32), dtype="int8")

    p0 = relay.Constant(tvm.nd.array(np.array([2], dtype="int8")))
    p1 = relay.nn.pad(A, pad_value=p0, pad_width=((1, 1), (1, 1)))

    func = relay.Function(relay.analysis.free_vars(p1), p1)
    mod = tvm.IRModule.from_expr(func)

    target = "llvm"
    lib = relay.build(
        mod,
        tvm.target.Target(target, host=target),
        runtime=relay.backend.Runtime("cpp"),
        executor=relay.backend.Executor("aot", {"unpacked-api": False, "interface-api": "packed"}),
    )


if __name__ == "__main__":
    tvm.testing.main()
