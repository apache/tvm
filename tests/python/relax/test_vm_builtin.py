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
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R


def test_multinomial_from_uniform():
    @tvm.script.ir_module
    class CallSample:
        @R.function
        def foo(x: R.Tensor((3, 5), "float32"), y: R.Tensor((3, 1), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.multinomial_from_uniform",
                x,
                y,
                sinfo_args=(R.Tensor((3, 1), dtype="int64")),
            )
            return z

    mod = CallSample
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    np_rand = np.random.rand(3, 5).astype(np.float32)
    # normalize it to get the random prob
    np_prob = np_rand / np_rand.sum(axis=1, keepdims=True)
    nd_prob = tvm.nd.array(np_prob)
    # special sample to get deterministic results
    nd_sample = tvm.nd.array(np.array([[1.1], [0], [1.001]]).astype(np.float32))

    vm = relax.VirtualMachine(ex, tvm.cpu())
    res = vm["foo"](nd_prob, nd_sample)
    tvm.testing.assert_allclose(res.numpy(), np.array([[4], [0], [4]]).astype(np.int64))


if __name__ == "__main__":
    tvm.testing.main()
