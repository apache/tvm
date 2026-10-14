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

"""Compilation through the native TIRx pipeline."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T


@pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", [None, "default", "tirx"])
def test_default_pipeline_allocations(pipeline):
    @T.function
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        temp = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(16):
            temp[i] = A[i] + T.float32(1)
        for i in range(16):
            B[i] = temp[i]

    module = tvm.tirx.compile(add_one, target="llvm", pipeline=pipeline)
    source = np.arange(16, dtype="float32")
    a = tvm.runtime.tensor(source)
    b = tvm.runtime.tensor(np.zeros(16, dtype="float32"))
    module(a, b)
    np.testing.assert_equal(b.numpy(), source + 1)


if __name__ == "__main__":
    tvm.testing.main()
