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
import pytest
import tvm
from tvm import te, topi
from tvm.meta_schedule.testing.te_workload import conv2d_winograd_nhwc, matmul
from tvm.tir.analysis import find_anchor_block


def test_matmul_add():
    n = m = k = 128
    A, B, C = matmul(n, m, k)
    mod = tvm.IRModule()
    mod["main"] = te.create_prim_func([A, B, C + A])

    block = find_anchor_block(mod)

    assert block.name_hint == "C"


def test_winograd():
    mod = tvm.IRModule()
    mod["main"] = te.create_prim_func(conv2d_winograd_nhwc(1, 14, 14, 128, 128, 6))

    block = find_anchor_block(mod)

    assert block.name_hint == "bgemm"


def test_no_anchor_block():
    inp = te.placeholder((10,), name="input")
    out = topi.nn.relu(inp + 1.0)
    mod = tvm.IRModule()
    mod["main"] = te.create_prim_func([inp, out])

    assert find_anchor_block(mod) is None


if __name__ == "__main__":
    tvm.testing.main()
