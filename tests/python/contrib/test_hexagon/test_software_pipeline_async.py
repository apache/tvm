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

import sys
import pytest
import numpy as np

import tvm
from tvm import tir
from tvm.contrib.hexagon.session import Session
from tvm.script import tir as T

outer = 16
inner = 128


@T.prim_func
def plus_one_primfunc(A: T.Buffer[(outer, inner), "uint8"], B: T.Buffer[(outer, inner), "uint8"]):
    for i in T.serial(outer):
        for j in T.serial(inner):
            with T.block("plus_one"):
                with T.block():
                    B[i, j] = A[i, j] + T.uint8(1)


@tvm.testing.requires_hexagon
def test_software_pipeline_with_cache_read(hexagon_launcher):
    sch = tir.Schedule(plus_one_primfunc)
    root = sch.get_block("root")
    plus_one = sch.get_block("plus_one")
    cache_read_block = sch.cache_read(plus_one, 0, "global")

    i, j = sch.get_loops(plus_one)
    sch.compute_at(cache_read_block, i)
    sch.annotate(i, "software_pipeline_stage", [0, 1])
    sch.annotate(i, "software_pipeline_order", [0, 1])
    sch.annotate(i, "software_pipeline_async_stages", [0])

    tvm.lower(sch.mod["main"]).show()

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(sch.mod["main"], target=tvm.target.Target(target_hexagon, host=target_hexagon))

    with hexagon_launcher.start_session() as hexagon_session:
        mod = hexagon_session.load_module(func)
        dev = hexagon_session.device

        a_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype("uint8")
        b_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype("uint8")
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        mod(a, b)
        ref = a_np + 1
        np.testing.assert_equal(b.numpy(), ref)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
