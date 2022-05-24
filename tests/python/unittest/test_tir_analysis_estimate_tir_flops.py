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
import tvm.testing
from tvm.ir import IRModule
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.tir.analysis import estimate_tir_flops


@pytest.mark.parametrize(
    "workload, flops",
    [
        ("C1D", 6291456),
        ("C2D", 236027904),
        ("C3D", 13217562624),
        ("CAP", 75497472),
        ("DEP", 7225344),
        ("DIL", 223552896),
        ("GMM", 4194304),
        ("GRP", 28901376),
        ("T2D", 268435456),
        ("C2d-BN-RELU", 239239168),
        ("TBG", 25165824),
        ("NRM", 131072),
        ("SFM", 262144),
    ],
)
def test_te_workload(workload, flops):
    te_workload = create_te_workload(workload, 0)
    mod = IRModule({"main": te_workload})
    assert float(flops) == estimate_tir_flops(mod)


if __name__ == "__main__":
    tvm.testing.main()
