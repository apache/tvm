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

from tvm.relay.backend.contrib.uma.api import UMAPartitioner
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.testing import resnet, mlp
from tvm.relay.backend.contrib.uma import uma_available

pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


def test_partition_table():
    partitioner = UMAPartitioner("test_partition")
    assert get_pattern_table("test_partition") is None

    partitioner.register()

    assert get_pattern_table("test_partition") is not None


@pytest.mark.parametrize(
    "workload,backend,merge,expected_partitions",
    [
        ("resnet", "dnnl", False, 17),
        ("resnet", "dnnl", True, 17),
        ("mlp", "dnnl", False, 1),
        ("resnet", "cutlass", False, 2),
        ("resnet", "cutlass", True, 2),
        ("mlp", "cutlass", False, 4),
        ("mlp", "cutlass", True, 2),
    ],
)
def test_existing_pattern_tables(workload, backend, merge, expected_partitions):
    partitioner = UMAPartitioner(backend + "_uma", merge)
    pattern_table = get_pattern_table(backend)

    for entry in pattern_table:
        partitioner.add_pattern(*entry)

    if workload == "resnet":
        net = resnet.get_net(1, 10)
    elif workload == "mlp":
        net = mlp.get_net(1, 10)
    else:
        assert False, f"don't know how to find workload for {workload}"

    mod = tvm.ir.IRModule()
    mod["main"] = net

    partitioner.register()
    partitioned_mod = partitioner.partition(mod)
    print(partitioned_mod)

    assert len(partitioned_mod.functions) == expected_partitions


if __name__ == "__main__":
    tvm.testing.main()
