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
import tvm.relay as relay
from tvm.relay.backend.contrib.uma import uma_available
from tvm.relay.backend.contrib.uma.api import UMAPartitioner
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.testing import mlp, resnet

pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


def test_partition_table():
    partitioner = UMAPartitioner("test_partition")
    assert get_pattern_table("test_partition") is None

    partitioner.register()

    assert get_pattern_table("test_partition") is not None


@pytest.mark.parametrize(
    "workload,backend,merge",
    [
        ("resnet", "dnnl", False),
        ("resnet", "dnnl", True),
        ("mlp", "dnnl", False),
        ("mlp", "dnnl", True),
        ("resnet", "cutlass", False),
        ("resnet", "cutlass", True),
        ("mlp", "cutlass", False),
        ("mlp", "cutlass", True),
    ],
)
def test_existing_pattern_tables(workload, backend, merge):
    """Tests that uma partitioner creates the same partitions than default BYOC partitioning"""
    partitioner = UMAPartitioner(backend, merge)
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

    def partition_default(mod):
        """partitions using default BYOC flow"""

        sequence = [
            relay.transform.MergeComposite(pattern_table),
            relay.transform.AnnotateTarget(backend),
        ]

        if merge:
            sequence.append(relay.transform.MergeCompilerRegions())

        sequence.append(relay.transform.PartitionGraph())
        sequential = tvm.transform.Sequential(sequence)

        return sequential(mod)

    default_partitioned_mod = partition_default(mod)

    assert len(partitioned_mod.functions) == len(default_partitioned_mod.functions)


if __name__ == "__main__":
    tvm.testing.main()
