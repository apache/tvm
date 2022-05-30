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
import argparse

from tvm.ir.memory_pools import PoolInfo
from tvm.driver.tvmc.workspace_pools import (
    generate_workspace_pools_args,
    workspace_pools_recombobulate,
)
from tvm.target import Target
from tvm.driver.tvmc import TVMCException


def test_workspace_pools_argparse():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, unparsed = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-size-hint-bytes=sram:400",
            "--workspace-pools-target-access=sram:c:rw",
            "--workspace-pools-clock-frequency-hz=sram:500",
            "--workspace-pools-read-bandwidth-bytes-per-cycle=sram:200",
            "--workspace-pools-write-bandwidth-bytes-per-cycle=sram:100",
            "--workspace-pools-read-latency-cycles=sram:50",
            "--workspace-pools-write-latency-cycles=sram:9001",
            "--workspace-pools-target-burst-bytes=sram:c:2",
            "--workspace-pools-is-internal=sram:0",
        ]
    )

    assert parsed.workspace_pools == "sram"
    assert parsed.workspace_pools_size_hint_bytes == "sram:400"
    assert parsed.workspace_pools_target_access == "sram:c:rw"
    assert parsed.workspace_pools_clock_frequency_hz == "sram:500"
    assert parsed.workspace_pools_read_bandwidth_bytes_per_cycle == "sram:200"
    assert parsed.workspace_pools_write_bandwidth_bytes_per_cycle == "sram:100"
    assert parsed.workspace_pools_read_latency_cycles == "sram:50"
    assert parsed.workspace_pools_write_latency_cycles == "sram:9001"
    assert parsed.workspace_pools_target_burst_bytes == "sram:c:2"

    assert unparsed == ["--workspace-pools-is-internal=sram:0"]


def test_workspace_pools_recombobulate():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-target-access=sram:llvm:ro",
            "--workspace-pools-size-hint-bytes=sram:400",
            "--workspace-pools-clock-frequency-hz=sram:500",
        ]
    )

    targets = [Target("llvm")]
    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 1
    assert memory_pools.pools[0].pool_name == "sram"
    assert memory_pools.pools[0].size_hint_bytes == 400
    assert memory_pools.pools[0].clock_frequency_hz == 500


def test_workspace_pools_defaults():
    parser = argparse.ArgumentParser()
    targets = [Target("llvm")]
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-target-access=sram:llvm:4",
        ]
    )

    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 1
    assert memory_pools.pools[0].pool_name == "sram"
    assert memory_pools.pools[0].size_hint_bytes == -1
    assert len(memory_pools.pools[0].target_access) == 1
    assert memory_pools.pools[0].clock_frequency_hz == -1
    assert memory_pools.pools[0].read_bandwidth_bytes_per_cycle == -1
    assert memory_pools.pools[0].write_bandwidth_bytes_per_cycle == -1
    assert memory_pools.pools[0].read_latency_cycles == 0
    assert memory_pools.pools[0].write_latency_cycles == 0
    assert len(memory_pools.pools[0].target_burst_bytes) == 0


def test_workspace_pools_recombobulate_multi_fields():
    parser = argparse.ArgumentParser()
    targets = [Target("c")]
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-size-hint-bytes=sram:400",
            "--workspace-pools-target-access=sram:c:rw",
            "--workspace-pools-clock-frequency-hz=sram:500",
            "--workspace-pools-read-bandwidth-bytes-per-cycle=sram:200",
            "--workspace-pools-write-bandwidth-bytes-per-cycle=sram:100",
            "--workspace-pools-read-latency-cycles=sram:50",
            "--workspace-pools-write-latency-cycles=sram:9001",
            "--workspace-pools-target-burst-bytes=sram:c:2",
        ]
    )

    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 1
    assert memory_pools.pools[0].pool_name == "sram"
    assert memory_pools.pools[0].size_hint_bytes == 400
    assert len(memory_pools.pools[0].target_access) == 1
    assert memory_pools.pools[0].target_access[targets[0]] == PoolInfo.READ_WRITE_ACCESS
    assert memory_pools.pools[0].clock_frequency_hz == 500
    assert memory_pools.pools[0].read_bandwidth_bytes_per_cycle == 200
    assert memory_pools.pools[0].write_bandwidth_bytes_per_cycle == 100
    assert memory_pools.pools[0].read_latency_cycles == 50
    assert memory_pools.pools[0].write_latency_cycles == 9001
    assert len(memory_pools.pools[0].target_burst_bytes) == 1
    assert memory_pools.pools[0].target_burst_bytes[targets[0]] == 2


def test_workspace_pools_recombobulate_multi_fields_variant():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=flash",
            "--workspace-pools-target-access=flash:c:ro",
            "--workspace-pools-size-hint-bytes=flash:2048",
            "--workspace-pools-clock-frequency-hz=flash:2000000",
            "--workspace-pools-read-bandwidth-bytes-per-cycle=flash:4",
            "--workspace-pools-write-bandwidth-bytes-per-cycle=flash:1",
            "--workspace-pools-read-latency-cycles=flash:2000",
            "--workspace-pools-write-latency-cycles=flash:1000",
            "--workspace-pools-target-burst-bytes=flash:c:4",
        ]
    )

    targets = [Target("c")]
    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 1
    assert memory_pools.pools[0].pool_name == "flash"
    assert memory_pools.pools[0].size_hint_bytes == 2048
    assert len(memory_pools.pools[0].target_access) == 1
    assert memory_pools.pools[0].target_access[targets[0]] == PoolInfo.READ_ONLY_ACCESS
    assert memory_pools.pools[0].clock_frequency_hz == 2000000
    assert memory_pools.pools[0].read_bandwidth_bytes_per_cycle == 4
    assert memory_pools.pools[0].write_bandwidth_bytes_per_cycle == 1
    assert memory_pools.pools[0].read_latency_cycles == 2000
    assert memory_pools.pools[0].write_latency_cycles == 1000
    assert len(memory_pools.pools[0].target_burst_bytes) == 1
    assert memory_pools.pools[0].target_burst_bytes[targets[0]] == 4


def test_workspace_pools_recombobulate_multi_fields_multi_pools():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram,flash",
            "--workspace-pools-size-hint-bytes=sram:1024,flash:2048",
            "--workspace-pools-target-access=sram:c:rw,flash:c:ro",
            "--workspace-pools-clock-frequency-hz=sram:4000000,flash:2000000",
            "--workspace-pools-read-bandwidth-bytes-per-cycle=sram:8,flash:4",
            "--workspace-pools-write-bandwidth-bytes-per-cycle=sram:4,flash:1",
            "--workspace-pools-read-latency-cycles=sram:250,flash:2000",
            "--workspace-pools-write-latency-cycles=sram:500,flash:1000",
            "--workspace-pools-target-burst-bytes=sram:c:8,flash:c:4",
        ]
    )

    targets = [Target("c")]
    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 2

    assert memory_pools.pools[0].pool_name == "sram"
    assert memory_pools.pools[0].size_hint_bytes == 1024
    assert len(memory_pools.pools[0].target_access) == 1
    assert memory_pools.pools[0].target_access[targets[0]] == PoolInfo.READ_WRITE_ACCESS
    assert memory_pools.pools[0].clock_frequency_hz == 4000000
    assert memory_pools.pools[0].read_bandwidth_bytes_per_cycle == 8
    assert memory_pools.pools[0].write_bandwidth_bytes_per_cycle == 4
    assert memory_pools.pools[0].read_latency_cycles == 250
    assert memory_pools.pools[0].write_latency_cycles == 500
    assert len(memory_pools.pools[0].target_burst_bytes) == 1
    assert memory_pools.pools[0].target_burst_bytes[targets[0]] == 8

    assert memory_pools.pools[1].pool_name == "flash"
    assert memory_pools.pools[1].size_hint_bytes == 2048
    assert len(memory_pools.pools[1].target_access) == 1
    assert memory_pools.pools[1].target_access[targets[0]] == PoolInfo.READ_ONLY_ACCESS
    assert memory_pools.pools[1].clock_frequency_hz == 2000000
    assert memory_pools.pools[1].read_bandwidth_bytes_per_cycle == 4
    assert memory_pools.pools[1].write_bandwidth_bytes_per_cycle == 1
    assert memory_pools.pools[1].read_latency_cycles == 2000
    assert memory_pools.pools[1].write_latency_cycles == 1000
    assert len(memory_pools.pools[1].target_burst_bytes) == 1
    assert memory_pools.pools[1].target_burst_bytes[targets[0]] == 4


def test_workspace_pools_recombobulate_multi_fields_ordering():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram,flash",
            "--workspace-pools-size-hint-bytes=flash:2048,sram:1024",
            "--workspace-pools-target-access=flash:c:ro,sram:c:rw",
            "--workspace-pools-clock-frequency-hz=sram:4000000,flash:2000000",
            "--workspace-pools-read-bandwidth-bytes-per-cycle=sram:8,flash:4",
            "--workspace-pools-write-bandwidth-bytes-per-cycle=sram:4,flash:1",
            "--workspace-pools-read-latency-cycles=sram:250,flash:2000",
            "--workspace-pools-write-latency-cycles=flash:1000,sram:500",
            "--workspace-pools-target-burst-bytes=sram:c:8,flash:c:4",
        ]
    )

    targets = [Target("c")]
    memory_pools = workspace_pools_recombobulate(parsed, targets)
    assert len(memory_pools.pools) == 2

    assert memory_pools.pools[0].pool_name == "sram"
    assert memory_pools.pools[0].size_hint_bytes == 1024
    assert memory_pools.pools[0].write_latency_cycles == 500

    assert memory_pools.pools[1].pool_name == "flash"
    assert memory_pools.pools[1].size_hint_bytes == 2048
    assert memory_pools.pools[1].write_latency_cycles == 1000


def test_workspace_pools_recombobulate_multi_target():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-target-access=sram:c:rw,sram:llvm:ro",
            "--workspace-pools-target-burst-bytes=sram:c:8,sram:llvm:4",
        ]
    )

    c_target = Target("c")
    llvm_target = Target("llvm")

    targets = [c_target, llvm_target]
    memory_pools = workspace_pools_recombobulate(parsed, targets)

    assert len(memory_pools.pools) == 1

    assert len(memory_pools.pools[0].target_access) == 2
    assert memory_pools.pools[0].target_access[c_target] == PoolInfo.READ_WRITE_ACCESS
    assert memory_pools.pools[0].target_access[llvm_target] == PoolInfo.READ_ONLY_ACCESS
    assert len(memory_pools.pools[0].target_burst_bytes) == 2
    assert memory_pools.pools[0].target_burst_bytes[c_target] == 8
    assert memory_pools.pools[0].target_burst_bytes[llvm_target] == 4


def test_workspace_pools_recombobulate_no_target_burst_bytes():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-target-access=sram:c:rw",
        ]
    )

    c_target = Target("c")
    targets = [c_target]

    workspace_pools_recombobulate(parsed, targets)


def test_workspace_pools_recombobulate_missing_target():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
        ]
    )

    c_target = Target("llvm")
    targets = [c_target]

    with pytest.raises(TVMCException):
        workspace_pools_recombobulate(parsed, targets)


def test_workspace_pools_recombobulate_multi_target_multi_pool():
    parser = argparse.ArgumentParser()
    generate_workspace_pools_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--workspace-pools=sram",
            "--workspace-pools-target-access=sram:c:rw,sram:llvm:ro",
            "--workspace-pools-target-burst-bytes=sram:c:8,sram:llvm:4",
        ]
    )

    c_target = Target("c")
    llvm_target = Target("llvm")

    targets = [c_target, llvm_target]
    memory_pools = workspace_pools_recombobulate(parsed, targets)

    assert len(memory_pools.pools) == 1

    assert len(memory_pools.pools[0].target_access) == 2
    assert memory_pools.pools[0].target_access[c_target] == PoolInfo.READ_WRITE_ACCESS
    assert memory_pools.pools[0].target_access[llvm_target] == PoolInfo.READ_ONLY_ACCESS
    assert len(memory_pools.pools[0].target_burst_bytes) == 2
    assert memory_pools.pools[0].target_burst_bytes[c_target] == 8
    assert memory_pools.pools[0].target_burst_bytes[llvm_target] == 4
