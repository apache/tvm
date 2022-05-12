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
"""
Functions for processing dynamic workspace pool TVMC args
"""


import logging
import re

from tvm.driver.tvmc import TVMCException
from tvm.ir.memory_pools import PoolInfo, WorkspaceMemoryPools


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


def generate_workspace_pools_args(parser):
    parser.add_argument(
        "--workspace-pools",
        help="""The name of the memory pool
                Example usage: --workspace-pools=flash""",
    )
    parser.add_argument(
        "--workspace-pools-size-hint-bytes",
        nargs="?",
        help="""The expected size hint to be used by the allocator.
                Example usage: --workspace-pools-size-hint-bytes=flash:8""",
    )
    parser.add_argument(
        "--workspace-pools-target-access",
        help="""A dictionary where keys describe which targets could
                access the pool where value could take the values :
                a) "rw" : read-write access
                b) "ro" : write-only access
                Example usage: --workspace-pools-target-access=flash:accel:ro""",
    )
    parser.add_argument(
        "--workspace-pools-clock-frequency-hz",
        nargs="?",
        help="""The clock frequency that the memory pool runs at in Hz.
                Example usage: --workspace-pools-clock-frequency-hz=flash:70000000""",
    )
    parser.add_argument(
        "--workspace-pools-read-bandwidth-bytes-per-cycle",
        nargs="?",
        help="""The read bandwidth of the memory pool in bytes/cycle.
                Example usage: --workspace-pools-read-bandwidth-bytes-per-cycle=flash:4""",
    )
    parser.add_argument(
        "--workspace-pools-write-bandwidth-bytes-per-cycle",
        nargs="?",
        help="""The write bandwidth of the memory pool in bytes/cycle.
                Example usage: --workspace-pools-write-bandwidth-bytes-per-cycle=flash:8""",
    )
    parser.add_argument(
        "--workspace-pools-read-latency-cycles",
        nargs="?",
        help="""The read latency of the memory pool in cycles.
                Example usage: --workspace-pools-read-latency-cycles=flash:4""",
    )
    parser.add_argument(
        "--workspace-pools-write-latency-cycles",
        nargs="?",
        help="""The write latency of the memory pool in cycles.
                Example usage: --workspace-pools-write-latency-cycles=flash:8""",
    )
    parser.add_argument(
        "--workspace-pools-target-burst-bytes",
        help="""The burst length of the memory pool in bytes per target.
                Example usage: --workspace-pools-target-burst-bytes=flash:accel:1""",
    )


def _parse_target_burst(attr, pool_name):
    if pool_name not in attr:
        return {}

    return {target: int(attr[pool_name][target]) for target in attr[pool_name]}


def _parse_target_access(attr, pool_name):
    try:
        return attr[pool_name]
    except:
        raise TVMCException(f'Workspace Pool "{pool_name}" using undefined Target.')


def _split_pools(attr):
    return re.split(",", attr) if attr else []


def _target_attributes_to_pools(attr, targets):
    if not targets or attr is None:
        return {}

    target_attributes = {}
    for pool_values in re.split(",", attr):
        pool_name, target_name, target_value = re.split(":", pool_values)
        if pool_name not in target_attributes:
            target_attributes[pool_name] = {}

        matched_targets = [target for target in targets if target_name == target.kind.name]
        if matched_targets:
            target_attributes[pool_name][matched_targets[0]] = target_value
        else:
            raise TVMCException(
                """The workspace pool target specification for target
                  access needs to be the same TVM target as when specifying targets to use."""
            )

    return target_attributes


def _attribute_to_pools(attr):
    return dict(pool.split(":", maxsplit=1) for pool in re.split(",", attr)) if attr else {}


def workspace_pools_recombobulate(parsed, targets):
    WORKSPACE_POOL_PARAMS = [
        "workspace_pools_size_hint_bytes",
        "workspace_pools_target_access",
        "workspace_pools_clock_frequency_hz",
        "workspace_pools_read_bandwidth_bytes_per_cycle",
        "workspace_pools_write_bandwidth_bytes_per_cycle",
        "workspace_pools_read_latency_cycles",
        "workspace_pools_write_latency_cycles",
        "workspace_pools_target_burst_bytes",
    ]
    WORKSPACE_POOL_TARGET_PARAMS = [
        "workspace_pools_target_access",
        "workspace_pools_target_burst_bytes",
    ]

    workspace_pools = _split_pools(parsed.workspace_pools)
    pool_to_attributes = {
        workspace_pool_param: _attribute_to_pools(getattr(parsed, workspace_pool_param))
        for workspace_pool_param in WORKSPACE_POOL_PARAMS
    }
    pool_to_target_attributes = {
        workspace_pool_param: _target_attributes_to_pools(
            getattr(parsed, workspace_pool_param), targets
        )
        for workspace_pool_param in WORKSPACE_POOL_TARGET_PARAMS
    }

    return WorkspaceMemoryPools(
        [
            PoolInfo(
                pool_name,
                target_access=_parse_target_access(
                    pool_to_target_attributes["workspace_pools_target_access"], pool_name
                ),
                size_hint_bytes=int(
                    pool_to_attributes["workspace_pools_size_hint_bytes"].get(pool_name, -1)
                ),
                clock_frequency_hz=int(
                    pool_to_attributes["workspace_pools_clock_frequency_hz"].get(pool_name, -1)
                ),
                read_bandwidth_bytes_per_cycle=int(
                    pool_to_attributes["workspace_pools_read_bandwidth_bytes_per_cycle"].get(
                        pool_name, -1
                    )
                ),
                write_bandwidth_bytes_per_cycle=int(
                    pool_to_attributes["workspace_pools_write_bandwidth_bytes_per_cycle"].get(
                        pool_name, -1
                    )
                ),
                read_latency_cycles=int(
                    pool_to_attributes["workspace_pools_read_latency_cycles"].get(pool_name, 0)
                ),
                write_latency_cycles=int(
                    pool_to_attributes["workspace_pools_write_latency_cycles"].get(pool_name, 0)
                ),
                target_burst_bytes=_parse_target_burst(
                    pool_to_target_attributes["workspace_pools_target_burst_bytes"], pool_name
                ),
            )
            for pool_name in workspace_pools
        ]
    )
