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
from tvm.target import Target
from tvm.ir.memory_pools import PoolInfoProperties, WorkspaceMemoryPools, WorkspacePoolInfo


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


def generate_workspace_pools_args(parser):
    """Generates arguments for each Workspace Pools's options"""
    parser.add_argument(
        "--workspace-pools",
        help="""The name of the memory pool
                Example usage: --workspace-pools=flash""",
    )
    parser.add_argument(
        "--workspace-pools-targets",
        help="""The name of the targets specified for the memory pool
                Example usage: --workspace-pools-targets=flash:llvm""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-size-hint-bytes",
        nargs="?",
        help="""The expected size hint to be used by the allocator.
                Example usage: --workspace-pools-size-hint-bytes=flash:8""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-clock-frequency-hz",
        nargs="?",
        help="""The clock frequency that the memory pool runs at in Hz.
                Example usage: --workspace-pools-clock-frequency-hz=flash:70000000""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-read-bandwidth-bytes-per-cycle",
        nargs="?",
        help="""The read bandwidth of the memory pool in bytes/cycle.
                Example usage: --workspace-pools-read-bandwidth-bytes-per-cycle=flash:4""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-write-bandwidth-bytes-per-cycle",
        nargs="?",
        help="""The write bandwidth of the memory pool in bytes/cycle.
                Example usage: --workspace-pools-write-bandwidth-bytes-per-cycle=flash:8""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-read-latency-cycles",
        nargs="?",
        help="""The read latency of the memory pool in cycles.
                Example usage: --workspace-pools-read-latency-cycles=flash:4""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-write-latency-cycles",
        nargs="?",
        help="""The write latency of the memory pool in cycles.
                Example usage: --workspace-pools-write-latency-cycles=flash:8""",
        action="append",
    )
    parser.add_argument(
        "--workspace-pools-target-burst-bytes",
        help="""The burst length of the memory pool in bytes per target.
                Example usage: --workspace-pools-target-burst-bytes=flash:accel:1""",
        action="append",
    )


def _parse_target_burst(attr_str, pool_name):
    if pool_name not in attr_str:
        return {}

    return {target: int(attr_str[pool_name][target]) for target in attr_str[pool_name]}


def _parse_target_string(attr_str, targets, pool_name):
    if attr_str is None:
        raise TVMCException(f'No target specified for Workspace Pool "{pool_name}"')

    target_name = [re.split(",", attr_str)]
    matched_targets = [
        target
        for target in targets
        if any(target.kind.name in target_string_match for target_string_match in target_name[0])
    ]
    if not matched_targets:
        raise TVMCException(f'Workspace Pool "{pool_name}" using undefined Target "{target_name}"')
    return matched_targets


def _split_pools_to_pool_names(attr_str):
    return re.split(",", attr_str) if attr_str else []


def _parse_target_attributes_of_pool_name(attr_str, targets):
    if not targets or attr_str is None:
        return {}

    target_attributes = {}
    for pool_values in attr_str:
        pool_name, target_name, target_value = re.split(":", pool_values)
        if pool_name not in target_attributes:
            target_attributes[pool_name] = {}

        matched_targets = [target for target in targets if target_name == target.kind.name]
        if matched_targets:
            target_attributes[pool_name][matched_targets[0]] = target_value
        else:
            raise TVMCException(
                "The workspace pool target specification "
                "needs to contain a subset of the same TVM "
                "targets as when specifying targets to use."
            )
    return target_attributes


def _parse_attribute_of_pool_name(attr_str):
    return dict(pool.split(":", maxsplit=1) for pool in attr_str) if attr_str else {}


def workspace_pools_recombobulate(parsed, targets, extra_target):
    """Reconstructs the Workspace Pools args and returns a WorkspaceMemoryPool object"""
    WORKSPACE_POOL_PARAMS = [
        "workspace_pools_size_hint_bytes",
        "workspace_pools_targets",
        "workspace_pools_clock_frequency_hz",
        "workspace_pools_read_bandwidth_bytes_per_cycle",
        "workspace_pools_write_bandwidth_bytes_per_cycle",
        "workspace_pools_read_latency_cycles",
        "workspace_pools_write_latency_cycles",
    ]
    WORKSPACE_POOL_TARGET_PARAMS = [
        "workspace_pools_target_burst_bytes",
    ]

    workspace_pools = _split_pools_to_pool_names(parsed.workspace_pools)
    if not workspace_pools:
        return None

    parse_attribute_to_pool_name = {
        workspace_pool_param: _parse_attribute_of_pool_name(getattr(parsed, workspace_pool_param))
        for workspace_pool_param in WORKSPACE_POOL_PARAMS
    }
    parse_target_burst_bytes_to_pool = {
        workspace_pool_param: _parse_target_attributes_of_pool_name(
            getattr(parsed, workspace_pool_param), targets
        )
        for workspace_pool_param in WORKSPACE_POOL_TARGET_PARAMS
    }

    # Load extra targets from CLI
    additional_targets = []

    for t in extra_target:
        additional_targets.append(Target(t["raw"], host=targets[0].host or targets[0]))

    target = targets + additional_targets
    if targets[0].host:
        target.append(targets[0].host)

    return WorkspaceMemoryPools(
        [
            WorkspacePoolInfo(
                pool_name,
                targets=_parse_target_string(
                    parse_attribute_to_pool_name["workspace_pools_targets"].get(pool_name),
                    target,
                    pool_name,
                ),
                pool_info_properties=PoolInfoProperties(
                    size_hint_bytes=int(
                        parse_attribute_to_pool_name["workspace_pools_size_hint_bytes"].get(
                            pool_name, -1
                        )
                    ),
                    clock_frequency_hz=int(
                        parse_attribute_to_pool_name["workspace_pools_clock_frequency_hz"].get(
                            pool_name, -1
                        )
                    ),
                    read_bandwidth_bytes_per_cycle=int(
                        parse_attribute_to_pool_name[
                            "workspace_pools_read_bandwidth_bytes_per_cycle"
                        ].get(pool_name, -1)
                    ),
                    write_bandwidth_bytes_per_cycle=int(
                        parse_attribute_to_pool_name[
                            "workspace_pools_write_bandwidth_bytes_per_cycle"
                        ].get(pool_name, -1)
                    ),
                    read_latency_cycles=int(
                        parse_attribute_to_pool_name["workspace_pools_read_latency_cycles"].get(
                            pool_name, 0
                        )
                    ),
                    write_latency_cycles=int(
                        parse_attribute_to_pool_name["workspace_pools_write_latency_cycles"].get(
                            pool_name, 0
                        )
                    ),
                    target_burst_bytes=_parse_target_burst(
                        parse_target_burst_bytes_to_pool["workspace_pools_target_burst_bytes"],
                        pool_name,
                    ),
                ),
            )
            for pool_name in workspace_pools
        ]
    )
