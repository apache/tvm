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
"""The CUDA kernel configuration registry. This module has no TVM dependencies.

``generate.py`` derives the public configuration classes, native field/attribute
encoders, and reference documentation from these declarations. Version numbers
denote the oldest toolkit supported by our encoder for the corresponding field.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    doc: str
    default: str = "None"
    members: tuple = ()
    enums: tuple = ()
    driver_id: str = ""
    runtime_id: str = ""
    native_member: str = ""
    version: int = 12000


CLUSTER_POLICY = (("default", 0), ("spread", 1), ("load_balancing", 2))
MEM_DOMAIN = (("default", 0), ("remote", 1))
ACCESS_PROPERTY = (("normal", 0), ("streaming", 1), ("persisting", 2))
PORTABLE_CLUSTER = (("default", 0), ("require_portable", 1), ("allow_non_portable", 2))
SHARED_MEMORY = (
    ("default", 0),
    ("require_portable", 1),
    ("allow_non_portable", 2),
    ("allow_oversized", 3),
    ("prefer_oversized", 4),
)

COMPOSITES = {
    "MemSyncDomainMap": (
        Field("default", "int", "Physical domain for the default logical domain.", "0"),
        Field("remote", "int", "Physical domain for the remote logical domain.", "1"),
    ),
    "AccessPolicyWindow": (
        Field("base_ptr", "handle", "Device pointer to the access-policy window.", "required"),
        Field("num_bytes", "int", "Window size in bytes.", "required"),
        Field("hit_ratio", "float", "Fraction of accesses receiving the hit policy.", "1.0"),
        Field("hit_prop", "enum", "Cache policy for hits.", "'persisting'", enums=ACCESS_PROPERTY),
        Field("miss_prop", "enum", "Cache policy for misses.", "'normal'", enums=ACCESS_PROPERTY),
    ),
    "ProgrammaticEvent": (
        Field("event", "handle", "Caller-owned, timing-disabled CUDA event.", "required"),
        Field("flags", "int", "Event record flags; external-event recording is unsupported.", "0"),
        Field("trigger_at_block_start", "bool", "Trigger when each block starts.", "False"),
    ),
    "LaunchCompletionEvent": (
        Field("event", "handle", "Caller-owned, timing-disabled CUDA event.", "required"),
        Field("flags", "int", "Event record flags; external-event recording is unsupported.", "0"),
    ),
}

LAUNCH_FIELDS = (
    Field(
        "grid",
        "dim3",
        "Grid dimensions in CTAs; an integer or one to three dimensions.",
        "required",
    ),
    Field("block", "dim3", "Threads per CTA; an integer or one to three dimensions.", "required"),
    Field(
        "cluster",
        "dim3",
        "CTAs per cluster. None differs from an explicit unit cluster.",
        driver_id="CLUSTER_DIMENSION",
        runtime_id="ClusterDimension",
        native_member="clusterDim",
    ),
    Field(
        "preferred_cluster",
        "dim3",
        "Preferred substitute cluster dimensions.",
        driver_id="PREFERRED_CLUSTER_DIMENSION",
        runtime_id="PreferredClusterDimension",
        native_member="preferredClusterDim",
        version=12080,
    ),
    Field(
        "dynamic_smem_bytes",
        "int",
        "Dynamic shared memory bytes; None infers allocation requirements.",
    ),
    Field("stream", "handle", "CUDA stream; None uses the current tvm-ffi stream."),
    Field(
        "cooperative",
        "bool",
        "Request a cooperative kernel launch.",
        driver_id="COOPERATIVE",
        runtime_id="Cooperative",
        native_member="cooperative",
    ),
    Field(
        "programmatic_stream_serialization",
        "bool",
        "Enable programmatic dependent launch.",
        driver_id="PROGRAMMATIC_STREAM_SERIALIZATION",
        runtime_id="ProgrammaticStreamSerialization",
        native_member="programmaticStreamSerializationAllowed",
    ),
    Field(
        "cluster_scheduling_policy",
        "enum",
        "Cluster scheduling preference.",
        enums=CLUSTER_POLICY,
        driver_id="CLUSTER_SCHEDULING_POLICY_PREFERENCE",
        runtime_id="ClusterSchedulingPolicyPreference",
        native_member="clusterSchedulingPolicyPreference",
    ),
    Field(
        "priority",
        "int",
        "Launch priority; CUDA may clamp it to the supported range.",
        driver_id="PRIORITY",
        runtime_id="Priority",
        native_member="priority",
    ),
    Field(
        "mem_sync_domain",
        "enum",
        "Logical memory synchronization domain.",
        enums=MEM_DOMAIN,
        driver_id="MEM_SYNC_DOMAIN",
        runtime_id="MemSyncDomain",
        native_member="memSyncDomain",
    ),
    Field(
        "mem_sync_domain_map",
        "MemSyncDomainMap",
        "Mapping of logical to physical memory domains.",
        members=(("default", "default_"), ("remote", "remote")),
        driver_id="MEM_SYNC_DOMAIN_MAP",
        runtime_id="MemSyncDomainMap",
        native_member="memSyncDomainMap",
    ),
    Field(
        "access_policy_window",
        "AccessPolicyWindow",
        "Per-launch L2 access policy window.",
        members=(
            ("base_ptr", "base_ptr"),
            ("num_bytes", "num_bytes"),
            ("hit_ratio", "hitRatio"),
            ("hit_prop", "hitProp"),
            ("miss_prop", "missProp"),
        ),
        driver_id="ACCESS_POLICY_WINDOW",
        runtime_id="AccessPolicyWindow",
        native_member="accessPolicyWindow",
    ),
    Field(
        "preferred_shared_memory_carveout",
        "int",
        "Preferred shared-memory carveout percentage, 0 through 100.",
        driver_id="PREFERRED_SHARED_MEMORY_CARVEOUT",
        runtime_id="PreferredSharedMemoryCarveout",
        native_member="sharedMemCarveout",
        version=12080,
    ),
    Field(
        "nvlink_util_centric_scheduling",
        "bool",
        "Best-effort NVLink utilization scheduling hint.",
        driver_id="NVLINK_UTIL_CENTRIC_SCHEDULING",
        runtime_id="NvlinkUtilCentricScheduling",
        native_member="nvlinkUtilCentricScheduling",
        version=13020,
    ),
    Field(
        "portable_cluster_size_mode",
        "enum",
        "Override cluster portability policy for this launch.",
        enums=PORTABLE_CLUSTER,
        driver_id="PORTABLE_CLUSTER_SIZE_MODE",
        runtime_id="PortableClusterSizeMode",
        native_member="portableClusterSizeMode",
        version=13020,
    ),
    Field(
        "shared_memory_mode",
        "enum",
        "Override shared-memory resource mode; oversized modes require CUDA 13.4.",
        enums=SHARED_MEMORY,
        driver_id="SHARED_MEMORY_MODE",
        runtime_id="SharedMemoryMode",
        native_member="sharedMemoryMode",
        version=13020,
    ),
    Field(
        "programmatic_event",
        "ProgrammaticEvent",
        "Record a programmatic dependency event.",
        members=(
            ("event", "event"),
            ("flags", "flags"),
            ("trigger_at_block_start", "triggerAtBlockStart"),
        ),
        driver_id="PROGRAMMATIC_EVENT",
        runtime_id="ProgrammaticEvent",
        native_member="programmaticEvent",
    ),
    Field(
        "launch_completion_event",
        "LaunchCompletionEvent",
        "Record an event associated with blocks beginning execution.",
        members=(("event", "event"), ("flags", "flags")),
        driver_id="LAUNCH_COMPLETION_EVENT",
        runtime_id="LaunchCompletionEvent",
        native_member="launchCompletionEvent",
        version=12040,
    ),
)

KERNEL_ATTR_FIELDS = (
    Field(
        "min_blocks_per_sm", "int", "Second launch-bounds operand: minimum resident CTAs per SM."
    ),
    Field(
        "max_blocks_per_cluster", "int", "Third launch-bounds operand; requires min_blocks_per_sm."
    ),
    Field(
        "max_registers_per_thread",
        "int",
        "Emit __maxnreg__; incompatible with explicit launch bounds and required block size.",
    ),
    Field(
        "required_block_size",
        "bool",
        "Fix block and cluster dimensions at compilation using __block_size__.",
        "False",
        version=13000,
    ),
)


def leaves():
    """Yield (field, suffix, scalar kind) in the stable packed-operand order."""
    for field in LAUNCH_FIELDS:
        if field.kind == "dim3":
            for axis in "xyz":
                yield field, axis, "int"
        elif field.kind in COMPOSITES:
            for member in COMPOSITES[field.kind]:
                yield field, member.name, member.kind
        else:
            yield field, "", field.kind
