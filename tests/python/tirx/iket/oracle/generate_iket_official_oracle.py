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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the normalized CUTLASS DSL 4.6.0 IKET oracle manifest.

This helper intentionally writes NVIDIA-generated binary artifacts only to the
requested output directory.  The repository stores the normalized manifest,
not the proprietary artifacts themselves.
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
from enum import IntEnum
from importlib import metadata
from pathlib import Path

import cutlass
import cutlass.cute as cute
import iket
from cutlass.cute.experimental import iket as cute_iket


@cute.kernel
def oracle_kernel():
    """Exercise every no-payload operation supported by the official backend."""
    cute_iket.mark("mark")
    token = cute_iket.range_start("token")
    cute_iket.range_end(token)
    cute_iket.range_push("stack")
    cute_iket.range_pop()
    cute_iket.mark("slash/name")
    cute_iket.mark("é")


@cute.jit
def oracle_launch():
    oracle_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


def _normalize(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    return value


def _sha256(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _artifact_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, str):
        path = Path(value)
        return path.read_bytes() if "\n" not in value and path.is_file() else value.encode()
    return bytes(value)


def _command_output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _normalized_metadata(context):
    events = sorted(_normalize(context.get_all_events()), key=lambda item: item["event_id"])
    ranges = sorted(_normalize(context.get_all_ranges()), key=lambda item: item["range_id"])
    kernels = []
    for kernel in _normalize(context.get_kernels_instrument_info()):
        kernels.append(
            {
                "kernel_name": kernel["kernel_name"],
                "event_sequence": [point["event_attr"] for point in kernel["instrument_points"]],
            }
        )
    return {
        "info": _normalize(context.get_meta_info()),
        "ranges": ranges,
        "events": events,
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    for artifact_name in (
        "unpatched-kernel",
        "patched-kernel",
        "unpatched-disassembly",
        "patched-disassembly",
    ):
        parser.add_argument(f"--{artifact_name}", type=Path)
    args = parser.parse_args()
    patch_artifacts = {
        name: getattr(args, name)
        for name in (
            "unpatched_kernel",
            "patched_kernel",
            "unpatched_disassembly",
            "patched_disassembly",
        )
    }
    if any(patch_artifacts.values()) and not all(patch_artifacts.values()):
        parser.error("all four patched/unpatched artifact paths must be supplied together")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cutlass.cuda.initialize_cuda_context()
    compiled = cute.compile(
        oracle_launch,
        options=(f"iket --dump-dir={args.output_dir} --keep-ptx --keep-cubin --keep-sass"),
    )
    artifacts = {
        name.lower(): _artifact_bytes(getattr(compiled.artifacts, name))
        for name in ("PTX", "CUBIN", "SASS")
    }
    cubin = artifacts["cubin"]
    context = iket.Context(cubin)
    if not context.is_instrumented():
        raise RuntimeError("CUTLASS DSL oracle CUBIN is not IKET-instrumented")

    normalized_metadata = _normalized_metadata(context)
    manifest = {
        "schema_version": 2,
        "profile": {
            "cutlass_dsl": metadata.version("nvidia-cutlass-dsl"),
            "instrument_method": "NativeDump",
            "driver": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                    "--id=0",
                ]
            ),
            "nvdisasm": _command_output(["nvdisasm", "--version"]),
            "compiler_flags": [
                "iket",
                "--dump-dir=<output>",
                "--keep-ptx",
                "--keep-cubin",
                "--keep-sass",
            ],
        },
        "wheels": {wheel.name: _sha256(wheel) for wheel in sorted(args.wheel)},
        "artifact_sha256": {name: _sha256(value) for name, value in artifacts.items()},
        "metadata": normalized_metadata,
        "metadata_sha256": _sha256(
            json.dumps(normalized_metadata, sort_keys=True, separators=(",", ":"))
        ),
        "native_dump_abi": {
            "meta_info_bytes": 48,
            "event_attributes_bytes": 60,
            "range_attributes_bytes": 72,
            "sentinel_event_id": 0,
            "range_pop_event_id": 31,
            "placeholder": [
                "READ_CLUSTER_CTARANK",
                "GLOBALTIMERLO",
                "ENCODE_EVENT_ID",
                "COMPUTE_SHARED_PLACEHOLDER_ADDRESS",
                "STORE_SHARED_WEAK_32",
                "PMEVENT",
            ],
            "patched_hot_path": [
                "GLOBALTIMERLO",
                "ENCODE_EVENT_ID",
                "STORE_GLOBAL_32",
                "ADD_WRITE_PTR_64_4",
            ],
        },
    }
    if all(patch_artifacts.values()):
        manifest["patch_artifact_sha256"] = {
            name: _sha256(path) for name, path in patch_artifacts.items()
        }
    manifest_path = args.output_dir / "iket_official_oracle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
