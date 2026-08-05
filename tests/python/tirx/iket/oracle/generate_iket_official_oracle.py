# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
def native_no_payload_kernel():
    """Exercise every no-payload operation in NativeDump mode."""
    cute_iket.mark("mark")
    token = cute_iket.range_start("token")
    cute_iket.range_end(token)
    cute_iket.range_push("stack")
    cute_iket.range_pop()
    cute_iket.mark("slash/name")
    cute_iket.mark("é")
    _native_no_payload_fillers()


@cute.jit
def native_no_payload_launch():
    native_no_payload_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


def _payload_events():
    """Emit the public scalar payload matrix and both range forms."""
    # CUTLASS 4.6.0's public helper currently forwards Python bool as i1 even
    # though the IKET dialect requires an integer payload of at least 8 bits.
    # UI8 is the verified wire representation used by the TIRx bool shim.
    cute_iket.mark("bool", cutlass.Uint8(True))
    cute_iket.mark("i8", cutlass.Int8(-8))
    cute_iket.mark("u8", cutlass.Uint8(8))
    cute_iket.mark("i16", cutlass.Int16(-16))
    cute_iket.mark("u16", cutlass.Uint16(16))
    cute_iket.mark("i32", cutlass.Int32(-32))
    cute_iket.mark("u32", cutlass.Uint32(32))
    cute_iket.mark("i64", cutlass.Int64(-64))
    cute_iket.mark("u64", cutlass.Uint64(64))
    cute_iket.mark("f32", cutlass.Float32(-3.25))
    cute_iket.mark("f64", cutlass.Float64(6.5))
    token = cute_iket.range_start("token_payload", cutlass.Int32(-7))
    cute_iket.range_end(token, cutlass.Int32(9))
    cute_iket.range_push("stack_payload", cutlass.Float32(1.5))
    cute_iket.range_pop()


def _native_no_payload_fillers():
    """Bring the five operation-covering declarations to the Native limit."""
    cute_iket.mark("native_filler00")
    cute_iket.mark("native_filler01")
    cute_iket.mark("native_filler02")
    cute_iket.mark("native_filler03")
    cute_iket.mark("native_filler04")
    cute_iket.mark("native_filler05")
    cute_iket.mark("native_filler06")
    cute_iket.mark("native_filler07")
    cute_iket.mark("native_filler08")
    cute_iket.mark("native_filler09")
    cute_iket.mark("native_filler10")
    cute_iket.mark("native_filler11")
    cute_iket.mark("native_filler12")
    cute_iket.mark("native_filler13")
    cute_iket.mark("native_filler14")
    cute_iket.mark("native_filler15")
    cute_iket.mark("native_filler16")
    cute_iket.mark("native_filler17")
    cute_iket.mark("native_filler18")
    cute_iket.mark("native_filler19")
    cute_iket.mark("native_filler20")
    cute_iket.mark("native_filler21")
    cute_iket.mark("native_filler22")
    cute_iket.mark("native_filler23")
    cute_iket.mark("native_filler24")


def _native_payload_fillers():
    """Bring the thirteen payload declarations to the Native limit."""
    cute_iket.mark("filler00")
    cute_iket.mark("filler01")
    cute_iket.mark("filler02")
    cute_iket.mark("filler03")
    cute_iket.mark("filler04")
    cute_iket.mark("filler05")
    cute_iket.mark("filler06")
    cute_iket.mark("filler07")
    cute_iket.mark("filler08")
    cute_iket.mark("filler09")
    cute_iket.mark("filler10")
    cute_iket.mark("filler11")
    cute_iket.mark("filler12")
    cute_iket.mark("filler13")
    cute_iket.mark("filler14")
    cute_iket.mark("filler15")
    cute_iket.mark("filler16")


@cute.kernel
def native_payload_kernel():
    """Exercise all public payload types in NativeDump mode."""
    _payload_events()
    _native_payload_fillers()


@cute.jit
def native_payload_launch():
    native_payload_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


def _extended_fillers():
    _native_payload_fillers()
    cute_iket.mark("filler17")


@cute.kernel
def extended_no_payload_kernel():
    """Use 31 distinct names to select ExtendedNativeDump."""
    cute_iket.mark("event00")
    cute_iket.mark("event01")
    cute_iket.mark("event02")
    cute_iket.mark("event03")
    cute_iket.mark("event04")
    cute_iket.mark("event05")
    cute_iket.mark("event06")
    cute_iket.mark("event07")
    cute_iket.mark("event08")
    cute_iket.mark("event09")
    cute_iket.mark("event10")
    cute_iket.mark("event11")
    cute_iket.mark("event12")
    cute_iket.mark("event13")
    cute_iket.mark("event14")
    cute_iket.mark("event15")
    cute_iket.mark("event16")
    cute_iket.mark("event17")
    cute_iket.mark("event18")
    cute_iket.mark("event19")
    cute_iket.mark("event20")
    cute_iket.mark("event21")
    cute_iket.mark("event22")
    cute_iket.mark("event23")
    cute_iket.mark("event24")
    cute_iket.mark("event25")
    cute_iket.mark("event26")
    cute_iket.mark("event27")
    cute_iket.mark("event28")
    cute_iket.mark("event29")
    cute_iket.mark("event30")


@cute.jit
def extended_no_payload_launch():
    extended_no_payload_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.kernel
def extended_payload_kernel():
    """Exercise payloads with 31 names in ExtendedNativeDump mode."""
    _payload_events()
    _extended_fillers()


@cute.jit
def extended_payload_launch():
    extended_payload_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


CASES = {
    "native_no_payload": native_no_payload_launch,
    "native_payload": native_payload_launch,
    "extended_no_payload": extended_no_payload_launch,
    "extended_payload": extended_payload_launch,
}


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


def _packaged_nvdisasm() -> Path:
    distribution = metadata.distribution("nvidia-cuda-nvdisasm")
    path = Path(distribution.locate_file("nvidia/cu13/bin/nvdisasm"))
    if not path.is_file():
        raise RuntimeError(f"packaged nvdisasm is missing: {path}")
    return path


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


def _instruction_features(ptx: str, sass: str) -> dict:
    ptx_features = (
        "activemask",
        "elect.sync",
        "globaltimer_lo",
        "st.weak.shared.u32",
        "st.weak.shared.u64",
        "pmevent.mask",
    )
    sass_features = (
        "S2R",
        "ELECT",
        "CS2R",
        "STS",
        "PMTRIG",
    )
    return {
        "ptx": {feature: ptx.count(feature) for feature in ptx_features},
        "sass": {feature: sass.count(feature) for feature in sass_features},
    }


def _compile_case(case_name: str, launch, output_dir: Path, nvdisasm: Path) -> dict:
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    compiled = cute.compile(
        launch,
        options=(f"iket --dump-dir={case_dir} --keep-ptx --keep-cubin --keep-sass"),
    )
    ptx = _artifact_bytes(compiled.artifacts.PTX)
    cubin = _artifact_bytes(compiled.artifacts.CUBIN)
    cubin_path = case_dir / f"{case_name}.cubin"
    cubin_path.write_bytes(cubin)
    sass = subprocess.run(
        [nvdisasm, "-c", cubin_path], check=True, capture_output=True, text=True
    ).stdout
    context = iket.Context(cubin)
    if not context.is_instrumented():
        raise RuntimeError(f"CUTLASS DSL oracle case {case_name} is not IKET-instrumented")

    normalized_metadata = _normalized_metadata(context)
    methods = sorted(
        {
            event["instrument_method"]
            for event in normalized_metadata["events"]
            if event["event_id"] not in (0, 31)
        }
    )
    return {
        "instrument_methods": methods,
        "artifact_sha256": {
            "ptx": _sha256(ptx),
            "cubin": _sha256(cubin),
            "sass": _sha256(sass),
        },
        "instruction_features": _instruction_features(ptx.decode(), sass),
        "metadata": normalized_metadata,
        "metadata_sha256": _sha256(
            json.dumps(normalized_metadata, sort_keys=True, separators=(",", ":"))
        ),
    }


def _wheel_hashes(wheels: list[Path]) -> dict[str, str]:
    if wheels:
        return {wheel.name: _sha256(wheel) for wheel in sorted(wheels)}
    prior_manifest = Path(__file__).with_name("iket_official_cutlass_4_6_0_oracle.json")
    if prior_manifest.is_file():
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("profile", {}).get("cutlass_dsl") == "4.6.0":
            return prior.get("wheels", {})
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cutlass.cuda.initialize_cuda_context()
    nvdisasm = _packaged_nvdisasm()
    cases = {
        name: _compile_case(name, launch, args.output_dir, nvdisasm)
        for name, launch in CASES.items()
    }
    manifest = {
        "schema_version": 3,
        "profile": {
            "cutlass_dsl": metadata.version("nvidia-cutlass-dsl"),
            "driver": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                    "--id=0",
                ]
            ),
            "nvdisasm": _command_output([nvdisasm, "--version"]),
            "nvdisasm_distribution": metadata.version("nvidia-cuda-nvdisasm"),
            "nvrtc_distribution": metadata.version("nvidia-cuda-nvrtc"),
            "compiler_flags": [
                "iket",
                "--dump-dir=<output>/<case>",
                "--keep-ptx",
                "--keep-cubin",
                "--keep-sass",
            ],
        },
        "wheels": _wheel_hashes(args.wheel),
        "cases": cases,
        "abi": {
            "meta_info_bytes": 48,
            "event_attributes_bytes": 60,
            "range_attributes_bytes": 72,
            "sentinel_event_id": 0,
            "range_pop_event_id": 31,
            "native_user_event_ids": [1, 30],
            "extended_user_event_ids": [64, 4095],
            "max_user_declarations": 4032,
            "native_patched_hot_path": [
                "GLOBALTIMERLO",
                "ENCODE_EVENT_ID",
                "STORE_GLOBAL_32",
                "ADD_WRITE_PTR_64_4",
            ],
            "record_layouts": {
                "native_no_payload": {"timestamp_event": [0, 4]},
                "native_payload_32": {"timestamp_event": [0, 4], "payload": [4, 4]},
                "native_payload_64": {"timestamp_event": [0, 4], "payload": [8, 8]},
                "extended_no_payload": {"timestamp_event": [0, 8]},
                "extended_payload_32": {"timestamp_event": [0, 8], "payload": [8, 4]},
                "extended_payload_64": {"timestamp_event": [0, 8], "payload": [8, 8]},
            },
        },
    }
    manifest_path = args.manifest_path or args.output_dir / "iket_official_oracle.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
