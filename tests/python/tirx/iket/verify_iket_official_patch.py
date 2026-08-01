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
"""Verify the locked run-iket NativeDump patch from retained trace artifacts.

Run ``run-iket`` with ``--log-level trace --keep`` so its per-kernel patched
and unpatched images are retained.  This helper reconstructs a disassemblable
patched CUBIN outside the repository and emits only a normalized JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

INSTRUCTION = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;")
PATCH_OFFSET = re.compile(
    r"Patching SASS => name: (?P<kernel>.*?), .*?driver_patch_offset: (?P<offset>\d+)"
)
STORE_REGISTER = re.compile(r"\bSTS\s+\[[^]]+\],\s+(R\d+)$")
GLOBAL_STORE_REGISTER = re.compile(r"\bSTG\.E\s+\[RZ\.U32\+UR0\],\s+(R\d+)$")
POINTER_INCREMENT = re.compile(r"^UIADD3\.64\s+UR0,\s+UPT,\s+UPT,\s+UR0,\s+0x4,\s+URZ$")
NORMALIZED_SIGNATURE = [
    "GLOBALTIMERLO",
    "ENCODE_EVENT_ID",
    "STORE_GLOBAL_32",
    "ADD_WRITE_PTR_64_4",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _single(paths, description: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"expected one {description}, found {len(paths)}")
    return paths[0]


def _discover_artifacts(run_dir: Path, kernel: str | None):
    unpatched_paths = list(run_dir.glob("iket/pid_*/unpatched_*"))
    if kernel is not None:
        unpatched_paths = [path for path in unpatched_paths if path.name == f"unpatched_{kernel}"]
    unpatched = _single(unpatched_paths, "unpatched kernel image")
    kernel = unpatched.name.removeprefix("unpatched_")
    patched = unpatched.with_name(kernel)
    if not patched.is_file():
        raise RuntimeError("patched kernel image is missing; rerun with --log-level trace --keep")
    injection_log = unpatched.with_name("smodel.injection.log")
    offsets = [
        int(match.group("offset"))
        for match in PATCH_OFFSET.finditer(injection_log.read_text(encoding="utf-8"))
        if match.group("kernel") == kernel
    ]
    if len(offsets) != 1:
        raise RuntimeError(f"expected one driver patch offset for {kernel}, found {offsets}")

    unpatched_bytes = unpatched.read_bytes()
    probe = unpatched_bytes[offsets[0] : offsets[0] + 256]
    matching_cubins = []
    for cubin in run_dir.glob("tracker/pid_*/module_*.cubin"):
        data = cubin.read_bytes()
        position = data.find(probe)
        if position >= 0 and data.find(probe, position + 1) < 0:
            matching_cubins.append((cubin, position))
    if len(matching_cubins) != 1:
        raise RuntimeError(
            f"expected one tracker CUBIN containing {kernel}, found {len(matching_cubins)}"
        )
    cubin, cubin_offset = matching_cubins[0]
    return kernel, cubin, unpatched, patched, offsets[0], cubin_offset


def _disassemble(nvdisasm: str, cubin: Path) -> str:
    return subprocess.run(
        [nvdisasm, "-c", str(cubin)], check=True, capture_output=True, text=True
    ).stdout


def _function_instructions(disassembly: str, kernel: str) -> dict[int, str]:
    marker = f"\n.text.{kernel}:\n"
    if marker not in disassembly:
        raise RuntimeError(f"nvdisasm output does not contain .text.{kernel}")
    function = disassembly.split(marker, 1)[1].split("\n//---------------------", 1)[0]
    return {int(address, 16): text.strip() for address, text in INSTRUCTION.findall(function)}


def _without_predicate(instruction: str) -> str:
    return re.sub(r"^@!?[A-Z0-9]+\s+", "", instruction)


def _verify_patch(unpatched_sass: str, patched_sass: str, kernel: str) -> int:
    before = _function_instructions(unpatched_sass, kernel)
    after = _function_instructions(patched_sass, kernel)
    if before.keys() != after.keys():
        raise RuntimeError("patched kernel changed the instruction address layout")
    changed = [address for address in before if before[address] != after[address]]
    if not changed or len(changed) % 2:
        raise RuntimeError(f"unexpected changed instruction count: {len(changed)}")

    addresses = sorted(before)
    address_index = {address: index for index, address in enumerate(addresses)}
    sites = []
    for index in range(0, len(changed), 2):
        store_address, increment_address = changed[index : index + 2]
        if increment_address not in (store_address + 16, store_address + 32):
            raise RuntimeError(
                f"patch changes are not a NativeDump store/increment pair at {store_address:#x}"
            )
        if increment_address == store_address + 32:
            middle = store_address + 16
            if before[middle] != after[middle]:
                raise RuntimeError(f"interleaved instruction changed at {middle:#x}")
        old_store = _without_predicate(before[store_address])
        new_store = _without_predicate(after[store_address])
        old_register = STORE_REGISTER.search(old_store)
        new_register = GLOBAL_STORE_REGISTER.search(new_store)
        if old_register is None or new_register is None:
            raise RuntimeError(f"unexpected store rewrite at {store_address:#x}")
        if old_register.group(1) != new_register.group(1):
            raise RuntimeError(f"store source register changed at {store_address:#x}")
        if not _without_predicate(before[increment_address]).startswith("PMTRIG "):
            raise RuntimeError(f"expected PMTRIG placeholder at {increment_address:#x}")
        if POINTER_INCREMENT.fullmatch(_without_predicate(after[increment_address])) is None:
            raise RuntimeError(f"unexpected pointer increment at {increment_address:#x}")

        position = address_index[store_address]
        prefix = [after[address] for address in addresses[max(0, position - 12) : position]]
        if not any("SR_GLOBALTIMERLO" in instruction for instruction in prefix):
            raise RuntimeError(f"event site lacks GLOBALTIMERLO at {store_address:#x}")
        if not any("LOP3.LUT" in instruction for instruction in prefix):
            raise RuntimeError(f"event site lacks event-id encoding at {store_address:#x}")
        sites.append(store_address)

    placeholder_count = sum("PMTRIG " in instruction for instruction in before.values())
    if placeholder_count != len(sites):
        raise RuntimeError(
            f"patched {len(sites)} of {placeholder_count} NativeDump placeholder sites"
        )
    if any("PMTRIG " in instruction for instruction in after.values()):
        raise RuntimeError("patched kernel still contains PMTRIG placeholders")
    return len(sites)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kernel")
    parser.add_argument("--nvdisasm", default=shutil.which("nvdisasm"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.nvdisasm is None:
        parser.error("nvdisasm is required")

    kernel, cubin, unpatched, patched, patch_offset, cubin_offset = _discover_artifacts(
        args.run_dir, args.kernel
    )
    unpatched_bytes = unpatched.read_bytes()
    patched_bytes = patched.read_bytes()
    if len(unpatched_bytes) != len(patched_bytes) or unpatched_bytes == patched_bytes:
        raise RuntimeError("patched and unpatched kernel images have invalid sizes or contents")

    cubin_bytes = bytearray(cubin.read_bytes())
    patched_code = patched_bytes[patch_offset:]
    end = cubin_offset + len(patched_code)
    if end > len(cubin_bytes):
        raise RuntimeError("patched kernel image extends beyond the tracker CUBIN")
    cubin_bytes[cubin_offset:end] = patched_code

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patched_cubin = args.output_dir / "patched.cubin"
    patched_cubin.write_bytes(cubin_bytes)
    unpatched_sass = _disassemble(args.nvdisasm, cubin)
    patched_sass = _disassemble(args.nvdisasm, patched_cubin)
    (args.output_dir / "unpatched.sass").write_text(unpatched_sass, encoding="utf-8")
    (args.output_dir / "patched.sass").write_text(patched_sass, encoding="utf-8")
    site_count = _verify_patch(unpatched_sass, patched_sass, kernel)

    report = {
        "schema_version": 1,
        "kernel": kernel,
        "site_count": site_count,
        "driver_patch_offset": patch_offset,
        "cubin_code_offset": cubin_offset,
        "normalized_signature": NORMALIZED_SIGNATURE,
        "sha256": {
            "tracker_cubin": _sha256(cubin),
            "unpatched_kernel": _sha256(unpatched),
            "patched_kernel": _sha256(patched),
            "unpatched_disassembly": hashlib.sha256(unpatched_sass.encode()).hexdigest(),
            "patched_disassembly": hashlib.sha256(patched_sass.encode()).hexdigest(),
        },
    }
    report_path = args.output_dir / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
