#!/usr/bin/env python3
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

"""Inject TVM's CUDA runtime DSO into a wheel and refresh RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


def _wheel_escape(value: str) -> str:
    """Escape a distribution component for wheel filenames and dist-info dirs."""

    return re.sub(r"[^\w\d.]+", "_", value).lower()


def _hash_record(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", str(len(data))


def _copy_info(info: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(filename=filename, date_time=info.date_time)
    copied.compress_type = info.compress_type
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    return copied


def _replace_header(metadata: bytes, key: str, value: str) -> bytes:
    text = metadata.decode("utf-8")
    lines = text.splitlines(keepends=True)
    prefix = f"{key.lower()}:"
    for index, line in enumerate(lines):
        if line.lower().startswith(prefix):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[index] = f"{key}: {value}{newline}"
            return "".join(lines).encode("utf-8")
    raise ValueError(f"METADATA does not contain {key!r}")


def _find_dist_info(names: list[str]) -> str:
    dist_infos = sorted({name.split("/", 1)[0] for name in names if ".dist-info/" in name})
    dist_infos = [name for name in dist_infos if name.endswith(".dist-info")]
    if len(dist_infos) != 1:
        raise ValueError(f"Expected one .dist-info directory, found {dist_infos}")
    return dist_infos[0]


def _metadata_headers(metadata: bytes) -> tuple[str, str]:
    headers = Parser().parsestr(metadata.decode("utf-8"))
    name = headers.get("Name")
    version = headers.get("Version")
    if not name or not version:
        raise ValueError("METADATA must contain Name and Version")
    return name, version


def _retag_wheel_filename(
    wheel: Path,
    dist_name: str,
    version: str,
) -> str:
    parts = wheel.name.removesuffix(".whl").split("-")
    if len(parts) not in (5, 6):
        raise ValueError(f"Unsupported wheel filename: {wheel.name}")
    tags = parts[2:]
    return f"{_wheel_escape(dist_name)}-{version}-{'-'.join(tags)}.whl"


def rewrite_wheel(
    wheel: Path,
    output_dir: Path,
    cuda_runtime: Path | None,
    target_path: str,
    distribution_name: str | None,
    distribution_version: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "r") as zin:
        original_names = zin.namelist()
        original_dist_info = _find_dist_info(original_names)
        metadata_path = f"{original_dist_info}/METADATA"
        original_name, original_version = _metadata_headers(zin.read(metadata_path))

        final_name = distribution_name or original_name
        final_version = distribution_version or original_version
        final_dist_info = f"{_wheel_escape(final_name)}-{final_version}.dist-info"
        record_path = f"{final_dist_info}/RECORD"
        output_path = output_dir / _retag_wheel_filename(wheel, final_name, final_version)

        entries: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info in zin.infolist():
            if info.is_dir():
                continue
            mapped_name = info.filename
            if mapped_name == f"{original_dist_info}/RECORD":
                continue
            if mapped_name.startswith(f"{original_dist_info}/"):
                mapped_name = f"{final_dist_info}/{mapped_name.split('/', 1)[1]}"
            if cuda_runtime is not None and mapped_name == target_path:
                continue

            data = zin.read(info.filename)
            if mapped_name == f"{final_dist_info}/METADATA":
                if distribution_name is not None:
                    data = _replace_header(data, "Name", final_name)
                if distribution_version is not None:
                    data = _replace_header(data, "Version", final_version)
            entries.append((_copy_info(info, mapped_name), data))

        if cuda_runtime is not None:
            data = cuda_runtime.read_bytes()
            info = zipfile.ZipInfo(target_path)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            entries.append((info, data))

    record_buffer = io.StringIO()
    writer = csv.writer(record_buffer, lineterminator="\n")
    for info, data in entries:
        digest, size = _hash_record(data)
        writer.writerow([info.filename, digest, size])
    writer.writerow([record_path, "", ""])

    record_info = zipfile.ZipInfo(record_path)
    record_info.compress_type = zipfile.ZIP_DEFLATED
    record_info.external_attr = 0o644 << 16
    entries.append((record_info, record_buffer.getvalue().encode("utf-8")))

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, data in entries:
            zout.writestr(info, data)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--cuda-runtime", type=Path)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distribution-name")
    parser.add_argument("--distribution-version")
    args = parser.parse_args()

    cuda_runtime = args.cuda_runtime
    if cuda_runtime is not None and not cuda_runtime.is_file():
        parser.error(f"CUDA runtime DSO does not exist: {cuda_runtime}")

    target_path = args.target_path
    if target_path is None:
        if cuda_runtime is None:
            target_path = "tvm/lib/libtvm_runtime_cuda.so"
        else:
            target_path = f"tvm/lib/{cuda_runtime.name}"

    output_path = rewrite_wheel(
        wheel=args.wheel,
        output_dir=args.output_dir,
        cuda_runtime=cuda_runtime,
        target_path=target_path,
        distribution_name=args.distribution_name,
        distribution_version=args.distribution_version,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
