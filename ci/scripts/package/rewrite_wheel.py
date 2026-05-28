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

"""Rewrite TVM wheel metadata and inject extra runtime files."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tempfile
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


def _is_elf_shared_lib(name: str, data: bytes) -> bool:
    return (
        name.startswith("tvm/lib/")
        and re.search(r"\.so(?:\.|$)", Path(name).name) is not None
        and data.startswith(b"\x7fELF")
    )


def _set_rpath(data: bytes, rpath: str, name: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / Path(name).name
        path.write_bytes(data)
        path.chmod(0o755)
        try:
            subprocess.run(["patchelf", "--set-rpath", rpath, str(path)], check=True)
        except subprocess.CalledProcessError as err:
            raise ValueError(f"patchelf failed while setting rpath on {name}") from err
        return path.read_bytes()


def _retag_wheel_filename(
    wheel: Path,
    dist_name: str,
    version: str,
) -> str:
    parts = wheel.name.removesuffix(".whl").split("-")
    if len(parts) not in (5, 6):
        raise ValueError(f"Unsupported wheel filename: {wheel.name}")
    tags = parts[2:]
    return f"{_wheel_escape(dist_name)}-{_wheel_escape(version)}-{'-'.join(tags)}.whl"


def _normalize_wheel_path(value: str, label: str) -> str:
    raw = value.replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw) is not None:
        raise argparse.ArgumentTypeError(
            f"{label} must be a relative wheel path without drive, empty, '.' or '..' segments"
        )
    normalized = raw
    parts = normalized.split("/")
    if (
        not normalized
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must be a relative wheel path without drive, empty, '.' or '..' segments"
        )
    return normalized


def _validate_wheel_member_path(value: str) -> str:
    if "\\" in value:
        raise ValueError(f"Wheel member path must use forward slashes: {value}")
    try:
        normalized = _normalize_wheel_path(value, "wheel member path")
    except argparse.ArgumentTypeError as err:
        raise ValueError(str(err)) from err
    if normalized != value:
        raise ValueError(f"Wheel member path is not normalized: {value}")
    return normalized


def _parse_extra_file(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("extra files must use SOURCE=TARGET format")
    source, target = value.split("=", 1)
    if not source or not target:
        raise argparse.ArgumentTypeError("extra files must use SOURCE=TARGET format")
    return Path(source), _normalize_wheel_path(target, "extra file target")


def _extra_library_files(
    library_dirs: list[Path],
    patterns: list[str],
    target_dir: str,
) -> list[tuple[Path, str]]:
    target_dir = _normalize_wheel_path(target_dir, "extra library target dir")
    extra_files: list[tuple[Path, str]] = []
    missing_dirs = [str(library_dir) for library_dir in library_dirs if not library_dir.is_dir()]
    if missing_dirs:
        raise ValueError(f"extra library dirs do not exist: {', '.join(missing_dirs)}")
    for library_dir in library_dirs:
        for pattern in patterns:
            for source in sorted(library_dir.glob(pattern)):
                if source.is_file():
                    extra_files.append((source, f"{target_dir}/{source.name}"))
    if library_dirs and patterns and not extra_files:
        raise ValueError(
            "extra library patterns did not match any files: " + ", ".join(patterns)
        )
    return sorted(extra_files, key=lambda item: (item[1], str(item[0])))


def _check_duplicate_targets(targets: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for target in targets:
        if target in seen:
            duplicates.add(target)
        seen.add(target)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate wheel target paths are not allowed: {joined}")


def rewrite_wheel(
    wheel: Path,
    output_dir: Path,
    cuda_runtime: Path | None,
    target_path: str,
    distribution_name: str | None,
    distribution_version: str | None,
    set_rpath: str | None,
    extra_files: list[tuple[Path, str]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_targets = {target for _, target in extra_files}
    with zipfile.ZipFile(wheel, "r") as zin:
        original_infos = [info for info in zin.infolist() if not info.is_dir()]
        original_names = [_validate_wheel_member_path(info.filename) for info in original_infos]
        _check_duplicate_targets(original_names)
        original_dist_info = _find_dist_info(original_names)
        metadata_path = f"{original_dist_info}/METADATA"
        if metadata_path not in original_names:
            raise ValueError(f"Wheel metadata is missing: {metadata_path}")
        original_name, original_version = _metadata_headers(zin.read(metadata_path))

        final_name = distribution_name or original_name
        final_version = distribution_version or original_version
        final_dist_info = f"{_wheel_escape(final_name)}-{_wheel_escape(final_version)}.dist-info"
        record_path = f"{final_dist_info}/RECORD"
        target_paths = [target for _, target in extra_files]
        if cuda_runtime is not None:
            target_paths.append(target_path)
        target_paths.append(record_path)
        _check_duplicate_targets(target_paths)
        output_path = output_dir / _retag_wheel_filename(wheel, final_name, final_version)

        entries: list[tuple[zipfile.ZipInfo, bytes]] = []
        entry_names: list[str] = []
        for info in original_infos:
            mapped_name = info.filename
            if mapped_name == f"{original_dist_info}/RECORD":
                continue
            if mapped_name.startswith(f"{original_dist_info}/"):
                mapped_name = f"{final_dist_info}/{mapped_name.split('/', 1)[1]}"
            mapped_name = _validate_wheel_member_path(mapped_name)
            if (
                cuda_runtime is not None and mapped_name == target_path
            ) or mapped_name in extra_targets:
                continue

            data = zin.read(info)
            if mapped_name == f"{final_dist_info}/METADATA":
                if distribution_name is not None:
                    data = _replace_header(data, "Name", final_name)
                if distribution_version is not None:
                    data = _replace_header(data, "Version", final_version)
            if set_rpath is not None and _is_elf_shared_lib(mapped_name, data):
                data = _set_rpath(data, set_rpath, mapped_name)
            entries.append((_copy_info(info, mapped_name), data))
            entry_names.append(mapped_name)

        if cuda_runtime is not None:
            data = cuda_runtime.read_bytes()
            if set_rpath is not None and _is_elf_shared_lib(target_path, data):
                data = _set_rpath(data, set_rpath, target_path)
            info = zipfile.ZipInfo(target_path)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            entries.append((info, data))
            entry_names.append(target_path)

        for source, target in extra_files:
            data = source.read_bytes()
            if set_rpath is not None and _is_elf_shared_lib(target, data):
                data = _set_rpath(data, set_rpath, target)
            info = zipfile.ZipInfo(target)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            entries.append((info, data))
            entry_names.append(target)

        _check_duplicate_targets([*entry_names, record_path])

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
    parser.add_argument("--set-rpath")
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        type=_parse_extra_file,
        help="Additional file to place in the wheel, using SOURCE=TARGET format.",
    )
    parser.add_argument(
        "--extra-library-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory to scan for extra runtime libraries.",
    )
    parser.add_argument(
        "--extra-library-pattern",
        action="append",
        default=[],
        help="Glob pattern for files under --extra-library-dir.",
    )
    parser.add_argument(
        "--extra-library-target-dir",
        default="tvm/lib",
        help="Wheel directory for files matched by --extra-library-pattern.",
    )
    args = parser.parse_args()

    cuda_runtime = args.cuda_runtime
    if cuda_runtime is not None and not cuda_runtime.is_file():
        parser.error(f"CUDA runtime DSO does not exist: {cuda_runtime}")
    if args.set_rpath and shutil.which("patchelf") is None:
        parser.error("--set-rpath requires patchelf on PATH")

    target_path = args.target_path
    if target_path is None:
        if cuda_runtime is None:
            target_path = "tvm/lib/libtvm_runtime_cuda.so"
        else:
            target_path = f"tvm/lib/{cuda_runtime.name}"
    else:
        try:
            target_path = _normalize_wheel_path(target_path, "target path")
        except argparse.ArgumentTypeError as err:
            parser.error(str(err))

    extra_files = list(args.extra_file)
    try:
        extra_files.extend(
            _extra_library_files(
                library_dirs=args.extra_library_dir,
                patterns=args.extra_library_pattern,
                target_dir=args.extra_library_target_dir,
            )
        )
    except (argparse.ArgumentTypeError, ValueError) as err:
        parser.error(str(err))
    missing_extra_files = [str(source) for source, _ in extra_files if not source.is_file()]
    if missing_extra_files:
        parser.error(f"extra files do not exist: {', '.join(missing_extra_files)}")
    target_paths = [target for _, target in extra_files]
    if cuda_runtime is not None:
        target_paths.append(target_path)
    try:
        _check_duplicate_targets(target_paths)
    except ValueError as err:
        parser.error(str(err))

    try:
        output_path = rewrite_wheel(
            wheel=args.wheel,
            output_dir=args.output_dir,
            cuda_runtime=cuda_runtime,
            target_path=target_path,
            distribution_name=args.distribution_name or None,
            distribution_version=args.distribution_version or None,
            set_rpath=args.set_rpath,
            extra_files=extra_files,
        )
    except (ValueError, zipfile.BadZipFile, KeyError) as err:
        parser.error(str(err))
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
