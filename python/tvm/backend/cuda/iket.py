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
"""NVIDIA IKET annotations and ``run-iket`` orchestration.

``profile`` runs an explicit replayable command.  ``run`` is intended for a
script's ``__main__`` block: the parent process asks ``run-iket`` to replay the
original script or ``python -m`` invocation, while the injected tracker and
capture processes execute the supplied callable and then exit.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import tvm
from tvm.script import tirx as T

_PROFILE_ENV = "TVM_IKET_OFFICIAL_PROFILE"
_INJECTED_CHILD_ENABLE_ENV = "TVM_IKET_INJECTED_CHILD_ENABLE"
_DEFAULT_PROFILE = "cutlass-4.6.0"
_POSTPROCESS_CHOICES = frozenset(("perfetto", "json", "html", "none", "all"))
_INJECTION_ENV_VARS = ("CUDA_INJECTION64_PATH", "SMODEL_INJECTION_CONFIG")
_OUTPUT_TAIL_LINES = 100
_TERMINATION_GRACE_SECONDS = 5.0

_OFFICIAL_PROFILES = {
    "cutlass-4.6.0": {
        "nvrtc_version": (13, 2),
        "minimum_versions": {
            "nvidia-cutlass-dsl": "4.6.0",
            "nvidia-cutlass-dsl-libs-base": "4.6.0",
            "nvidia-cutlass-dsl-libs-core": "4.6.0",
            "nvidia-cutlass-dsl-libs-cu13": "4.6.0",
        },
        "exact_versions": {
            "nvidia-cuda-nvdisasm": "13.3.73",
            "nvidia-cuda-nvrtc": "13.2.78",
        },
    }
}


class IketProfileError(RuntimeError):
    """An error while validating or running an official IKET profile."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        command: Sequence[str] = (),
        output_tail: str = "",
        timeout: float | None = None,
    ) -> None:
        self.returncode = returncode
        self.command = tuple(command)
        self.output_tail = output_tail
        self.timeout = timeout
        super().__init__(message)


def _requested_formats(postprocess: str) -> frozenset[str]:
    if postprocess == "json":
        return frozenset(("json",))
    if postprocess == "perfetto":
        return frozenset(("perfetto",))
    if postprocess == "html":
        return frozenset(("perfetto", "html"))
    if postprocess == "all":
        return frozenset(("json", "perfetto", "html"))
    if postprocess == "none":
        return frozenset()
    raise ValueError(
        f"postprocess must be one of {sorted(_POSTPROCESS_CHOICES)}, got {postprocess!r}"
    )


@dataclass(frozen=True)
class IketProfileResult:
    """Published artifacts from a successful official IKET profile."""

    output_dir: Path
    postprocess: str
    json_traces: tuple[Path, ...]
    perfetto_traces: tuple[Path, ...]
    html_reports: tuple[Path, ...]
    command: tuple[str, ...] = ()

    def _one(self, format_name: str, artifacts: tuple[Path, ...], plural_name: str) -> Path:
        if format_name not in _requested_formats(self.postprocess):
            raise IketProfileError(
                f"The {format_name} artifact was not requested (postprocess={self.postprocess!r})"
            )
        if not artifacts:
            raise IketProfileError(
                f"The requested {format_name} artifact is missing from {self.output_dir}"
            )
        if len(artifacts) != 1:
            raise IketProfileError(
                f"The profile produced {len(artifacts)} {format_name} artifacts; "
                f"use {plural_name} for multi-process results"
            )
        return artifacts[0]

    @property
    def trace_json(self) -> Path:
        """Return the only JSON trace requested by this profile."""
        return self._one("json", self.json_traces, "json_traces")

    @property
    def perfetto(self) -> Path:
        """Return the only Perfetto trace requested by this profile."""
        return self._one("perfetto", self.perfetto_traces, "perfetto_traces")

    @property
    def html(self) -> Path:
        """Return the only HTML report requested by this profile."""
        return self._one("html", self.html_reports, "html_reports")

    @property
    def trace(self) -> Any:
        """Return the JSON trace without imposing a schema wrapper."""
        return json.loads(self.trace_json.read_text(encoding="utf-8"))

    @property
    def launches(self) -> Any:
        """Return ``trace[\"launches\"]`` without reshaping it."""
        return self.trace["launches"]


def _profile_error(message: str) -> IketProfileError:
    return IketProfileError(
        f"Official IKET profile validation failed: {message}. "
        "Use tvm.backend.cuda.iket.profile(command) or call "
        "tvm.backend.cuda.iket.run(main) from a replayable entry point."
    )


def _is_newer_release(actual: str, expected: str) -> bool:
    """Compare the numeric release versions published by NVIDIA wheels."""
    try:
        actual_parts = tuple(int(part) for part in actual.split("."))
        expected_parts = tuple(int(part) for part in expected.split("."))
    except ValueError:
        return False
    width = max(len(actual_parts), len(expected_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) > expected_parts + (0,) * (
        width - len(expected_parts)
    )


def _validate_run_iket_entrypoint() -> str:
    executable = shutil.which("run-iket")
    if executable is None or not os.access(executable, os.X_OK):
        raise _profile_error("the run-iket executable is unavailable")
    entry_points = metadata.distribution("nvidia-cutlass-dsl-libs-base").entry_points
    if not any(
        item.group == "console_scripts"
        and item.name == "run-iket"
        and item.value == "iket.cli.main:entrypoint"
        for item in entry_points
    ):
        raise _profile_error(
            "the run-iket entry point does not match the supported CUTLASS profile"
        )
    return executable


def _validate_nvrtc_version(expected_version: tuple[int, int]) -> None:
    expected_label = ".".join(str(part) for part in expected_version)
    try:
        from cuda.bindings import nvrtc

        error, major, minor = nvrtc.nvrtcVersion()
    except (ImportError, OSError, RuntimeError) as err:
        raise _profile_error(f"CUDA NVRTC {expected_label} is unavailable") from err
    actual_version = (int(major), int(minor))
    if int(error) != 0 or actual_version != expected_version:
        raise _profile_error(
            f"CUDA NVRTC {expected_label} is required, got {actual_version[0]}.{actual_version[1]}"
        )


def _validate_official_installation(profile_name: str) -> str:
    """Validate host-side package versions and return the official executable."""
    if profile_name not in _OFFICIAL_PROFILES:
        raise _profile_error(f"unsupported profile {profile_name!r}; expected {_DEFAULT_PROFILE!r}")
    profile_config = _OFFICIAL_PROFILES[profile_name]
    version_groups = (
        (profile_config["minimum_versions"], True),
        (profile_config["exact_versions"], False),
    )
    for versions, allow_newer in version_groups:
        for distribution_name, expected_version in versions.items():
            try:
                distribution = metadata.distribution(distribution_name)
            except metadata.PackageNotFoundError as err:
                operator = ">=" if allow_newer else "=="
                raise _profile_error(
                    f"{distribution_name}{operator}{expected_version} is not installed"
                ) from err
            if distribution.version != expected_version and not (
                allow_newer and _is_newer_release(distribution.version, expected_version)
            ):
                requirement = f"{expected_version} or newer" if allow_newer else expected_version
                raise _profile_error(
                    f"{distribution_name} must be {requirement}, got {distribution.version}"
                )

    executable = _validate_run_iket_entrypoint()
    _validate_nvrtc_version(profile_config["nvrtc_version"])
    return executable


def _validate_injection_environment() -> None:
    injection_value = os.environ.get("CUDA_INJECTION64_PATH")
    injection_path = Path(injection_value) if injection_value else None
    if injection_path is None or not injection_path.is_file():
        raise _profile_error("CUDA_INJECTION64_PATH was not supplied by run-iket")

    config_value = os.environ.get("SMODEL_INJECTION_CONFIG")
    config_path = Path(config_value) if config_value else None
    if config_path is None or not config_path.is_file():
        raise _profile_error("SMODEL_INJECTION_CONFIG was not supplied by run-iket")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise _profile_error("SMODEL_INJECTION_CONFIG is not valid JSON") from err
    if not isinstance(config, dict):
        raise _profile_error("SMODEL_INJECTION_CONFIG must contain a JSON object")
    tool_name = config.get("toolName")
    if tool_name not in ("tracker", "iket"):
        raise _profile_error("SMODEL_INJECTION_CONFIG was not generated by run-iket profile")
    if tool_name == "tracker":
        return
    tool_config = config.get("toolConfig")
    if not isinstance(tool_config, dict):
        raise _profile_error("run-iket capture config is missing toolConfig")
    instrument_path = Path(tool_config.get("appInstrument", ""))
    if not instrument_path.is_file():
        raise _profile_error("run-iket did not provide an application instrumentation manifest")


def _validate_official_environment() -> str:
    """Validate the installation plus tracker/capture injection environment."""
    profile_name = os.environ.get(_PROFILE_ENV)
    if profile_name not in _OFFICIAL_PROFILES:
        raise _profile_error(
            f"{_PROFILE_ENV} must be set to {_DEFAULT_PROFILE}, got {profile_name!r}"
        )
    executable = _validate_official_installation(profile_name)
    _validate_injection_environment()
    return executable


def validate_official_environment() -> None:
    """Validate the supported official runtime before an instrumented CUBIN is loaded."""
    _validate_official_environment()


class _OfficialIketExecutable:
    """Executable whose CUDA image is patched and decoded by NVIDIA run-iket."""

    def __init__(self, executable) -> None:
        self._executable = executable
        self._jitted_module = None
        self._jit_lock = threading.Lock()

    @property
    def mod(self):
        """Return the offline module without loading its CUDA image."""
        return self._executable.mod

    def jit(self, *args, **kwargs):
        """Validate run-iket before the first load and every forced recompile."""
        force_recompile = bool(kwargs.get("force_recompile", False))
        with self._jit_lock:
            if self._jitted_module is None or force_recompile:
                _validate_official_environment()
                self._jitted_module = self._executable.jit(*args, **kwargs)
            return self._jitted_module

    def export_library(self, *_args, **_kwargs):
        raise RuntimeError(
            "Official IKET executables cannot be exported; run-iket must patch the "
            "JIT-loaded CUDA image in the profiling process"
        )

    def __getitem__(self, name):
        def invoke(*args, **kwargs):
            return self.jit().get_function(name, query_imports=True)(*args, **kwargs)

        return invoke

    def __call__(self, *args, **kwargs):
        return self.jit().main(*args, **kwargs)


@T.meta_class
class IketProfiler:
    """TIRx annotations compiled for NVIDIA's official IKET runtime."""

    def mark(self, name: str, payload=None):
        if payload is None:
            T.evaluate(T.cuda.iket.mark(name))
        else:
            T.evaluate(T.cuda.iket.mark(name, payload))

    def range_start(self, name: str, payload=None):
        if payload is None:
            return T.cuda.iket.range_start(name)
        return T.cuda.iket.range_start(name, payload)

    def range_end(self, token: tvm.tirx.Expr, payload=None):
        if payload is None:
            T.evaluate(T.cuda.iket.range_end(token))
        else:
            T.evaluate(T.cuda.iket.range_end(token, payload))

    def range_push(self, name: str, payload=None):
        if payload is None:
            T.evaluate(T.cuda.iket.range_push(name))
        else:
            T.evaluate(T.cuda.iket.range_push(name, payload))

    def range_pop(self):
        T.evaluate(T.cuda.iket.range_pop())

    def sentinel_token(self, name: str):
        return T.cuda.iket.sentinel_token(name)

    def compile(self, mod, target=None, *, tir_pipeline="tirx"):
        """Compile official IKET metadata and NativeDump placeholders."""
        if isinstance(mod, tvm.tirx.PrimFunc):
            mod = tvm.IRModule.from_expr(mod)
        if not isinstance(mod, tvm.IRModule):
            raise TypeError("IketProfiler.compile expects a TIRx PrimFunc or IRModule")
        enabled_mod = mod.with_attr("tirx.iket.enabled", True)
        executable = tvm.compile(enabled_mod, target=target, tir_pipeline=tir_pipeline)
        return _OfficialIketExecutable(executable)


def _normalize_command(command: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    if isinstance(command, str | bytes | os.PathLike):
        raise TypeError("command must be a sequence of arguments, not a shell command string")
    try:
        normalized = tuple(os.fspath(arg) for arg in command)
    except TypeError as err:
        raise TypeError("every command argument must be a string or path-like object") from err
    if not normalized:
        raise ValueError("command must contain at least one argument")
    if any(not isinstance(arg, str) for arg in normalized):
        raise TypeError("every command argument must resolve to a string")
    if any("\x00" in arg for arg in normalized):
        raise ValueError("command arguments cannot contain NUL bytes")
    return normalized


def _validate_profile_options(
    postprocess: str, timeout: float | None, max_ts_cnt_per_warp: int | None
) -> float | None:
    _requested_formats(postprocess)
    if timeout is not None:
        if isinstance(timeout, bool):
            raise TypeError("timeout must be a positive number or None")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive or None")
    if max_ts_cnt_per_warp is not None:
        if isinstance(max_ts_cnt_per_warp, bool) or not isinstance(max_ts_cnt_per_warp, int):
            raise TypeError("max_ts_cnt_per_warp must be an integer or None")
        if max_ts_cnt_per_warp <= 0:
            raise ValueError("max_ts_cnt_per_warp must be positive")
    return timeout


def _child_environment(
    profile_name: str, env: Mapping[str, str | os.PathLike[str]] | None
) -> dict[str, str]:
    if not isinstance(profile_name, str):
        raise TypeError("profile_name must be a string")
    child_env = os.environ.copy()
    if env is not None:
        if not isinstance(env, Mapping):
            raise TypeError("env must be a mapping or None")
        for key, value in env.items():
            normalized_key = os.fspath(key)
            normalized_value = os.fspath(value)
            if not isinstance(normalized_key, str) or not isinstance(normalized_value, str):
                raise TypeError("env keys and values must resolve to strings")
            child_env[normalized_key] = normalized_value
    inherited_profile = child_env.get(_PROFILE_ENV)
    if inherited_profile is not None and inherited_profile != profile_name:
        warnings.warn(
            f"Ignoring inherited {_PROFILE_ENV}={inherited_profile!r}; "
            f"profile_name={profile_name!r} takes precedence",
            RuntimeWarning,
            stacklevel=3,
        )
    child_env[_PROFILE_ENV] = profile_name
    # LowerIket also requires the two run-iket injection variables before it
    # honors this marker.  This enables ordinary TIRx JIT compilation only in
    # children started by this validated profiling entry point.
    child_env[_INJECTED_CHILD_ENABLE_ENV] = "1"
    return child_env


def _build_run_iket_argv(
    executable: str,
    command: Sequence[str],
    *,
    output_dir: Path,
    postprocess: str,
    max_ts_cnt_per_warp: int | None,
    keep: bool,
) -> list[str]:
    argv = [
        executable,
        "--output-dir",
        str(output_dir),
        "--clobber",
        "profile",
        "--postprocess",
        postprocess,
        "--keep" if keep else "--no-keep",
    ]
    if max_ts_cnt_per_warp is not None:
        argv.extend(("--max-ts-cnt-per-warp", str(max_ts_cnt_per_warp)))
    argv.append("--")
    argv.extend(command)
    return argv


def _stream_output(pipe, output_tail: deque[str]) -> None:
    try:
        while True:
            raw_line = pipe.readline()
            if raw_line in ("", b""):
                break
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line
            sys.stdout.write(line)
            sys.stdout.flush()
            split_lines = line.splitlines()
            if not split_lines:
                split_lines = [line.rstrip("\r\n")]
            output_tail.extend(split_lines)
    except (OSError, ValueError):
        # The waiting thread closes the pipe to stop a reader after termination.
        return


def _stop_output_thread(proc: subprocess.Popen, output_thread: threading.Thread | None) -> None:
    if output_thread is None:
        return
    output_thread.join(timeout=1.0)
    if output_thread.is_alive() and proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass
        output_thread.join(timeout=1.0)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(proc: subprocess.Popen) -> None:
    process_group = proc.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            proc.wait(timeout=min(0.1, max(remaining, 0.0)))
        except subprocess.TimeoutExpired:
            continue
        if _process_group_exists(process_group):
            time.sleep(min(0.05, max(remaining, 0.0)))

    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # The process has been signalled and will be reaped by the interpreter
        # if a broken test double or an unusual platform cannot report it here.
        pass


def _format_process_error(
    message: str,
    *,
    command: Sequence[str],
    output_tail: deque[str],
    returncode: int | None,
    timeout: float | None,
) -> IketProfileError:
    tail_text = "\n".join(output_tail)
    detail = f"{message}\nCommand: {shlex.join(command)}"
    if tail_text:
        detail += f"\nLast {len(output_tail)} output line(s):\n{tail_text}"
    return IketProfileError(
        detail,
        returncode=returncode,
        command=command,
        output_tail=tail_text,
        timeout=timeout,
    )


def _run_process(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None,
    env: Mapping[str, str],
    timeout: float | None,
) -> None:
    output_tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            shell=False,
        )
    except OSError as err:
        raise _format_process_error(
            f"Failed to start official IKET profiler: {err}",
            command=argv,
            output_tail=output_tail,
            returncode=None,
            timeout=timeout,
        ) from err

    output_thread = None
    if proc.stdout is not None:
        output_thread = threading.Thread(
            target=_stream_output,
            args=(proc.stdout, output_tail),
            name="tvm-iket-output",
            daemon=True,
        )
        output_thread.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as err:
        _terminate_process_group(proc)
        _stop_output_thread(proc, output_thread)
        raise _format_process_error(
            f"Official IKET profiling timed out after {timeout} seconds",
            command=argv,
            output_tail=output_tail,
            returncode=None,
            timeout=timeout,
        ) from err
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        _stop_output_thread(proc, output_thread)
        raise

    _stop_output_thread(proc, output_thread)
    if returncode != 0:
        raise _format_process_error(
            f"Official IKET profiler exited with status {returncode}",
            command=argv,
            output_tail=output_tail,
            returncode=returncode,
            timeout=timeout,
        )


def _collect_artifacts(
    output_dir: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    root_json = {path for path in output_dir.glob("*.json") if path.is_file()}
    nested_trace_json = {path for path in output_dir.rglob("*.trace.json") if path.is_file()}
    json_traces = tuple(sorted(root_json | nested_trace_json))
    perfetto_traces = tuple(
        sorted(path for path in output_dir.rglob("*.pftrace") if path.is_file())
    )
    html_reports = tuple(sorted(path for path in output_dir.rglob("*.html") if path.is_file()))
    return json_traces, perfetto_traces, html_reports


def _validate_requested_artifacts(output_dir: Path, postprocess: str) -> None:
    json_traces, perfetto_traces, html_reports = _collect_artifacts(output_dir)
    requested = _requested_formats(postprocess)
    missing = []
    if "json" in requested and not json_traces:
        missing.append("JSON trace")
    if "perfetto" in requested and not perfetto_traces:
        missing.append("Perfetto trace")
    if "html" in requested and not html_reports:
        missing.append("HTML report")
    if missing:
        raise IketProfileError(
            f"Official IKET profiling completed without the requested "
            f"{', '.join(missing)} in {output_dir}"
        )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publish_staging(staging: Path, output_dir: Path, *, clobber: bool) -> None:
    if not _path_exists(output_dir):
        os.replace(staging, output_dir)
        return
    if not clobber:
        raise FileExistsError(f"IKET output directory already exists: {output_dir}")

    backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    try:
        _remove_path(backup)
    except OSError:
        # The new result is already published and the prior result remains in
        # a uniquely named backup rather than risking rollback of valid data.
        pass


def profile(
    command: Sequence[str | os.PathLike[str]],
    *,
    output_dir: str | os.PathLike[str],
    profile_name: str = _DEFAULT_PROFILE,
    postprocess: str = "all",
    clobber: bool = False,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str | os.PathLike[str]] | None = None,
    max_ts_cnt_per_warp: int | None = None,
    keep: bool = False,
    timeout: float | None = 600.0,
) -> IketProfileResult:
    """Profile a replayable command with NVIDIA's official ``run-iket`` tool."""
    if _is_injected_process():
        raise IketProfileError(
            "iket.profile() cannot start nested profiling from an injected IKET process"
        )
    normalized_command = _normalize_command(command)
    timeout = _validate_profile_options(postprocess, timeout, max_ts_cnt_per_warp)
    target = Path(output_dir).expanduser().resolve()
    if target == target.parent:
        raise ValueError("output_dir cannot be a filesystem root")
    if _path_exists(target) and not clobber:
        raise FileExistsError(f"IKET output directory already exists: {target}")

    child_env = _child_environment(profile_name, env)
    executable = _validate_official_installation(profile_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    published = False
    try:
        argv = _build_run_iket_argv(
            executable,
            normalized_command,
            output_dir=staging,
            postprocess=postprocess,
            max_ts_cnt_per_warp=max_ts_cnt_per_warp,
            keep=keep,
        )
        _run_process(argv, cwd=cwd, env=child_env, timeout=timeout)
        _validate_requested_artifacts(staging, postprocess)
        _publish_staging(staging, target, clobber=clobber)
        published = True
        json_traces, perfetto_traces, html_reports = _collect_artifacts(target)
        return IketProfileResult(
            output_dir=target,
            postprocess=postprocess,
            json_traces=json_traces,
            perfetto_traces=perfetto_traces,
            html_reports=html_reports,
            command=normalized_command,
        )
    finally:
        if not published and _path_exists(staging):
            _remove_path(staging)


def _is_injected_process() -> bool:
    return any(name in os.environ for name in _INJECTION_ENV_VARS)


def _unwrap_partial(callable_obj):
    while isinstance(callable_obj, functools.partial):
        callable_obj = callable_obj.func
    return callable_obj


def _replay_error(reason: str) -> ValueError:
    return ValueError(
        f"iket.run() cannot safely replay this entry point: {reason}. "
        "Use iket.profile(command) with an explicit replayable command instead."
    )


def _parse_python_entry(orig_argv: Sequence[str]) -> tuple[str, str]:
    if not orig_argv:
        raise _replay_error("sys.orig_argv is unavailable")
    options_with_value = frozenset(("-W", "-X", "--check-hash-based-pycs"))
    index = 1
    while index < len(orig_argv):
        arg = orig_argv[index]
        if arg == "-m":
            if index + 1 >= len(orig_argv):
                raise _replay_error("python -m has no module name")
            return "module", orig_argv[index + 1]
        if arg.startswith("-m") and len(arg) > 2:
            return "module", arg[2:]
        if arg == "-c" or (arg.startswith("-c") and len(arg) > 2):
            raise _replay_error("python -c has no replayable source file")
        if arg == "-":
            raise _replay_error("stdin has no replayable source file")
        if arg == "--":
            if index + 1 >= len(orig_argv) or orig_argv[index + 1] == "-":
                raise _replay_error("stdin has no replayable source file")
            return "script", orig_argv[index + 1]
        if arg in options_with_value:
            index += 2
            continue
        if arg.startswith("-W") or arg.startswith("-X"):
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return "script", arg
    raise _replay_error("sys.orig_argv does not name a script or python -m module")


def _resolve_module_entry(module_name: str, main_module: ModuleType) -> Path:
    main_spec = getattr(main_module, "__spec__", None)
    valid_names = (module_name, f"{module_name}.__main__")
    if main_spec is not None and main_spec.name in valid_names and main_spec.origin:
        return Path(main_spec.origin).resolve()
    try:
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is not None and module_spec.submodule_search_locations is not None:
            module_spec = importlib.util.find_spec(f"{module_name}.__main__")
    except (ImportError, AttributeError, ValueError) as err:
        raise _replay_error(f"cannot resolve python -m module {module_name!r}") from err
    if module_spec is None or not module_spec.origin:
        raise _replay_error(f"cannot resolve python -m module {module_name!r}")
    return Path(module_spec.origin).resolve()


def _replay_command(main) -> tuple[str, ...]:
    if not callable(main):
        raise TypeError("main must be callable")
    base_callable = _unwrap_partial(main)
    if getattr(base_callable, "__module__", None) != "__main__":
        raise _replay_error("the callable is not defined in __main__")

    main_module = sys.modules.get("__main__")
    if main_module is None:
        raise _replay_error("the __main__ module is unavailable")
    if inspect.isfunction(base_callable) and base_callable.__globals__ is not vars(main_module):
        raise _replay_error("the callable is not owned by the active __main__ module")

    try:
        callable_source_name = inspect.getsourcefile(base_callable) or inspect.getfile(
            base_callable
        )
    except (OSError, TypeError) as err:
        raise _replay_error("the callable source file cannot be located") from err
    main_file_name = getattr(main_module, "__file__", None)
    if not callable_source_name or not main_file_name:
        raise _replay_error("the callable or __main__ source file cannot be located")
    callable_source = Path(callable_source_name).resolve()
    main_file = Path(main_file_name).resolve()
    if not callable_source.is_file() or not main_file.is_file():
        raise _replay_error("the callable and __main__ must resolve to real files")

    orig_argv = tuple(getattr(sys, "orig_argv", ()))
    entry_kind, entry_value = _parse_python_entry(orig_argv)
    if entry_kind == "script":
        replay_source = Path(entry_value).expanduser().resolve()
    else:
        replay_source = _resolve_module_entry(entry_value, main_module)
    if not replay_source.is_file():
        raise _replay_error("the replay entry point does not resolve to a real file")
    if not (callable_source == main_file == replay_source):
        raise _replay_error(
            "the callable source, __main__.__file__, and replay entry point are different files"
        )
    return orig_argv


def run(
    main,
    *,
    output_dir: str | os.PathLike[str],
    profile_name: str = _DEFAULT_PROFILE,
    postprocess: str = "all",
    clobber: bool = False,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str | os.PathLike[str]] | None = None,
    max_ts_cnt_per_warp: int | None = None,
    keep: bool = False,
    timeout: float | None = 600.0,
) -> IketProfileResult:
    """Replay the active script under ``run-iket`` and execute ``main`` in its passes.

    Module-level code executes once in the parent, tracker, and capture
    processes.  ``main`` executes only in tracker and capture.  Code following
    ``run`` executes only in the parent after a successful profile.
    """
    if not callable(main):
        raise TypeError("main must be callable")
    if _is_injected_process():
        _validate_official_environment()
        main()
        raise SystemExit(0)

    command = _replay_command(main)
    return profile(
        command,
        output_dir=output_dir,
        profile_name=profile_name,
        postprocess=postprocess,
        clobber=clobber,
        cwd=cwd,
        env=env,
        max_ts_cnt_per_warp=max_ts_cnt_per_warp,
        keep=keep,
        timeout=timeout,
    )


__all__ = [
    "IketProfileError",
    "IketProfileResult",
    "IketProfiler",
    "profile",
    "run",
]
