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

"""Tests for NVIDIA IKET command orchestration and replay semantics."""

from __future__ import annotations

import importlib.machinery
import os
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

from tvm.tirx.cuda import iket


def _write_artifacts(output_dir: Path, postprocess: str, *, count: int = 1) -> None:
    for index in range(count):
        if postprocess in ("json", "all"):
            (output_dir / f"iket_pid_{index}.trace.json").write_text(
                '{"launches": [{"index": %d}]}' % index, encoding="utf-8"
            )
        if postprocess in ("perfetto", "html", "all"):
            (output_dir / f"iket_pid_{index}.pftrace").write_bytes(b"perfetto")
        if postprocess in ("html", "all"):
            (output_dir / f"iket_pid_{index}.html").write_text("<html />", encoding="utf-8")


def _fake_profile_process(captured, *, artifact_count: int = 1):
    def run_process(argv, *, cwd, env, timeout):
        captured.update(argv=list(argv), cwd=cwd, env=env, timeout=timeout)
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        postprocess = argv[argv.index("--postprocess") + 1]
        _write_artifacts(output_dir, postprocess, count=artifact_count)

    return run_process


def test_public_namespace_and_old_bench_path_is_absent():
    pytest.importorskip("triton")  # tvm.tirx.bench pulls in the triton harness
    import tvm.tirx.bench as bench

    assert iket.__all__ == [
        "IketProfileError",
        "IketProfileResult",
        "IketProfiler",
        "profile",
        "run",
    ]
    assert not hasattr(bench, "IketProfiler")


@pytest.mark.parametrize("injection_variable", ["CUDA_INJECTION64_PATH", "SMODEL_INJECTION_CONFIG"])
def test_profile_rejects_nested_injection_before_validation(
    injection_variable, tmp_path, monkeypatch
):
    validation_started = False

    def validate(_name):
        nonlocal validation_started
        validation_started = True

    monkeypatch.setenv(injection_variable, "injected")
    monkeypatch.setattr(iket, "_validate_official_installation", validate)
    target = tmp_path / "result"

    with pytest.raises(iket.IketProfileError, match="nested profiling"):
        iket.profile(("python", "workload.py"), output_dir=target)

    assert not validation_started
    assert not target.exists()


@pytest.mark.parametrize(("keep", "keep_arg"), [(True, "--keep"), (False, "--no-keep")])
def test_build_run_iket_argv_is_exact(keep, keep_arg, tmp_path):
    output_dir = tmp_path / "staging"
    assert iket._build_run_iket_argv(  # pylint: disable=protected-access
        "/opt/bin/run-iket",
        ("python", "workload.py", "--size", "4"),
        output_dir=output_dir,
        postprocess="html",
        max_ts_cnt_per_warp=23,
        keep=keep,
    ) == [
        "/opt/bin/run-iket",
        "--output-dir",
        str(output_dir),
        "--clobber",
        "profile",
        "--postprocess",
        "html",
        keep_arg,
        "--max-ts-cnt-per-warp",
        "23",
        "--",
        "python",
        "workload.py",
        "--size",
        "4",
    ]


def test_profile_forwards_cwd_environment_timeout_and_publishes(tmp_path, monkeypatch):
    target = tmp_path / "published"
    cwd = tmp_path / "work"
    cwd.mkdir()
    captured = {}
    monkeypatch.setattr(
        iket, "_validate_official_installation", lambda profile_name: "/opt/bin/run-iket"
    )
    monkeypatch.setattr(iket, "_run_process", _fake_profile_process(captured))
    monkeypatch.setenv("TVM_IKET_OFFICIAL_PROFILE", "inherited-profile")

    with pytest.warns(RuntimeWarning, match="takes precedence"):
        result = iket.profile(
            (sys.executable, "workload.py"),
            output_dir=target,
            profile_name="cutlass-4.6.0",
            postprocess="all",
            clobber=False,
            cwd=cwd,
            env={"IKET_TEST_ENV": "present"},
            max_ts_cnt_per_warp=17,
            keep=True,
            timeout=12.5,
        )

    staging = Path(captured["argv"][2])
    assert staging.parent == target.parent
    assert staging != target
    assert captured["cwd"] == cwd
    assert captured["timeout"] == 12.5
    assert captured["env"]["IKET_TEST_ENV"] == "present"
    assert captured["env"]["TVM_IKET_OFFICIAL_PROFILE"] == "cutlass-4.6.0"
    assert captured["env"]["TVM_IKET_INJECTED_CHILD_ENABLE"] == "1"
    assert os.environ["TVM_IKET_OFFICIAL_PROFILE"] == "inherited-profile"
    assert result.output_dir == target
    assert result.command == (sys.executable, "workload.py")
    assert result.trace == {"launches": [{"index": 0}]}
    assert result.launches == [{"index": 0}]
    assert result.trace_json.parent == target
    assert result.perfetto.parent == target
    assert result.html.parent == target


@pytest.mark.parametrize("postprocess", ["perfetto", "json", "html", "none", "all"])
def test_postprocess_artifact_contract(postprocess, tmp_path, monkeypatch):
    target = tmp_path / postprocess
    captured = {}
    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: "run-iket")
    monkeypatch.setattr(iket, "_run_process", _fake_profile_process(captured))

    result = iket.profile(
        ("python", "workload.py"), output_dir=target, postprocess=postprocess, timeout=None
    )
    assert result.postprocess == postprocess
    assert captured["timeout"] is None

    requested = iket._requested_formats(postprocess)  # pylint: disable=protected-access
    properties = {
        "json": lambda: result.trace_json,
        "perfetto": lambda: result.perfetto,
        "html": lambda: result.html,
    }
    for format_name, getter in properties.items():
        if format_name in requested:
            assert getter().is_file()
        else:
            with pytest.raises(iket.IketProfileError, match="was not requested"):
                getter()


def test_multi_process_singular_properties_point_to_plural(tmp_path, monkeypatch):
    target = tmp_path / "multi"
    captured = {}
    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: "run-iket")
    monkeypatch.setattr(iket, "_run_process", _fake_profile_process(captured, artifact_count=2))
    result = iket.profile(("python", "workload.py"), output_dir=target)

    assert len(result.json_traces) == 2
    assert len(result.perfetto_traces) == 2
    assert len(result.html_reports) == 2
    with pytest.raises(iket.IketProfileError, match="json_traces"):
        _ = result.trace_json
    with pytest.raises(iket.IketProfileError, match="perfetto_traces"):
        _ = result.perfetto
    with pytest.raises(iket.IketProfileError, match="html_reports"):
        _ = result.html


def test_missing_artifact_rolls_back_existing_output(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.mkdir()
    sentinel = target / "old.txt"
    sentinel.write_text("old", encoding="utf-8")

    def incomplete(argv, **_kwargs):
        staging = Path(argv[argv.index("--output-dir") + 1])
        (staging / "iket_pid_1.trace.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: "run-iket")
    monkeypatch.setattr(iket, "_run_process", incomplete)
    with pytest.raises(iket.IketProfileError, match="Perfetto trace"):
        iket.profile(("python", "workload.py"), output_dir=target, clobber=True)

    assert sentinel.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".result.staging-*"))


def test_successful_clobber_replaces_only_after_profile_success(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: "run-iket")
    monkeypatch.setattr(iket, "_run_process", _fake_profile_process(captured))

    result = iket.profile(("python", "workload.py"), output_dir=target, clobber=True)
    assert not (target / "old.txt").exists()
    assert result.trace_json.is_file()
    assert not list(tmp_path.glob(".result.backup-*"))


def test_existing_output_fails_before_installation_validation(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.mkdir()
    called = False

    def validate(_name):
        nonlocal called
        called = True

    monkeypatch.setattr(iket, "_validate_official_installation", validate)
    with pytest.raises(FileExistsError, match="already exists"):
        iket.profile(("python", "workload.py"), output_dir=target)
    assert not called


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_validation(timeout, tmp_path, monkeypatch):
    called = False

    def validate(_name):
        nonlocal called
        called = True

    monkeypatch.setattr(iket, "_validate_official_installation", validate)
    with pytest.raises(ValueError, match="timeout"):
        iket.profile(("python", "workload.py"), output_dir=tmp_path / "result", timeout=timeout)
    assert not called


def test_command_must_be_an_argument_sequence(tmp_path):
    with pytest.raises(TypeError, match="shell command string"):
        iket.profile("python workload.py", output_dir=tmp_path / "result")


def _make_executable(path: Path, source: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(source), encoding="utf-8")
    path.chmod(0o755)
    return path


def test_nonzero_exit_keeps_last_100_output_lines(tmp_path, monkeypatch):
    executable = _make_executable(
        tmp_path / "run-iket",
        """
        import sys
        for index in range(105):
            print(f"line-{index}", flush=True)
        sys.exit(7)
        """,
    )
    target = tmp_path / "result"
    target.mkdir()
    sentinel = target / "old.txt"
    sentinel.write_text("old", encoding="utf-8")
    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: str(executable))

    with pytest.raises(iket.IketProfileError) as error_info:
        iket.profile(
            ("python", "workload.py"),
            output_dir=target,
            postprocess="none",
            clobber=True,
            keep=True,
        )
    error = error_info.value
    assert error.returncode == 7
    assert len(error.output_tail.splitlines()) == 100
    assert error.output_tail.splitlines()[0] == "line-5"
    assert error.output_tail.splitlines()[-1] == "line-104"
    assert sentinel.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".result.staging-*"))


def _pid_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    try:
        state = stat_path.read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


def test_timeout_kills_run_iket_workload_and_grandchild(tmp_path, monkeypatch):
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    executable = _make_executable(
        tmp_path / "run-iket",
        """
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        Path(os.environ["PARENT_PID_FILE"]).write_text(str(os.getpid()))
        grandchild_source = """
        + repr(
            textwrap.dedent(
                """
                import os
                import signal
                import time
                from pathlib import Path
                Path(os.environ["GRANDCHILD_PID_FILE"]).write_text(str(os.getpid()))
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                while True:
                    time.sleep(1)
                """
            )
        )
        + "\n"
        + """
        child_source = "import os, signal, subprocess, sys, time; " \\
            "from pathlib import Path; " \\
            "Path(os.environ['CHILD_PID_FILE']).write_text(str(os.getpid())); " \\
            f"subprocess.Popen([sys.executable, '-c', {grandchild_source!r}]); " \\
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); " \\
            "time.sleep(3600)"
        child = subprocess.Popen([sys.executable, "-c", child_source])
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        deadline = time.monotonic() + 5
        while not Path(os.environ["GRANDCHILD_PID_FILE"]).exists():
            if time.monotonic() > deadline:
                raise RuntimeError("grandchild did not start")
            time.sleep(0.01)
        print("fake run-iket ready", flush=True)
        time.sleep(3600)
        """,
    )
    monkeypatch.setattr(iket, "_validate_official_installation", lambda _name: str(executable))
    monkeypatch.setattr(iket, "_TERMINATION_GRACE_SECONDS", 0.2)

    with pytest.raises(iket.IketProfileError) as error_info:
        iket.profile(
            ("python", "workload.py"),
            output_dir=tmp_path / "result",
            postprocess="none",
            env={
                "PARENT_PID_FILE": str(parent_pid),
                "CHILD_PID_FILE": str(child_pid),
                "GRANDCHILD_PID_FILE": str(grandchild_pid),
            },
            timeout=1.0,
            keep=True,
        )
    error = error_info.value
    assert error.returncode is None
    assert error.timeout == 1.0
    assert "fake run-iket ready" in error.output_tail

    pids = [
        int(path.read_text(encoding="utf-8")) for path in (parent_pid, child_pid, grandchild_pid)
    ]
    deadline = time.monotonic() + 2
    while any(_pid_is_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(_pid_is_running(pid) for pid in pids)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.staging-*"))


def test_injected_run_preserves_main_exit_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "injected")
    monkeypatch.setattr(iket, "_validate_official_environment", lambda: None)

    with pytest.raises(SystemExit) as normal_exit:
        iket.run(lambda: None, output_dir=tmp_path / "normal")
    assert normal_exit.value.code == 0

    def fail():
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        iket.run(fail, output_dir=tmp_path / "failed")

    def explicit_exit():
        raise SystemExit(9)

    with pytest.raises(SystemExit) as nonzero_exit:
        iket.run(explicit_exit, output_dir=tmp_path / "nonzero")
    assert nonzero_exit.value.code == 9

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        iket.run(interrupt, output_dir=tmp_path / "interrupt")


def test_imported_callable_is_rejected_before_profile_starts(tmp_path, monkeypatch):
    started = False

    def should_not_start(*_args, **_kwargs):
        nonlocal started
        started = True

    monkeypatch.delenv("CUDA_INJECTION64_PATH", raising=False)
    monkeypatch.delenv("SMODEL_INJECTION_CONFIG", raising=False)
    monkeypatch.setattr(iket, "profile", should_not_start)
    with pytest.raises(ValueError, match=r"iket\.profile\(command\)"):
        iket.run(test_imported_callable_is_rejected_before_profile_starts, output_dir=tmp_path)
    assert not started


def _artificial_main(source_path: Path):
    module = types.ModuleType("__main__")
    module.__file__ = str(source_path)
    exec(compile("def entry():\n    return None\n", str(source_path), "exec"), module.__dict__)
    return module


def test_replay_guard_accepts_matching_script_file(tmp_path, monkeypatch):
    source_path = tmp_path / "workload.py"
    source_path.write_text("def entry():\n    return None\n", encoding="utf-8")
    main_module = _artificial_main(source_path)
    monkeypatch.setitem(sys.modules, "__main__", main_module)
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, str(source_path), "--size", "4"])

    assert iket._replay_command(main_module.entry) == (  # pylint: disable=protected-access
        sys.executable,
        str(source_path),
        "--size",
        "4",
    )


def test_replay_guard_accepts_matching_python_module(tmp_path, monkeypatch):
    source_path = tmp_path / "workload.py"
    source_path.write_text("def entry():\n    return None\n", encoding="utf-8")
    main_module = _artificial_main(source_path)
    main_module.__spec__ = importlib.machinery.ModuleSpec(
        "package.workload", loader=None, origin=str(source_path)
    )
    monkeypatch.setitem(sys.modules, "__main__", main_module)
    monkeypatch.setattr(
        sys, "orig_argv", [sys.executable, "-X", "dev", "-m", "package.workload", "--size", "4"]
    )

    assert iket._replay_command(main_module.entry) == (  # pylint: disable=protected-access
        sys.executable,
        "-X",
        "dev",
        "-m",
        "package.workload",
        "--size",
        "4",
    )


def test_replay_guard_rejects_file_mismatch(tmp_path, monkeypatch):
    source_path = tmp_path / "workload.py"
    other_path = tmp_path / "other.py"
    source_path.write_text("def entry():\n    return None\n", encoding="utf-8")
    other_path.write_text("pass\n", encoding="utf-8")
    main_module = _artificial_main(source_path)
    monkeypatch.setitem(sys.modules, "__main__", main_module)
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, str(other_path)])

    with pytest.raises(ValueError, match="different files"):
        iket._replay_command(main_module.entry)  # pylint: disable=protected-access
