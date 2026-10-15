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
"""Host-side Jenkins commands preserve retries, paths and build configuration."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from .test_utils import JENKINS_SCRIPT_ROOT, REPO_ROOT, TempGit


def load_s3(script_root=JENKINS_SCRIPT_ROOT):
    spec = importlib.util.spec_from_file_location("jenkins_s3", script_root / "s3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executable(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("failures", [2, 3])
def test_s3_transfer_retries(tmp_path, monkeypatch, failures):
    script_root = JENKINS_SCRIPT_ROOT
    # Run the real retry.sh with fake AWS and sleep, without network or delays.
    executable(
        tmp_path / "aws",
        """
import json, os, pathlib, sys
log = pathlib.Path(os.environ["AWS_LOG"])
with log.open("a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
attempt = len(log.read_text().splitlines())
print("download: s3://bucket/file to file name")
sys.exit(1 if attempt <= int(os.environ["FAILURES"]) else 0)
""",
    )
    executable(
        tmp_path / "sleep",
        """
import os, pathlib, sys
with pathlib.Path(os.environ["SLEEP_LOG"]).open("a") as out:
    out.write(sys.argv[1] + "\\n")
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("FAILURES", str(failures))
    monkeypatch.setenv("SLEEP_LOG", str(tmp_path / "sleeps"))
    monkeypatch.syspath_prepend(str(script_root))
    destination = "file name"
    for transfer in range(2):
        log = tmp_path / f"aws-{transfer}"
        monkeypatch.setenv("AWS_LOG", str(log))
        s3 = load_s3(script_root)
        if failures == 3:
            with pytest.raises(subprocess.CalledProcessError):
                s3.s3("s3://bucket/file", destination, False)
        else:
            assert s3.parse_output_files(s3.s3("s3://bucket/file", destination, False))
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        assert commands == [["s3", "cp", "--no-progress", "s3://bucket/file", destination]] * 3
    delays = (tmp_path / "sleeps").read_text().splitlines()
    assert len(delays) == 4
    assert all(30 <= int(delay) <= 200 for delay in delays)


def test_build_environment_and_target(tmp_path):
    # Run the entry point in a minimal checkout, capturing the tool boundary.
    script = tmp_path / "tests/scripts/task_build.py"
    script.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tests/scripts/task_build.py", script)
    (tmp_path / "build").mkdir()
    ffi_path = tmp_path / "3rdparty/tvm-ffi"
    ffi_path.mkdir(parents=True)
    ffi_repo = TempGit(ffi_path)
    ffi_repo.run("config", "user.name", "CI Test")
    ffi_repo.run("config", "user.email", "ci-test@example.com")
    ffi_repo.run("commit", "--allow-empty", "-m", "FFI fixture")
    ffi_commit = ffi_repo.run("rev-parse", "HEAD", stdout=subprocess.PIPE).stdout.strip()
    ffi_marker = tmp_path / "python/tvm_ffi/.source-commit"
    ffi_marker.parent.mkdir(parents=True)
    ffi_marker.write_text(f"{ffi_commit} {sys.implementation.cache_tag}\n")
    executable(
        tmp_path / "cmake",
        """
import json, os, pathlib, sys
with pathlib.Path(os.environ["COMMAND_LOG"]).open("a") as out:
    entry = [sys.argv[1:], os.getcwd(), os.environ["SCCACHE_BUCKET"], os.environ["CC"]]
    out.write(json.dumps(entry) + "\\n")
""",
    )
    executable(tmp_path / "sccache", "pass\n")
    log = tmp_path / "commands"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        COMMAND_LOG=str(log),
        AWS_ACCESS_KEY_ID="test",
        CI_NUM_EXECUTORS="2",
        VERBOSE="false",
    )
    subprocess.run(
        [
            sys.executable,
            script,
            "--debug",
            "--cmake-target",
            "cpptest",
            "--sccache-bucket",
            "test-cache",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    configure, build = [json.loads(line) for line in log.read_text().splitlines()]
    assert configure == [
        ["-GNinja", "-DCMAKE_BUILD_TYPE=Debug", ".."],
        str(tmp_path / "build"),
        "test-cache",
        "/opt/sccache/cc",
    ]
    assert build[0][:4] == ["--build", ".", "--", "cpptest"]
    assert build[0][4].startswith("-j") and int(build[0][4][2:]) >= 1
    assert len(build[0]) == 5
    assert build[1:] == configure[1:]


@pytest.fixture
def artifact_cli(tmp_path):
    script_root = tmp_path / "ci/jenkins/scripts"
    script_root.mkdir(parents=True)
    for name in ("s3.py", "retry.sh"):
        shutil.copy(JENKINS_SCRIPT_ROOT / name, script_root / name)
    registry = {
        "wheels": ["build/dist/*.whl", "build/dist/a.whl", "./build/dist/a.whl"],
        "extra": ["build/exact.so", "build/dir", "build/missing.so", "build/no-match/*.so"],
    }
    (script_root.parent / "artifacts.py").write_text("ARTIFACTS = " + repr(registry))
    for name in (
        "build/dist/b space.whl",
        "build/dist/a.whl",
        "build/exact.so",
        "build/dir/nested",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    executable(
        binary_dir / "aws",
        """
import json, os, pathlib, sys
with pathlib.Path(os.environ["AWS_LOG"]).open("a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[-1] == ".":
    pathlib.Path("downloaded file").write_text("downloaded")
    print("download: s3://bucket/prefix/downloaded file to downloaded file")
""",
    )
    log = tmp_path / "commands"
    env = dict(os.environ, PATH=f"{binary_dir}:{os.environ['PATH']}", AWS_LOG=str(log))

    def run(*options, action="upload"):
        log.unlink(missing_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                script_root / "s3.py",
                "--action",
                action,
                "--bucket",
                "bucket",
                "--prefix",
                "/prefix/",
                *options,
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
        )
        commands = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        return result, commands

    return run


def test_artifact_upload_aliases_and_globs(artifact_cli):
    expected = [
        ["s3", "cp", "--no-progress", "--recursive", "build/dir", "s3://bucket/prefix/build/dir"],
        ["s3", "cp", "--no-progress", "build/dist/a.whl", "s3://bucket/prefix/build/dist/a.whl"],
        [
            "s3",
            "cp",
            "--no-progress",
            "build/dist/b space.whl",
            "s3://bucket/prefix/build/dist/b space.whl",
        ],
        ["s3", "cp", "--no-progress", "build/exact.so", "s3://bucket/prefix/build/exact.so"],
    ]
    for flags in (
        ["--artifacts", "wheels", "extra"],
        ["--artifacts", "extra", "--artifacts", "wheels"],
        ["--bundle", "wheels", "--bundle", "extra"],
        ["--bundle", "extra", "wheels"],
    ):
        result, commands = artifact_cli(*flags)
        assert result.returncode == 0, result.stderr
        assert commands == expected
        assert "No artifacts matched pattern: build/no-match/*.so" in result.stderr
        assert "The path doesn't exist: build/missing.so" in result.stderr


def test_artifact_paths_alias_and_unmatched_pattern(artifact_cli):
    for flag in ("--paths", "--items"):
        result, commands = artifact_cli(flag, "build/dist/*.whl", "./build/dist/a.whl")
        assert result.returncode == 0, result.stderr
        assert [command[-2] for command in commands] == [
            "build/dist/a.whl",
            "build/dist/b space.whl",
        ]
        result, commands = artifact_cli(flag, "absent/*.whl")
        assert result.returncode == 0
        assert commands == []
        assert "No artifacts matched pattern: absent/*.whl" in result.stderr


@pytest.mark.parametrize("artifacts_flag", ["--artifacts", "--bundle"])
@pytest.mark.parametrize("paths_flag", ["--paths", "--items"])
def test_artifact_flags_are_mutually_exclusive(artifact_cli, artifacts_flag, paths_flag):
    result, commands = artifact_cli(artifacts_flag, "wheels", paths_flag, "build/exact.so")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert commands == []


def test_artifact_download_copies_whole_prefix(artifact_cli, tmp_path):
    result, commands = artifact_cli(action="download")
    assert result.returncode == 0, result.stderr
    assert commands == [["s3", "cp", "--no-progress", "--recursive", "s3://bucket/prefix", "."]]
    downloaded = tmp_path / "downloaded file"
    assert downloaded.read_text() == "downloaded"
    assert downloaded.stat().st_mode & 0o111


@pytest.mark.parametrize("absolute", [False, True])
def test_legacy_artifact_literal_paths(artifact_cli, tmp_path, absolute):
    outside = tmp_path.parent / "outside.so"
    outside.write_text("artifact")
    path = str(outside) if absolute else "../outside.so"
    result, commands = artifact_cli("--items", path)
    assert result.returncode == 0, result.stderr
    assert commands == [["s3", "cp", "--no-progress", path, "s3://bucket/prefix/" + path]]


def test_recursive_artifact_overlap(artifact_cli):
    result, commands = artifact_cli("--paths", "build/dir", "build/dir/*", "build/dir/nested")
    assert result.returncode == 0, result.stderr
    assert commands == [
        ["s3", "cp", "--no-progress", "--recursive", "build/dir", "s3://bucket/prefix/build/dir"]
    ]


def test_artifact_empty_directory_and_unknown_name(artifact_cli, tmp_path):
    (tmp_path / "empty").mkdir()
    result, commands = artifact_cli("--paths", "empty")
    assert result.returncode != 0
    assert "Cannot upload empty folder" in result.stderr
    assert commands == []
    result, commands = artifact_cli("--artifacts", "unknown")
    assert result.returncode != 0
    assert "Unknown artifact" in result.stderr
    assert commands == []


def test_current_pipeline_artifact_options(artifact_cli, tmp_path):
    # c14bc80 passes repeated --bundle names; the new Jenkinsfiles use --artifacts.
    registry = load_s3().load_artifacts()
    (tmp_path / "ci/jenkins/artifacts.py").write_text("ARTIFACTS = " + repr(registry))
    for entries in registry.values():
        for item in entries:
            path = tmp_path / item
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact")
    for names in (
        ["tvm_lib", "cpptest"],
        ["tvm_lib", "tvm_lib_gpu_extra", "cpptest"],
        ["tvm_runtime"],
    ):
        old = [option for name in names for option in ("--bundle", name)]
        old_result, old_commands = artifact_cli(*old)
        new_result, new_commands = artifact_cli("--artifacts", *names)
        assert old_result.returncode == new_result.returncode == 0
        assert old_commands == new_commands
        assert [command[-2] for command in new_commands] == sorted(
            {item for name in names for item in registry[name]}
        )
