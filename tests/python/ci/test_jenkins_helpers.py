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

from .test_utils import JENKINS_SCRIPT_ROOT, REPO_ROOT


def load_s3():
    spec = importlib.util.spec_from_file_location("jenkins_s3", JENKINS_SCRIPT_ROOT / "s3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executable(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize("failures", [2, 3])
def test_s3_transfer_retries(tmp_path, monkeypatch, failures):
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
    s3 = load_s3()
    for transfer in range(2):
        log = tmp_path / f"aws-{transfer}"
        monkeypatch.setenv("AWS_LOG", str(log))
        if failures == 3:
            with pytest.raises(subprocess.CalledProcessError):
                s3.s3("s3://bucket/file", "file name", False)
        else:
            assert s3.parse_output_files(s3.s3("s3://bucket/file", "file name", False))
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        assert commands == [["s3", "cp", "--no-progress", "s3://bucket/file", "file name"]] * 3
    delays = (tmp_path / "sleeps").read_text().splitlines()
    assert len(delays) == 4
    assert all(30 <= int(delay) <= 200 for delay in delays)


def test_build_environment_and_target(tmp_path):
    # Run the entry point in a minimal checkout, capturing the tool boundary.
    script = tmp_path / "tests/scripts/task_build.py"
    script.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tests/scripts/task_build.py", script)
    (tmp_path / "build").mkdir()
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
