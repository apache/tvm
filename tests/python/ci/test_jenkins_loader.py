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
"""Exercise the native helper trust boundary and its isolated Git fetch."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .test_utils import REPO_ROOT, TempGit

PIPELINES = sorted((REPO_ROOT / "ci/jenkins/generated").glob("*_jenkinsfile.groovy"))
HELPER = "ci/jenkins/generated/common.groovy"


def test_native_loader_trust_boundary():
    groovy = shutil.which("groovy")
    if groovy is None:
        pytest.skip("Groovy is required for the executable pipeline loader checks")
    subprocess.run(
        [groovy, str(Path(__file__).with_name("jenkins_loader_test.groovy")), *PIPELINES],
        check=True,
    )


@pytest.mark.parametrize("failure", [None, "fetch", "missing-helper"])
def test_trusted_main_fetch(tmp_path, failure):
    # Execute the actual loader shell against a local upstream. A Git shim only
    # substitutes the transport URL; ref selection and object reads use real Git.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    repo = TempGit(upstream)
    repo.run("config", "user.name", "CI Test")
    repo.run("config", "user.email", "ci-test@example.com")
    helper = upstream / HELPER
    helper.parent.mkdir(parents=True)
    helper.write_text("old main helper")
    repo.run("add", ".")
    repo.run("commit", "-m", "initial main")
    repo.run("branch", "release")
    helper.write_text("latest main helper")
    repo.run("commit", "-am", "advance main")
    if failure == "missing-helper":
        helper.unlink()
        repo.run("commit", "-am", "remove helper")

    checkout = tmp_path / "untrusted-checkout"
    checkout.mkdir()
    untrusted_helper = checkout / HELPER
    untrusted_helper.parent.mkdir(parents=True)
    untrusted_helper.write_text("rejected PR helper")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    git = binary_dir / "git"
    git.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "url = 'https://github.com/apache/tvm.git'\n"
        "if 'fetch' in args:\n"
        "    assert url in args\n"
        "    assert '+refs/heads/main:refs/heads/trusted-main' in args\n"
        "    if os.environ.get('FAIL_FETCH'): sys.exit(7)\n"
        "    args[args.index(url)] = os.environ['TEST_UPSTREAM']\n"
        "os.execv(os.environ['REAL_GIT'], ['git'] + args)\n"
    )
    git.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{binary_dir}:{os.environ['PATH']}",
        REAL_GIT=shutil.which("git"),
        TEST_UPSTREAM=str(upstream),
        FAIL_FETCH="yes" if failure == "fetch" else "",
        CHANGE_TARGET="release",
    )
    scripts = {
        p.read_text().split("label: 'Read trusted helper from upstream main',")[1].split("'" * 3)[1]
        for p in PIPELINES
    }
    assert len(scripts) == 1
    result = subprocess.run(
        ["sh", "-c", scripts.pop()], cwd=checkout, env=env, text=True, capture_output=True
    )
    if failure:
        assert result.returncode != 0
        assert result.stdout == ""
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "latest main helper"
    assert untrusted_helper.read_text() == "rejected PR helper"
