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

import subprocess
import json
import sys
import pytest

from pathlib import Path

from test_utils import REPO_ROOT


class TempGit:
    def __init__(self, cwd):
        self.cwd = cwd

    def run(self, *args):
        proc = subprocess.run(["git"] + list(args), cwd=self.cwd)
        if proc.returncode != 0:
            raise RuntimeError(f"git command failed: '{args}'")


def test_mergebot(tmpdir_factory):
    mergebot_script = REPO_ROOT / "tests" / "scripts" / "github_mergebot.py"
    test_json_dir = Path(__file__).resolve().parent / "sample_prs"

    def run(number, filename, expected):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
        with open(test_json_dir / filename) as f:
            test_data = json.load(f)

        proc = subprocess.run(
            [
                str(mergebot_script),
                "--pr",
                str(number),
                "--dry-run",
                "--testing-pr-json",
                json.dumps(test_data),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            cwd=git.cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        if expected not in proc.stderr:
            raise RuntimeError(f"{proc.stderr}\ndid not contain\n{expected}")

    run(
        number=10786,
        filename="pr10786-merges.json",
        expected="Dry run, would have merged with url=pulls/10786/merge",
    )
    run(
        number=10786,
        filename="pr10786-nottriggered.json",
        expected="No merge requested, exiting",
    )
    run(
        number=10786,
        filename="pr10786-badci.json",
        expected="Cannot merge, these CI jobs failed on",
    )
    run(
        number=10786,
        filename="pr10786-oldreview.json",
        expected="Cannot merge, did not find any approving reviews",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
