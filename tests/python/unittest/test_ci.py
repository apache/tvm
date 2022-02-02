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

import pathlib
import subprocess
import sys
import json
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class TempGit:
    def __init__(self, cwd):
        self.cwd = cwd

    def run(self, *args):
        proc = subprocess.run(["git"] + list(args), cwd=self.cwd)
        if proc.returncode != 0:
            raise RuntimeError(f"git command failed: '{args}'")


def test_cc_reviewers(tmpdir_factory):
    reviewers_script = REPO_ROOT / "tests" / "scripts" / "github_cc_reviewers.py"

    def run(pr_body, expected_reviewers):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
        proc = subprocess.run(
            [str(reviewers_script), "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PR": json.dumps({"number": 1, "body": pr_body})},
            encoding="utf-8",
            cwd=git.cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        assert proc.stdout.strip().endswith(f"Adding reviewers: {expected_reviewers}")

    run(pr_body="abc", expected_reviewers=[])
    run(pr_body="cc @abc", expected_reviewers=["abc"])
    run(pr_body="cc @", expected_reviewers=[])
    run(pr_body="cc @abc @def", expected_reviewers=["abc", "def"])
    run(pr_body="some text cc @abc @def something else", expected_reviewers=["abc", "def"])
    run(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        expected_reviewers=["abc", "def", "zzz"],
    )


def test_skip_ci(tmpdir_factory):
    skip_ci_script = REPO_ROOT / "tests" / "scripts" / "git_skip_ci.py"

    def test(commands, should_skip, pr_title, why):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        # Jenkins git is too old and doesn't have 'git init --initial-branch'
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
        git.run("config", "user.name", "ci")
        git.run("config", "user.email", "email@example.com")
        git.run("commit", "--allow-empty", "--message", "base commit")
        for command in commands:
            git.run(*command)
        pr_number = "1234"
        proc = subprocess.run(
            [str(skip_ci_script), "--pr", pr_number, "--pr-title", pr_title], cwd=git.cwd  # dir
        )
        expected = 0 if should_skip else 1
        assert proc.returncode == expected, why

    test(
        commands=[],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped",
    )

    test(
        commands=[
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    )

    test(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped on a branch with [skip ci] in the last commit",
    )

    test(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[no skip ci] test",
        why="ci should not be skipped on a branch with [skip ci] in the last commit but not the PR title",
    )

    test(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on a branch without [skip ci] in the last commit",
    )

    test(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on a branch without [skip ci] in the last commit",
    )

    test(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
            ["commit", "--allow-empty", "--message", "commit 3"],
            ["commit", "--allow-empty", "--message", "commit 4"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on a branch without [skip ci] in the last commit",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
