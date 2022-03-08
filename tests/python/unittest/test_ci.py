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
import textwrap
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


def test_update_branch(tmpdir_factory):
    update_script = REPO_ROOT / "tests" / "scripts" / "update_branch.py"

    def run(statuses, expected_rc, expected_output):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
        commit = {
            "statusCheckRollup": {"contexts": {"nodes": statuses}},
            "oid": "123",
            "messageHeadline": "hello",
        }
        data = {
            "data": {
                "repository": {
                    "defaultBranchRef": {"target": {"history": {"edges": [], "nodes": [commit]}}}
                }
            }
        }
        proc = subprocess.run(
            [str(update_script), "--dry-run", "--testonly-json", json.dumps(data)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            cwd=git.cwd,
        )

        if proc.returncode != expected_rc:
            raise RuntimeError(
                f"Wrong return code:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        if expected_output not in proc.stdout:
            raise RuntimeError(
                f"Missing {expected_output}:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

    # Missing expected tvm-ci/branch test
    run(
        statuses=[
            {
                "context": "test",
                "state": "SUCCESS",
            }
        ],
        expected_rc=1,
        expected_output="No good commits found in the last 1 commits",
    )

    # Only has the right passing test
    run(
        statuses=[
            {
                "context": "tvm-ci/branch",
                "state": "SUCCESS",
            }
        ],
        expected_rc=0,
        expected_output="Found last good commit: 123: hello",
    )

    # Check with many statuses
    run(
        statuses=[
            {
                "context": "tvm-ci/branch",
                "state": "SUCCESS",
            },
            {
                "context": "tvm-ci/branch2",
                "state": "SUCCESS",
            },
            {
                "context": "tvm-ci/branch3",
                "state": "FAILED",
            },
        ],
        expected_rc=1,
        expected_output="No good commits found in the last 1 commits",
    )
    run(
        statuses=[
            {
                "context": "tvm-ci/branch",
                "state": "SUCCESS",
            },
            {
                "context": "tvm-ci/branch2",
                "state": "SUCCESS",
            },
            {
                "context": "tvm-ci/branch3",
                "state": "SUCCESS",
            },
        ],
        expected_rc=0,
        expected_output="Found last good commit: 123: hello",
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
            [str(skip_ci_script), "--pr", pr_number, "--pr-title", pr_title], cwd=git.cwd
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


def test_skip_globs(tmpdir_factory):
    script = REPO_ROOT / "tests" / "scripts" / "git_skip_ci_globs.py"

    def run(files, should_skip):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        # Jenkins git is too old and doesn't have 'git init --initial-branch'
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

        proc = subprocess.run(
            [
                str(script),
                "--files",
                ",".join(files),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            cwd=git.cwd,
        )

        if should_skip:
            assert proc.returncode == 0
        else:
            assert proc.returncode == 1

    run([], should_skip=True)
    run(["README.md"], should_skip=True)
    run(["test.c"], should_skip=False)
    run(["test.c", "README.md"], should_skip=False)
    run(["src/autotvm/feature_visitor.cc", "README.md"], should_skip=False)
    run([".asf.yaml", "docs/README.md"], should_skip=True)


def test_ping_reviewers(tmpdir_factory):
    reviewers_script = REPO_ROOT / "tests" / "scripts" / "ping_reviewers.py"

    def run(pr, check):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        # Jenkins git is too old and doesn't have 'git init --initial-branch'
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

        data = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "nodes": [pr],
                        "edges": [],
                    }
                }
            }
        }
        proc = subprocess.run(
            [
                str(reviewers_script),
                "--dry-run",
                "--wait-time-minutes",
                "1",
                "--cutoff-pr-number",
                "5",
                "--allowlist",
                "user",
                "--pr-json",
                json.dumps(data),
                "--now",
                "2022-01-26T17:54:19Z",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            cwd=git.cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        assert check in proc.stdout

    def all_time_keys(time):
        return {
            "updatedAt": time,
            "lastEditedAt": time,
            "createdAt": time,
            "publishedAt": time,
        }

    run(
        {
            "isDraft": True,
            "number": 2,
        },
        "Checking 0 of 1 fetched",
    )

    run(
        {
            "isDraft": False,
            "number": 2,
        },
        "Checking 0 of 1 fetched",
    )

    run(
        {
            "number": 123,
            "url": "https://github.com/apache/tvm/pull/123",
            "body": "cc @someone",
            "isDraft": False,
            "author": {"login": "user"},
            "reviews": {"nodes": []},
            **all_time_keys("2022-01-18T17:54:19Z"),
            "comments": {"nodes": []},
        },
        "Pinging reviewers ['someone'] on https://github.com/apache/tvm/pull/123",
    )

    # Check allowlist functionality
    run(
        {
            "number": 123,
            "url": "https://github.com/apache/tvm/pull/123",
            "body": "cc @someone",
            "isDraft": False,
            "author": {"login": "user2"},
            "reviews": {"nodes": []},
            **all_time_keys("2022-01-18T17:54:19Z"),
            "comments": {
                "nodes": [
                    {**all_time_keys("2022-01-19T17:54:19Z"), "bodyText": "abc"},
                ]
            },
        },
        "Checking 0 of 1 fetched",
    )

    # Old comment, ping
    run(
        {
            "number": 123,
            "url": "https://github.com/apache/tvm/pull/123",
            "body": "cc @someone",
            "isDraft": False,
            "author": {"login": "user"},
            "reviews": {"nodes": []},
            **all_time_keys("2022-01-18T17:54:19Z"),
            "comments": {
                "nodes": [
                    {
                        **all_time_keys("2022-01-18T17:54:19Z"),
                        "bodyText": "abc",
                    },
                ]
            },
        },
        "Pinging reviewers ['someone'] on https://github.com/apache/tvm/pull/123",
    )

    # New comment, don't ping
    run(
        {
            "number": 123,
            "url": "https://github.com/apache/tvm/pull/123",
            "body": "cc @someone",
            "isDraft": False,
            "author": {"login": "user"},
            "reviews": {"nodes": []},
            **all_time_keys("2022-01-18T17:54:19Z"),
            "comments": {
                "nodes": [
                    {**all_time_keys("2022-01-27T17:54:19Z"), "bodyText": "abc"},
                ]
            },
        },
        "Not pinging PR 123",
    )


def assert_in(needle: str, haystack: str):
    if needle not in haystack:
        raise AssertionError(f"item not found:\n{needle}\nin:\n{haystack}")


def test_github_tag_teams(tmpdir_factory):
    tag_script = REPO_ROOT / "tests" / "scripts" / "github_tag_teams.py"

    def run(type, data, check):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

        issue_body = """
        some text
        [temporary] opt-in: @person5

        - something: @person1 @person2
        - something else @person1 @person2
        - something else2: @person1 @person2
        - something-else @person1 @person2
        """
        comment1 = """
        another thing: @person3
        another-thing @person3
        """
        comment2 = """
        something @person4
        """
        teams = {
            "data": {
                "repository": {
                    "issue": {
                        "body": issue_body,
                        "comments": {"nodes": [{"body": comment1}, {"body": comment2}]},
                    }
                }
            }
        }
        env = {
            type: json.dumps(data),
        }
        proc = subprocess.run(
            [
                str(tag_script),
                "--dry-run",
                "--team-issue-json",
                json.dumps(teams),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            cwd=git.cwd,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        assert_in(check, proc.stdout)

    run(
        "ISSUE",
        {
            "title": "A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "abc"}],
            "body": textwrap.dedent(
                """
            hello
            """.strip()
            ),
        },
        "No one to cc, exiting",
    )

    run(
        "ISSUE",
        {
            "title": "A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "abc"}],
            "body": textwrap.dedent(
                """
            hello

            cc @test
            """.strip()
            ),
        },
        "No one to cc, exiting",
    )

    run(
        type="ISSUE",
        data={
            "title": "A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something"}],
            "body": textwrap.dedent(
                """
                hello

                something"""
            ),
        },
        check="would have updated issues/1234 with {'body': '\\nhello\\n\\nsomething\\n\\ncc @person1 @person2 @person4'}",
    )

    run(
        type="ISSUE",
        data={
            "title": "A title",
            "number": 1234,
            "user": {
                "login": "person6",
            },
            "labels": [{"name": "something"}],
            "body": textwrap.dedent(
                """
                hello

                something"""
            ),
        },
        check="Author person6 is not opted in, quitting",
    )

    run(
        type="ISSUE",
        data={
            "title": "A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something"}],
            "body": textwrap.dedent(
                """
                hello

                cc @person1 @person2 @person4"""
            ),
        },
        check="Everyone to cc is already cc'ed, no update needed",
    )

    run(
        type="ISSUE",
        data={
            "title": "[something] A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": textwrap.dedent(
                """
                hello

                something"""
            ),
        },
        check="would have updated issues/1234 with {'body': '\\nhello\\n\\nsomething\\n\\ncc @person1 @person2 @person4'}",
    )

    run(
        type="ISSUE",
        data={
            "title": "[something] A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": textwrap.dedent(
                """
                hello

                cc @person1 @person2 @person4"""
            ),
        },
        check="Everyone to cc is already cc'ed, no update needed",
    )

    run(
        type="PR",
        data={
            "title": "[something] A title",
            "number": 1234,
            "draft": False,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": textwrap.dedent(
                """
                hello

                cc @person1 @person2 @person4"""
            ),
        },
        check="Everyone to cc is already cc'ed, no update needed",
    )

    run(
        type="PR",
        data={
            "title": "[something] A title",
            "number": 1234,
            "draft": True,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": textwrap.dedent(
                """
                hello

                cc @person1 @person2 @person4"""
            ),
        },
        check="Terminating since 1234 is a draft",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
