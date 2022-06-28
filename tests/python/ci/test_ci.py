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
"""Test various CI scripts and GitHub Actions workflows"""
import subprocess
import json
import textwrap
from pathlib import Path

import pytest
import tvm.testing
from .test_utils import REPO_ROOT, TempGit


def parameterize_named(*values):
    keys = list(values[0].keys())
    if len(keys) == 1:
        return pytest.mark.parametrize(",".join(keys), [d[keys[0]] for d in values])

    return pytest.mark.parametrize(",".join(keys), [tuple(d.values()) for d in values])


@pytest.mark.parametrize(
    "target_url,base_url,commit_sha,expected_url,expected_body",
    [
        (
            "https://ci.tlcpack.ai/job/tvm/job/PR-11594/3/display/redirect",
            "https://pr-docs.tlcpack.ai",
            "SHA",
            "issues/11594/comments",
            "Built docs for commit SHA can be found "
            "[here](https://pr-docs.tlcpack.ai/PR-11594/3/docs/index.html).",
        )
    ],
)
def test_docs_comment(
    tmpdir_factory, target_url, base_url, commit_sha, expected_url, expected_body
):
    """
    Test that a comment with a link to the docs is successfully left on PRs
    """
    docs_comment_script = REPO_ROOT / "tests" / "scripts" / "github_docs_comment.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("init")
    git.run("checkout", "-b", "main")
    git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
    proc = subprocess.run(
        [str(docs_comment_script), "--dry-run", f"--base-url-docs={base_url}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"TARGET_URL": target_url, "COMMIT_SHA": commit_sha},
        encoding="utf-8",
        cwd=git.cwd,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

    assert f"Dry run, would have posted {expected_url} with data {expected_body}." in proc.stderr


@tvm.testing.skip_if_wheel_test
def test_cc_reviewers(tmpdir_factory):
    """
    Test that reviewers are added from 'cc @someone' messages in PRs
    """
    reviewers_script = REPO_ROOT / "tests" / "scripts" / "github_cc_reviewers.py"

    def run(pr_body, requested_reviewers, existing_review_users, expected_reviewers):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
        reviews = [{"user": {"login": r}} for r in existing_review_users]
        requested_reviewers = [{"login": r} for r in requested_reviewers]
        proc = subprocess.run(
            [str(reviewers_script), "--dry-run", "--testing-reviews-json", json.dumps(reviews)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "PR": json.dumps(
                    {"number": 1, "body": pr_body, "requested_reviewers": requested_reviewers}
                )
            },
            encoding="utf-8",
            cwd=git.cwd,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        assert f"After filtering existing reviewers, adding: {expected_reviewers}" in proc.stdout

    run(pr_body="abc", requested_reviewers=[], existing_review_users=[], expected_reviewers=[])
    run(
        pr_body="cc @abc",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc"],
    )
    run(pr_body="cc @", requested_reviewers=[], existing_review_users=[], expected_reviewers=[])
    run(
        pr_body="cc @abc @def",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def"],
    )
    run(
        pr_body="some text cc @abc @def something else",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def"],
    )
    run(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def", "zzz"],
    )
    run(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=["abc"],
        existing_review_users=[],
        expected_reviewers=["def", "zzz"],
    )
    run(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=["abc"],
        existing_review_users=["abc"],
        expected_reviewers=["def", "zzz"],
    )
    run(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=[],
        existing_review_users=["abc"],
        expected_reviewers=["def", "zzz"],
    )


def test_update_branch(tmpdir_factory):
    """
    Test that the last-successful branch script updates successfully
    """
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
            check=False,
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


@parameterize_named(
    dict(
        commands=[],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    dict(
        commands=[
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped on a branch with [skip ci] in the last commit",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[no skip ci] test",
        why="ci should not be skipped on a branch with "
        "[skip ci] in the last commit but not the PR title",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
            ["commit", "--allow-empty", "--message", "commit 3"],
            ["commit", "--allow-empty", "--message", "commit 4"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
        ],
        should_skip=True,
        pr_title="[something][skip ci] test",
        why="skip ci tag should work anywhere in title",
    ),
)
def test_skip_ci(tmpdir_factory, commands, should_skip, pr_title, why):
    """
    Test that CI is skipped when it should be
    """
    skip_ci_script = REPO_ROOT / "tests" / "scripts" / "git_skip_ci.py"

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
        [str(skip_ci_script), "--pr", pr_number, "--pr-title", pr_title],
        cwd=git.cwd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=False,
    )
    expected = 0 if should_skip else 1
    if proc.returncode != expected:
        raise RuntimeError(
            f"Unexpected return code {proc.returncode} "
            f"(expected {expected}) in {why}:\n{proc.stdout}"
        )


def test_skip_globs(tmpdir_factory):
    """
    Test that CI is skipped if only certain files are edited
    """
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
            check=False,
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
    """
    Test that reviewers are messaged after a time period of inactivity
    """
    reviewers_script = REPO_ROOT / "tests" / "scripts" / "ping_reviewers.py"

    def run(pull_request, check):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        # Jenkins git is too old and doesn't have 'git init --initial-branch'
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

        data = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "nodes": [pull_request],
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
            check=False,
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
    """
    Check that 'needle' is in 'haystack'
    """
    if needle not in haystack:
        raise AssertionError(f"item not found:\n{needle}\nin:\n{haystack}")


@tvm.testing.skip_if_wheel_test
def test_github_tag_teams(tmpdir_factory):
    """
    Check that individuals are tagged from team headers
    """
    tag_script = REPO_ROOT / "tests" / "scripts" / "github_tag_teams.py"

    def run(source_type, data, check):
        git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
        git.run("init")
        git.run("checkout", "-b", "main")
        git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

        issue_body = """
        some text
        [temporary] opt-in: @person5

        - something: @person1 @person2
        - something3: @person1 @person2 @SOME1-ONE-
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
        @person5
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
            source_type: json.dumps(data),
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
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

        assert_in(check, proc.stdout)

    run(
        source_type="ISSUE",
        data={
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
        check="No one to cc, exiting",
    )

    run(
        source_type="ISSUE",
        data={
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
        check="No one to cc, exiting",
    )

    run(
        source_type="ISSUE",
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
        check="would have updated issues/1234 with {'body': "
        "'\\nhello\\n\\nsomething\\n\\ncc @person1 @person2 @person4'}",
    )

    run(
        source_type="ISSUE",
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
        check="No one to cc, exiting",
    )

    run(
        source_type="ISSUE",
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
        check="would have updated issues/1234 with {'body': "
        "'\\nhello\\n\\nsomething\\n\\ncc @person1 @person2 @person4'}",
    )

    run(
        source_type="ISSUE",
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
        check="No one to cc, exiting",
    )

    run(
        source_type="PR",
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
        check="No one to cc, exiting",
    )

    run(
        source_type="PR",
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

    run(
        source_type="ISSUE",
        data={
            "title": "[something] A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": "`mold` and `lld` can be a much faster alternative to `ld` from gcc. "
            "We should modify our CMakeLists.txt to detect and use these when possible. cc @person1"
            "\n\ncc @person4",
        },
        check="would have updated issues/1234 with {'body': '`mold` and `lld` can be a much"
        " faster alternative to `ld` from gcc. We should modify our CMakeLists.txt to "
        "detect and use these when possible. cc @person1\\n\\ncc @person2 @person4'}",
    )

    run(
        source_type="ISSUE",
        data={
            "title": "[something3] A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [{"name": "something2"}],
            "body": "@person2 @SOME1-ONE-",
        },
        check="Dry run, would have updated issues/1234 with"
        " {'body': '@person2 @SOME1-ONE-\\n\\ncc @person1'}",
    )

    run(
        source_type="ISSUE",
        data={
            "title": "[] A title",
            "number": 1234,
            "user": {
                "login": "person5",
            },
            "labels": [],
            "body": "@person2 @SOME1-ONE-",
        },
        check="No one to cc, exiting",
    )


@parameterize_named(
    dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "abc-abc-123",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "abc-abc-123",
                },
            ]
        },
        expected="Tag names were the same, no update needed",
    ),
    dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "abc-abc-234",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "abc-abc-123",
                },
            ]
        },
        expected="Using tlcpackstaging tag on tlcpack",
    ),
    dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "abc-abc-123",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:01:00.123456Z",
                    "name": "abc-abc-234",
                },
            ]
        },
        expected="Found newer image, using: tlcpack",
    ),
)
def test_open_docker_update_pr(tmpdir_factory, tlcpackstaging_body, tlcpack_body, expected):
    """Test workflow to open a PR to update Docker images"""
    tag_script = REPO_ROOT / "tests" / "scripts" / "open_docker_update_pr.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("init")
    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")
    git.run("checkout", "-b", "main")
    git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")
    images = [
        "ci_lint",
        "ci_gpu",
        "ci_cpu",
        "ci_wasm",
        "ci_i386",
        "ci_qemu",
        "ci_arm",
        "ci_hexagon",
    ]

    docker_data = {}
    for image in images:
        docker_data[f"repositories/tlcpackstaging/{image}/tags"] = tlcpackstaging_body
        docker_data[f"repositories/tlcpack/{image.replace('_', '-')}/tags"] = tlcpack_body

    proc = subprocess.run(
        [
            str(tag_script),
            "--dry-run",
            "--testing-docker-data",
            json.dumps(docker_data),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        cwd=git.cwd,
        env={"GITHUB_TOKEN": "1234"},
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Process failed:\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

    assert_in(expected, proc.stdout)


@pytest.mark.parametrize(
    "images,expected",
    [
        (
            ["ci_arm=tlcpack/ci-arm:abc-abc-123", "ci_lint=tlcpack/ci-lint:abc-abc-234"],
            {
                "ci_arm": "tlcpack/ci-arm:abc-abc-123",
                "ci_lint": "tlcpack/ci-lint:abc-abc-234",
            },
        ),
        (
            ["ci_arm2=tlcpack/ci-arm2:abc-abc-123"],
            {
                "ci_arm2": "tlcpackstaging/ci_arm2:abc-abc-123",
            },
        ),
    ],
)
def test_determine_docker_images(tmpdir_factory, images, expected):
    """Test script to decide whether to use tlcpack or tlcpackstaging for images"""
    tag_script = REPO_ROOT / "tests" / "scripts" / "determine_docker_images.py"

    git_dir = tmpdir_factory.mktemp("tmp_git_dir")

    docker_data = {
        "repositories/tlcpack/ci-arm/tags/abc-abc-123": {},
        "repositories/tlcpack/ci-lint/tags/abc-abc-234": {},
    }

    proc = subprocess.run(
        [
            str(tag_script),
            "--testing-docker-data",
            json.dumps(docker_data),
            "--base-dir",
            git_dir,
        ]
        + images,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        cwd=git_dir,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to run script:\n{proc.stdout}")

    for expected_filename, expected_image in expected.items():
        with open(Path(git_dir) / expected_filename) as f:
            actual_image = f.read()

        assert actual_image == expected_image


@pytest.mark.parametrize(
    "changed_files,name,check,expected_code",
    [
        d.values()
        for d in [
            dict(
                changed_files=[],
                name="abc",
                check="Image abc is not using new naming scheme",
                expected_code=1,
            ),
            dict(
                changed_files=[], name="123-123-abc", check="No extant hash found", expected_code=1
            ),
            dict(
                changed_files=[["test.txt"]],
                name=None,
                check="Did not find changes, no rebuild necessary",
                expected_code=0,
            ),
            dict(
                changed_files=[["test.txt"], ["docker/test.txt"]],
                name=None,
                check="Found docker changes",
                expected_code=2,
            ),
        ]
    ],
)
def test_should_rebuild_docker(tmpdir_factory, changed_files, name, check, expected_code):
    """
    Check that the Docker images are built when necessary
    """
    tag_script = REPO_ROOT / "tests" / "scripts" / "should_rebuild_docker.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("init")
    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")
    git.run("checkout", "-b", "main")
    git.run("remote", "add", "origin", "https://github.com/apache/tvm.git")

    git_path = Path(git.cwd)
    for i, commits in enumerate(changed_files):
        for filename in commits:
            path = git_path / filename
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
            git.run("add", filename)

        git.run("commit", "-m", f"message {i}")

    if name is None:
        ref = "HEAD"
        if len(changed_files) > 1:
            ref = f"HEAD~{len(changed_files) - 1}"
        proc = git.run("rev-parse", ref, stdout=subprocess.PIPE)
        last_hash = proc.stdout.strip()
        name = f"123-123-{last_hash}"

    docker_data = {
        "repositories/tlcpack": {
            "results": [
                {
                    "name": "ci-something",
                },
                {
                    "name": "something-else",
                },
            ],
        },
        "repositories/tlcpack/ci-something/tags": {
            "results": [{"name": name}, {"name": name + "old"}],
        },
    }

    proc = subprocess.run(
        [
            str(tag_script),
            "--testing-docker-data",
            json.dumps(docker_data),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        cwd=git.cwd,
        check=False,
    )

    assert_in(check, proc.stdout)
    assert proc.returncode == expected_code


if __name__ == "__main__":
    tvm.testing.main()
