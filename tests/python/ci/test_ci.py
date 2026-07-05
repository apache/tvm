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
# ruff: noqa: RUF005
"""Test various CI scripts and GitHub Actions workflows"""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

import tvm.testing

from .test_utils import GITHUB_SCRIPT_ROOT, JENKINS_SCRIPT_ROOT, TempGit, run_script


def parameterize_named(**kwargs):
    keys = next(iter(kwargs.values())).keys()
    return pytest.mark.parametrize(
        ",".join(keys), [tuple(d.values()) for d in kwargs.values()], ids=kwargs.keys()
    )


@tvm.testing.skip_if_wheel_test
@parameterize_named(
    cc_no_one=dict(
        pr_body="abc", requested_reviewers=[], existing_review_users=[], expected_reviewers=[]
    ),
    cc_abc=dict(
        pr_body="cc @abc",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc"],
    ),
    bad_cc_line=dict(
        pr_body="cc @", requested_reviewers=[], existing_review_users=[], expected_reviewers=[]
    ),
    cc_multiple=dict(
        pr_body="cc @abc @def",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def"],
    ),
    with_existing=dict(
        pr_body="some text cc @abc @def something else",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def"],
    ),
    with_existing_split=dict(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=[],
        existing_review_users=[],
        expected_reviewers=["abc", "def", "zzz"],
    ),
    with_existing_request=dict(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=["abc"],
        existing_review_users=[],
        expected_reviewers=["def", "zzz"],
    ),
    with_existing_reviewers=dict(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=["abc"],
        existing_review_users=["abc"],
        expected_reviewers=["def", "zzz"],
    ),
    with_no_reviewers=dict(
        pr_body="some text cc @abc @def something else\n\n another cc @zzz z",
        requested_reviewers=[],
        existing_review_users=["abc"],
        expected_reviewers=["def", "zzz"],
    ),
)
def test_cc_reviewers(
    tmpdir_factory, pr_body, requested_reviewers, existing_review_users, expected_reviewers
):
    """
    Test that reviewers are added from 'cc @someone' messages in PRs
    """
    reviewers_script = GITHUB_SCRIPT_ROOT / "github_cc_reviewers.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    reviews = [{"user": {"login": r}} for r in existing_review_users]
    requested_reviewers = [{"login": r} for r in requested_reviewers]
    proc = run_script(
        [reviewers_script, "--dry-run", "--testing-reviews-json", json.dumps(reviews)],
        env={
            "PR": json.dumps(
                {"number": 1, "body": pr_body, "requested_reviewers": requested_reviewers}
            )
        },
        cwd=git.cwd,
    )

    assert f"After filtering existing reviewers, adding: {expected_reviewers}" in proc.stdout


@parameterize_named(
    dont_skip_main=dict(
        commands=[],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    dont_skip_main_with_commit=dict(
        commands=[
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[skip ci] test",
        why="ci should not be skipped on main",
    ),
    skip_on_new_branch=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped on a branch with [skip ci] in the last commit",
    ),
    no_skip_in_pr_title=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
        ],
        should_skip=False,
        pr_title="[no skip ci] test",
        why="ci should not be skipped on a branch with "
        "[skip ci] in the last commit but not the PR title",
    ),
    skip_in_pr_title=dict(
        commands=[
            ["checkout", "-b", "some_new_branch"],
            ["commit", "--allow-empty", "--message", "[skip ci] commit 1"],
            ["commit", "--allow-empty", "--message", "commit 2"],
        ],
        should_skip=True,
        pr_title="[skip ci] test",
        why="ci should be skipped with [skip ci] in the PR title",
    ),
    skip_in_pr_title_many_commits=dict(
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
    skip_anywhere_in_title=dict(
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
    skip_ci_script = JENKINS_SCRIPT_ROOT / "git_skip_ci.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")
    git.run("commit", "--allow-empty", "--message", "base commit")
    for command in commands:
        git.run(*command)
    pr_number = "1234"
    proc = run_script(
        [skip_ci_script, "--pr", pr_number, "--pr-title", pr_title],
        cwd=git.cwd,
        check=False,
    )
    expected = 0 if should_skip else 1
    if proc.returncode != expected:
        raise RuntimeError(
            f"Unexpected return code {proc.returncode} "
            f"(expected {expected}) in {why}:\n{proc.stdout}"
        )


@parameterize_named(
    no_file=dict(files=[], should_skip=True),
    readme=dict(files=["README.md"], should_skip=True),
    c_file=dict(files=["test.c"], should_skip=False),
    c_and_readme=dict(files=["test.c", "README.md"], should_skip=False),
    src_file_and_readme=dict(
        files=["src/autotvm/feature_visitor.cc", "README.md"], should_skip=False
    ),
    yaml_and_readme=dict(files=[".asf.yaml", "docs/README.md"], should_skip=True),
)
def test_skip_globs(tmpdir_factory, files, should_skip):
    """
    Test that CI is skipped if only certain files are edited
    """
    script = JENKINS_SCRIPT_ROOT / "git_skip_ci_globs.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

    proc = run_script(
        [
            script,
            "--files",
            ",".join(files),
        ],
        check=False,
        cwd=git.cwd,
    )

    if should_skip:
        assert proc.returncode == 0
    else:
        assert proc.returncode == 1


def assert_in(needle: str, haystack: str):
    """
    Check that 'needle' is in 'haystack'
    """
    if needle not in haystack:
        raise AssertionError(f"item not found:\n{needle}\nin:\n{haystack}")


@tvm.testing.skip_if_wheel_test
@parameterize_named(
    no_cc=dict(
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
    ),
    no_additional_cc=dict(
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
    ),
    cc_update=dict(
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
    ),
    already_cced=dict(
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
    ),
    not_already_cced=dict(
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
    ),
    no_new_ccs=dict(
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
    ),
    mismatching_tags=dict(
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
    ),
    draft_pr=dict(
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
    ),
    edit_inplace=dict(
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
    ),
    edit_out_of_place=dict(
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
    ),
    atted_but_not_cced=dict(
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
    ),
)
def test_github_tag_teams(tmpdir_factory, source_type, data, check):
    """
    Check that individuals are tagged from team headers
    """
    tag_script = GITHUB_SCRIPT_ROOT / "github_tag_teams.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))

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
    proc = run_script(
        [
            tag_script,
            "--dry-run",
            "--team-issue-json",
            json.dumps(teams),
        ],
        stderr=subprocess.STDOUT,
        cwd=git.cwd,
        env=env,
    )

    assert_in(check, proc.stdout)


@tvm.testing.skip_if_wheel_test
@parameterize_named(
    same_tags=dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "123-123-abc",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "123-123-abc",
                },
            ]
        },
        expected="Tag names were the same, no update needed",
        expected_images=[],
    ),
    staging_update=dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T01:00:00.123456Z",
                    "name": "234-234-abc-staging",
                },
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "456-456-abc",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "123-123-abc",
                },
            ]
        },
        expected="Using tlcpackstaging tag on tlcpack",
        expected_images=[
            "ci_arm: tlcpack/ci-arm:456-456-abc",
        ],
    ),
    tlcpack_update=dict(
        tlcpackstaging_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:00:00.123456Z",
                    "name": "123-123-abc",
                },
            ]
        },
        tlcpack_body={
            "results": [
                {
                    "last_updated": "2022-06-01T00:01:00.123456Z",
                    "name": "234-234-abc",
                },
            ]
        },
        expected="Found newer image, using: tlcpack",
        expected_images=[
            "ci_arm: tlcpack/ci-arm:234-234-abc",
        ],
    ),
)
def test_open_docker_update_pr(
    tmpdir_factory, tlcpackstaging_body, tlcpack_body, expected, expected_images
):
    """Test workflow to open a PR to update Docker images"""
    tag_script = JENKINS_SCRIPT_ROOT / "open_docker_update_pr.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")
    images = [
        "ci_arm",
        "ci_cortexm",
        "ci_cpu",
        "ci_gpu",
        "ci_minimal",
        "ci_riscv",
        "ci_wasm",
    ]

    docker_data = {}
    for image in images:
        docker_data[f"repositories/tlcpackstaging/{image}/tags"] = tlcpackstaging_body
        docker_data[f"repositories/tlcpack/{image.replace('_', '-')}/tags"] = tlcpack_body

    proc = run_script(
        [
            tag_script,
            "--dry-run",
            "--testing-docker-data",
            json.dumps(docker_data),
        ],
        cwd=git.cwd,
        env={"GITHUB_TOKEN": "1234"},
        stderr=subprocess.STDOUT,
    )

    for line in expected_images:
        if line not in proc.stdout:
            raise RuntimeError(f"Missing line {line} in output:\n{proc.stdout}")

    assert_in(expected, proc.stdout)


@parameterize_named(
    use_tlcpack=dict(
        images=["ci_arm", "ci_cpu"],
        expected={
            "ci_arm": "tlcpack/ci-arm:abc-abc-123",
            "ci_cpu": "tlcpack/ci-cpu:abc-abc-234",
        },
    ),
    use_staging=dict(
        images=["ci_arm2"],
        expected={
            "ci_arm2": "tlcpackstaging/ci_arm2:abc-abc-123",
        },
    ),
)
def test_determine_docker_images(tmpdir_factory, images, expected):
    """Test script to decide whether to use tlcpack or tlcpackstaging for images"""
    script = JENKINS_SCRIPT_ROOT / "determine_docker_images.py"

    git_dir = tmpdir_factory.mktemp("tmp_git_dir")

    docker_data = {
        "repositories/tlcpack/ci-arm/tags/abc-abc-123": {},
        "repositories/tlcpack/ci-cpu/tags/abc-abc-234": {},
    }

    images_data = {
        "ci_arm": "tlcpack/ci-arm:abc-abc-123",
        "ci_cpu": "tlcpack/ci-cpu:abc-abc-234",
        "ci_arm2": "tlcpack/ci-arm2:abc-abc-123",
    }

    run_script(
        [
            script,
            "--testing-docker-data",
            json.dumps(docker_data),
            "--testing-images-data",
            json.dumps(images_data),
            "--base-dir",
            git_dir,
        ]
        + images,
        cwd=git_dir,
    )

    for expected_filename, expected_image in expected.items():
        with open(Path(git_dir) / expected_filename) as f:
            actual_image = f.read()

        assert actual_image == expected_image


@parameterize_named(
    invalid_name=dict(
        changed_files=[],
        name="abc",
        check="Image abc is not using new naming scheme",
        expected_code=1,
    ),
    no_hash=dict(
        changed_files=[], name="123-123-abc", check="No extant hash found", expected_code=1
    ),
    no_changes=dict(
        changed_files=[["test.txt"]],
        name=None,
        check="Did not find changes, no rebuild necessary",
        expected_code=0,
    ),
    docker_changes=dict(
        changed_files=[["test.txt"], ["docker/test.txt"]],
        name=None,
        check="Found docker changes",
        expected_code=2,
    ),
)
def test_should_rebuild_docker(tmpdir_factory, changed_files, name, check, expected_code):
    """
    Check that the Docker images are built when necessary
    """
    tag_script = JENKINS_SCRIPT_ROOT / "should_rebuild_docker.py"

    git = TempGit(tmpdir_factory.mktemp("tmp_git_dir"))
    git.run("config", "user.name", "ci")
    git.run("config", "user.email", "email@example.com")

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

    proc = run_script(
        [
            tag_script,
            "--testing-docker-data",
            json.dumps(docker_data),
        ],
        stderr=subprocess.STDOUT,
        cwd=git.cwd,
        check=False,
    )

    assert_in(check, proc.stdout)
    assert proc.returncode == expected_code


if __name__ == "__main__":
    tvm.testing.main()
